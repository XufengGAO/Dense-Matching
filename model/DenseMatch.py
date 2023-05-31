"""Implementation of : Semantic Correspondence as an Optimal Transport Problem"""

from functools import reduce
from operator import add
import torch.nn.functional as F
import torch
from . import geometry
from . import resnet
import torch.nn as nn
from .custom_modules import DynamicFeatureSelection

class Model(nn.Module):
    r"""SCOT framework"""
    def __init__(self, args, layers):
        r"""Constructor for SCOT framework"""
        super(Model, self).__init__()

        # 1. Backbone
        if args.backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            nbottlenecks = [3, 4, 6, 3]
            self.in_channels = [64, 256, 256, 256, 512, 512, 512, 512, 1024,
                                1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048]
        elif args.backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            nbottlenecks = [3, 4, 23, 3]
            self.in_channels = [64, 256, 256, 256, 512, 512, 512, 512,
                                1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                1024, 1024, 2048, 2048, 2048]
        else:
            raise Exception('Unavailable backbone: %s' % args.backbone)
        
        if len(args.cam) > 0: 
            print('Use Identity in FC layer')
            self.backbone.fc = nn.Identity()
        self.backbone.eval()

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.layers = layers

        if args.backbone in ['resnet50']:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 32, 32, 32])
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139, 171, 203, 235, 267, 299, 363, 427])
        else: 
            # args.backbone in ['resnet101']
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 16, 16, \
                                     16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, \
                                     16, 16, 16, 16, 32, 32, 32])
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139,\
                                      171, 203, 235, 267, 299, 331, 363, 395, 427, 459, 491, 523, 555, 587,\
                                      619, 651, 683, 715, 747, 779, 811, 843, 907, 971])

        # 2. Miscellaneous
        self.relu = nn.ReLU(inplace=True)

        # set geometry
        self.img_side, self.feat_h, self.feat_w = None, None, None
        self.channels = [self.in_channels[i] for i in self.layers]
        self.jsz = self.jsz[self.layers[0]]
        self.rfsz = self.rfsz[self.layers[0]]
        self.set_geometry = False

        # 3. Custom modules
        # dynamic feature selection
        self.learner = DynamicFeatureSelection(self.channels, len(self.layers), args.init_type, args.w_group, args.use_mp)

        if args.w_group > 1:
            self.corr_projector = torch.nn.Conv2d(args.w_group, 1, (1,1))
        else:
            self.corr_projector = None

        # Projectors
        # feature projection
        self.init_type = args.init_type
        if args.use_feat_project:
            in_dim = sum(self.channels)
            self.feat_projector = torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, args.embed_dim, (1,1)),
                self.relu,
            )
            self.feat_projector.apply(self.init_projector)
        else:
            self.feat_projector = None
        
        # correlation projection
        if args.w_group > 1:
            self.corr_projector = nn.Sequential(
                torch.nn.Conv2d(args.w_group, 1, (1,1)),
                self.relu,
            )
            self.corr_projector.apply(self.init_projector)
        else:
            self.corr_projector = None

        

    def init_projector(self, layer):
        if isinstance(layer, nn.Conv2d):
            if self.init_type == 'kaiming_norm':
                nn.init.kaiming_normal_(layer.weight.data)
            elif self.init_type == 'xavier_norm':
                nn.init.xavier_normal_(layer.weight.data,)
            else:
                nn.init.uniform_(layer.weight.data)

            if layer.bias is not None:
                layer.bias.data.zero_()

    def get_geo_info(self):
        return self.rfsz, self.jsz, self.feat_h, self.feat_w, self.img_side

    def forward(self, imgs, bsz):
        r"""Forward pass"""
        # feature extraction
        b = imgs.size()[0]  # src + trg
        with torch.no_grad():
            feats, feat_map, fc = self.extract_feats(imgs, return_fc=False)
            src_f, trg_f = [f[0:b//2] for f in feats], [f[b//2:] for f in feats]
            base_feat_size = tuple(feats[0].size()[2:])
            src_feat = [F.interpolate(src_f[i], size=base_feat_size, mode='bilinear', align_corners=False) for i in range(len(self.layers))]
            trg_feat = [F.interpolate(trg_f[i], size=base_feat_size, mode='bilinear', align_corners=False) for i in range(len(self.layers))]

            if not self.set_geometry:
                self.feat_h, self.feat_w = int(base_feat_size[0]), int(base_feat_size[1])
                self.set_geometry = True
                self.img_side = imgs.size()[2:][::-1] # (w,h)

        # dynamic feature selection
        if self.learner is not None:
            if self.learner.use_mp:
                src_feat = torch.cat(src_feat, dim=1)  # list of [B, Ci, H, W] -> [B, C, H, W]
                trg_feat = torch.cat(trg_feat, dim=1)
            src_feat = self.learner(src_feat)
            trg_feat = self.learner(trg_feat)

        #FIXME: projection, if you this part, the following code should be modified accordingly
        # if self.feat_projector is not None:
            # proj = self.feat_projector(torch.cat([src_feat, trg_feat],dim=0)) # [2B, C, H, W] -> [2B, D, H, W]
            # src_proj, trg_proj = proj[0:b//2], proj[b//2:] # [B, D, H, W]

        # shape = [B, D, C, H, W] -> [B, D, C, HW] -> [B, D, HW, C]
        src_feat = src_feat.view(src_feat.size()[0], src_feat.size()[1], src_feat.size()[2], -1).transpose(2, 3)
        trg_feat = trg_feat.view(trg_feat.size()[0], trg_feat.size()[1], trg_feat.size()[2], -1).transpose(2, 3)

        # 3. Calculate Corr, [B, D, HW, HW] 
        sim = self.calculate_sim(src_feat, trg_feat, bsz)

        #FIXME, if use projector, projects to [B, HW, HW]
        if self.corr_projector is not None:
            sim = self.corr_projector(sim)
        
        assert sim.size()[1] == 1, 'Fault dim'
        sim = sim.squeeze(1)

        
        #TODO: remove weight during training
        # CAM weights pixel importance, shape = [B, HW]
        # if self.classmap in [1]: 
        #     if masks is None:
        #         masks = self.get_CAM_multi(imgs, feat_map, fc, sz=(imgs.size(2),imgs.size(3)), top_k=2)                 
        #     scale = 1.0
        #     del feat_map, fc
            
        #     hselect = masks[:, self.hpos[:,1].long(), self.hpos[:,0].long()].to(self.hpos.device)

        #     weights = 0.5*torch.ones(hselect.size()).to(self.hpos.device)
        #     weights[hselect>0.4*scale] = 0.8
        #     weights[hselect>0.5*scale] = 0.9
        #     weights[hselect>0.6*scale] = 1.0

        #     del hselect
        # else:
        #     HW = src_proj.size()[2]
        #     weights = torch.ones(b, HW).to(src_feat.device) # [B, HW]

        return src_feat, trg_feat, sim

    def calculate_sim(self, src_feats, trg_feats):

        src_feats = F.normalize(src_feats, p=2, dim=3) # [B, D, HW, C]
        trg_feats = F.normalize(trg_feats, p=2, dim=3)

        # cross_sim = self.relu(torch.bmm(src_feats, trg_feats.transpose(1, 2))) # s->t, [B, HW, HW]
        # s->t, [B, D, HW, HW]
        sim = self.relu(torch.einsum('bdlc,bdcL->bdlL', src_feats, trg_feats.transpose(2,3)))
          

        del src_feats, trg_feats

        return sim
    
    def extract_feats(self, img, return_fc=False):
        r"""Extract a list of intermediate features 
        """

        feats = []

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.layers:
            feats.append(feat)

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)
            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)
            feat += res

            if hid + 1 in self.layers:
                feats.append(feat)

            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        if return_fc:
            # Global Average Pooling feature map
            feat_map = feat  # feature map before gloabl avg-pool

            x = self.backbone.avgpool(feat)
            x = torch.flatten(x, 1)
            fc = self.backbone.fc(x)  # fc output
        else:
            feat_map = None
            fc = None

        return feats, feat_map, fc

    def get_CAM(self, feat_map, fc, sz, top_k=2):
        logits = F.softmax(fc, dim=1)
        scores, pred_labels = torch.topk(logits, k=top_k, dim=1)

        pred_labels = pred_labels[0]
        bz, nc, h, w = feat_map.size()

        output_cam = []
        for label in pred_labels:
            cam = self.backbone.fc.weight[label,:].unsqueeze(0).mm(feat_map.view(nc,h*w))
            cam = cam.view(1,1,h,w)
            cam = F.interpolate(cam, (sz[0],sz[1]), None, 'bilinear', True)[0,0] # HxW
            cam = (cam-cam.min()) / cam.max()
            output_cam.append(cam)
        output_cam = torch.stack(output_cam,dim=0) # kxHxW
        output_cam = output_cam.max(dim=0)[0] # HxW

        return output_cam

    def get_CAM_multi2(self, img, feat_map, fc, sz, top_k=2):
        scales = [1.0,1.5,2.0]
        map_list = []
        for scale in scales:
            if scale>1.0:
                if scale*scale*sz[0]*sz[1] > 800*800:
                    scale = min(800/sz[0],800/sz[1])
                    scale = min(1.5,scale)
                img = F.interpolate(img, (int(scale*sz[0]),int(scale*sz[1])), None, 'bilinear', True) # 1x3xHxW
                feat_map, fc = self.extract_feats(img,return_hp=False)

            logits = F.softmax(fc, dim=1)
            scores, pred_labels = torch.topk(logits, k=top_k, dim=1)
            pred_labels = pred_labels[0]
            bz, nc, h, w = feat_map.size()

            output_cam = []
            for label in pred_labels:
                cam = self.backbone.fc.weight[label,:].unsqueeze(0).mm(feat_map.view(nc,h*w))
                cam = cam.view(1,1,h,w)
                cam = F.interpolate(cam, (sz[0],sz[1]), None, 'bilinear', True)[0,0] # HxW
                cam = (cam-cam.min()) / cam.max()
                output_cam.append(cam)
            output_cam = torch.stack(output_cam,dim=0) # kxHxW
            output_cam = output_cam.max(dim=0)[0] # HxW
            
            map_list.append(output_cam)
        map_list = torch.stack(map_list,dim=0)
        sum_cam = map_list.sum(0)
        norm_cam = sum_cam / (sum_cam.max()+1e-5)

        return norm_cam
    
    def get_CAM_multi(self, img, feat_map, fc, sz, top_k=2):
        # img = Bx3x256x256
        # featmap = Bx2048x8x8
        # fc = Bx1000
        # sz = 256x256
        scales = [1.0,1.5,2.0]
        map_list = []
        for scale in scales:
            if scale>1.0:
                if scale*scale*sz[0]*sz[1] > 800*800:
                    scale = min(800/sz[0],800/sz[1])
                    scale = min(1.5,scale)
                img = F.interpolate(img, (int(scale*sz[0]),int(scale*sz[1])), None, 'bilinear', True) # Bx3xHxW
                feat_map, fc = self.extract_feats(img,return_hp=False)

            logits = F.softmax(fc, dim=1)
            _, pred_labels = torch.topk(logits, k=top_k, dim=1) # Bx2
            bz, nc, h, w = feat_map.size()
            output_cam = []

            # print(self.backbone.fc.weight.size()) # 1000x2048
            cam = self.backbone.fc.weight[pred_labels,:].bmm(feat_map.view(bz, nc, h*w)) # Bx2048x64
            cam = cam.view(bz,-1,h,w) # Bx2x8x8
            cam = F.interpolate(cam, (sz[0],sz[1]), None, 'bilinear', True) # Bx2x240x240
            
            cam_min, _ = torch.min(cam.view(bz, top_k, -1), dim=-1, keepdim=True) #Bx2x1
            cam_max, _ = torch.max(cam.view(bz, top_k, -1), dim=-1, keepdim=True)
            cam_min = cam_min.unsqueeze(-1)  #Bx2x1x1
            cam_max = cam_max.unsqueeze(-1)
            cam = (cam-cam_min)/cam_max # Bx2x240x240
            output_cam = cam.max(dim=1)[0] # Bx240x240
            map_list.append(output_cam)

            del output_cam, cam

        map_list = torch.stack(map_list,dim=0) # 3xBx240x240
        sum_cam = map_list.sum(dim=0) # Bx240x240
        sum_cam_max = sum_cam.view(bz,-1).max(dim=-1,keepdim=True)[0].unsqueeze(-1)
        norm_cam = sum_cam / (sum_cam_max+1e-10) # Bx240x240
        # print(map_list.size(), sum_cam.size(), sum_cam_max.size(), norm_cam.size())
        # transform = T.ToPILImage()
        # for idx, outputcam in enumerate(norm_cam):
        #     imgm = transform(outputcam)
        #     file_name = "{}".format(idx)
        #     imgm.save("/home/xufeng/Documents/EPFL_Course/sp_code/SCOT/img/{}.png".format(file_name))
        del map_list, sum_cam, sum_cam_max

        return norm_cam

    def get_FCN_map(self, img, feat_map, fc, sz):
        #scales = [1.0,1.5,2.0]
        scales = [1.0]
        map_list = []
        for scale in scales:
            if scale*scale*sz[0]*sz[1] > 1200*800:
                scale = 1.5
            img = F.interpolate(img, (int(scale*sz[0]),int(scale*sz[1])), None, 'bilinear', True) # 1x3xHxW
            #feat_map, fc = self.extract_intermediate_feat(img,return_hp=False,backbone='fcn101')
            feat_map = self.backbone1.evaluate(img)
            
            predict = torch.max(feat_map, 1)[1]
            mask = predict-torch.min(predict)
            mask_map = mask / torch.max(mask)
            mask_map = F.interpolate(mask_map.unsqueeze(0).double(), (sz[0],sz[1]), None, 'bilinear', True)[0,0] # HxW
    
        return mask_map
    

    def state_dict(self):
        return self.learner.state_dict()
    
    def load_backbone(self, state_dict, strict=False):
        self.backbone.load_state_dict(state_dict, strict=strict)
