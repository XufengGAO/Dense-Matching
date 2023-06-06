"""Implementation of : Semantic Correspondence as an Optimal Transport Problem"""

from functools import reduce
from operator import add
import torch.nn.functional as F
import torch
from . import resnet
import torch.nn as nn
from .custom_modules import DynamicFeatureSelection
from contextlib import nullcontext
from tools.logger import Logger

class Model(nn.Module):
    r"""SCOT framework"""
    def __init__(self, model, loss):
        r"""Constructor for SCOT framework"""
        super(Model, self).__init__()

        Logger.info(f">>>>>>>>>> Creating model:{model.backbone.type}{model.backbone.depth} + {model.backbone.pretrain} <<<<<<<<<<")
        str_layer = ', '.join(str(i) for i in model.backbone.layers)
        Logger.info(f'>>>>>>> Use {len(model.backbone.layers)} layers: {str_layer} <<<<<<<<<<')

        self.freeze = model.backbone.freeze

        # 1. Backbone
        if model.backbone.type == 'resnet':
            if model.backbone.depth == 50:
                self.backbone = resnet.resnet50(pretrained=True)
                nbottlenecks = [3, 4, 6, 3]
                self.in_channels = [64, 256, 256, 256, 512, 512, 512, 512, 1024,
                                    1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048]
                self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 32, 32, 32])
                self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139, 171, 203, 235, 267, 299, 363, 427])
                
            elif model.backbone.depth == 101:
                self.backbone = resnet.resnet101(pretrained=True)
                nbottlenecks = [3, 4, 23, 3]
                self.in_channels = [64, 256, 256, 256, 512, 512, 512, 512,
                                    1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                    1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                    1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                    1024, 1024, 2048, 2048, 2048]                
                self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 16, 16, \
                                        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, \
                                        16, 16, 16, 16, 32, 32, 32])
                self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139,\
                                        171, 203, 235, 267, 299, 331, 363, 395, 427, 459, 491, 523, 555, 587,\
                                        619, 651, 683, 715, 747, 779, 811, 843, 907, 971])
            
            else:
                raise Exception('Unavailable depth')
            
            if len(model.backbone.cam) > 0:
                print('Use Identity in FC layer')
                self.backbone.fc = nn.Identity()
            self.cam = model.backbone.cam 

        else:
            raise Exception('Unavailable backbone: %s' % model.backbone.type)
        
        self.backbone.eval()

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.layers = model.backbone.layers

        # 2. Miscellaneous
        self.img_side, self.feat_h, self.feat_w = None, None, None
        self.channels = [self.in_channels[i] for i in self.layers]
        self.jsz = self.jsz[self.layers[0]]
        self.rfsz = self.rfsz[self.layers[0]]
        self.set_geometry = False

        # 3. Custom modules
        # dynamic feature selection
        if model.use_neck:
            assert model.neck.D >= 1, 'D should be >= 1'
            self.neck = DynamicFeatureSelection(self.channels, len(self.layers), model.neck)

            # correlation projection
            if model.neck.D > 1:
                self.corr_projector = nn.Sequential(
                    torch.nn.Conv2d(model.neck.D, 1, (1,1)),
                    nn.ReLU(inplace=True),
                )
                self.init_type = model.neck.init_type
                self.corr_projector.apply(self.init_projector)
            else:
                self.corr_projector = None
        else:
            self.neck = None

        # 4. Feature projectors
        if model.use_head:
            in_dim = sum(self.channels)
            self.feat_projector = torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, model.head.embed_dim, (1,1)),
                nn.ReLU(inplace=True),
            )
            self.init_type = model.head.init_type
            self.feat_projector.apply(self.init_projector)
        else:
            self.feat_projector = None

        # 5. loss
        self.match_layers = []
        if loss.type == 'weak':
            if loss.match_loss_weight>0.0 and len(loss.match_layers)>0:
                for i in loss.match_layers:
                    self.match_layers.append(self.layers.index(i))


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

    def forward(self, imgs, return_mask=False):
        r"""Forward pass"""
        # feature extraction
        with torch.no_grad() if self.freeze else nullcontext():
            b = imgs.size()[0]  # src + trg
            # not return fc for training
            # return fc for testing
            feats, feat_map, fc = self.extract_feats(imgs, return_fc=return_mask, return_feats=True)
            src_f, trg_f = [f[0:b//2] for f in feats], [f[b//2:] for f in feats]
            base_feat_size = tuple(feats[0].size()[2:])

            # Original features
            src_feat = [F.interpolate(src_f[i], size=base_feat_size, mode='bilinear', align_corners=True) for i in range(len(self.layers))]
            trg_feat = [F.interpolate(trg_f[i], size=base_feat_size, mode='bilinear', align_corners=True) for i in range(len(self.layers))]

            del feats, src_f, trg_f

            if not self.set_geometry:
                self.feat_h, self.feat_w = int(base_feat_size[0]), int(base_feat_size[1])
                self.set_geometry = True
                self.img_side = imgs.size()[2:][::-1] # (w,h)

            if len(self.match_layers)>0:
                match_src_feat = torch.cat([src_feat[i] for i in self.match_layers], dim=1)  # list of [B, Ci, H, W] -> [B, C, H, W]
                match_trg_feat = torch.cat([trg_feat[i] for i in self.match_layers], dim=1)
                # [B, HW, C]
                match_src_feat = match_src_feat.view(match_src_feat.size()[0], match_src_feat.size()[1], -1).permute(0, 2, 1)
                match_trg_feat = match_trg_feat.view(match_trg_feat.size()[0], match_trg_feat.size()[1], -1).permute(0, 2, 1)
            else:
                match_src_feat, match_trg_feat = None, None

        # dynamic feature selection
        if self.neck is not None:
            src_feat = self.neck(src_feat)
            trg_feat = self.neck(trg_feat) # [B, D, C, H, W]
        else:
            src_feat = torch.cat(src_feat, dim=1)  # list of [B, Ci, H, W] -> [B, C, H, W]
            trg_feat = torch.cat(trg_feat, dim=1)

        #FIXME: projection, if you this part, the following code should be modified accordingly
        # if self.feat_projector is not None:
        #     proj = self.feat_projector(torch.cat([src_feat, trg_feat],dim=0)) # [2B, C, H, W] -> [2B, D, H, W]
        #     b = proj.size()[0]
        #     src_proj, trg_proj = proj[0:b//2], proj[b//2:] # [B, D, H, W]

        if src_feat.dim() == 4: # [B, C, H, W]
            src_feat = src_feat.unsqueeze(1)
            trg_feat = trg_feat.unsqueeze(1) # [B, D, C, H, W]

        # shape = [B, D, C, H, W] -> [B, D, C, HW] -> [B, D, HW, C]
        src_feat = src_feat.view(src_feat.size()[0], src_feat.size()[1], src_feat.size()[2], -1).transpose(2, 3)
        trg_feat = trg_feat.view(trg_feat.size()[0], trg_feat.size()[1], trg_feat.size()[2], -1).transpose(2, 3)

        # 3. Calculate Corr, [B, D, HW, HW] 
        sim = self.calculate_sim(src_feat, trg_feat)

        #FIXME, if use projector, projects to [B, 1, HW0, HW1]
        if self.corr_projector is not None:
            sim = self.corr_projector(sim)
        
        assert sim.size()[1] == 1, 'Fault dim'
        sim = sim.squeeze(1) # [B, HW0, HW1]

        del src_feat, trg_feat

        #TODO remove negaive
        if return_mask:
            bsz, sz = imgs.size()[0], (imgs.size()[2], imgs.size()[3])
            masks = [self.get_CAM_multi(imgs[i].unsqueeze(0), feat_map[i].unsqueeze(0), fc[i].unsqueeze(0), sz, top_k=2) for i in range(bsz)]
            src_m, trg_m = torch.cat(masks[0:bsz//2], dim=0), torch.cat(masks[bsz//2:], dim=0)

            return match_src_feat, match_trg_feat, sim, src_m, trg_m
        else:
            #TODO, may return projected features
            return match_src_feat, match_trg_feat, sim

    def calculate_sim(self, src_feats, trg_feats):

        src_feats = F.normalize(src_feats, p=2, dim=3) # [B, D, HW0, C]
        trg_feats = F.normalize(trg_feats, p=2, dim=3)

        # cross_sim = self.relu(torch.bmm(src_feats, trg_feats.transpose(1, 2))) # s->t, [B, HW, HW]
        # s->t, [B, D, HW0, HW1]
        sim = F.relu(torch.einsum('bdlc,bdcL->bdlL', src_feats, trg_feats.transpose(2,3)), inplace=True)
          

        del src_feats, trg_feats

        return sim
    
    def extract_feats(self, img, return_fc=False, return_feats=True):
        r"""Extract a list of intermediate features 
        """

        feats = []

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.layers and return_feats:
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

            if hid + 1 in self.layers and return_feats:
                feats.append(feat.clone())

            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        # only return feats
        if return_feats and not return_fc:
            return feats, None, None
        
        # Global Average Pooling feature map
        feat_map = feat  # feature map before gloabl avg-pool

        x = self.backbone.avgpool(feat)
        x = torch.flatten(x, 1)
        fc = self.backbone.fc(x)  # fc output

        if not return_feats and return_fc:
            return feat_map, fc

        if return_feats and return_fc:
            return feats, feat_map, fc


    def get_CAM_multi(self, img, feat_map, fc, sz, top_k=2):
        scales = [1.0,1.5,2.0]
        map_list = []
        for scale in scales:
            if scale>1.0:
                if scale*scale*sz[0]*sz[1] > 800*800:
                    scale = min(800/sz[0],800/sz[1])
                    scale = min(1.5,scale)
                img = F.interpolate(img, (int(scale*sz[0]),int(scale*sz[1])), None, 'bilinear', True) # 1x3xHxW
                feat_map, fc = self.extract_feats(img, return_fc=True, return_feats=False)

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

        return norm_cam.unsqueeze(0)
    

    
    def load_backbone(self, state_dict, strict=False):
        self.backbone.load_state_dict(state_dict, strict=strict)

