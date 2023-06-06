import math
import torch
import torch.nn.functional as F
from . import geometry


class PostProcess:
    r"""Optimal matching and Regularized Hough matching algorithm"""
    def __init__(self, rfsz, jsz, feat_h, feat_w, device, img_side, ncells=8192):
        r"""Constructor of HoughMatching"""
        super(PostProcess, self).__init__()

        self.rf = geometry.receptive_fields(rfsz, jsz, feat_h, feat_w).to(device)
        self.rf_center = geometry.center(self.rf)

        self.num_feat = feat_h * feat_w
        
        self.nbins_x, self.nbins_y, hs_cellsize = self.build_hspace(img_side, img_side, ncells)
        self.bin_ids = self.compute_bin_id(img_side, self.rf, self.rf, hs_cellsize, self.nbins_x)
        self.hspace = self.rf.new_zeros((len(self.rf), self.nbins_y * self.nbins_x))
        self.hbin_ids = self.bin_ids.add(torch.arange(0, len(self.rf)).to(device).
                                         mul(self.hspace.size(1)).unsqueeze(1).expand_as(self.bin_ids))
        self.hsfilter = geometry.gaussian2d(7).to(device)
    
    def rhm(self, corr):
        r"""Regularized Hough matching"""
        hspace = self.hspace.view(-1).index_add(0, self.hbin_ids.view(-1), corr.view(-1)).view_as(self.hspace)
        hspace = torch.sum(hspace, dim=0)
        hspace = F.conv2d(hspace.view(1, 1, self.nbins_y, self.nbins_x),
                          self.hsfilter.unsqueeze(0).unsqueeze(0), padding=3).view(-1)

        return torch.index_select(hspace, dim=0, index=self.bin_ids.view(-1)).view_as(corr)
    
    def return_weight(self, mask):
        # print(mask.size()) # [32, 200, 300])
        
        hselect = mask[:, self.rf_center[:,1].long(),self.rf_center[:,0].long()]
        weights = 0.5*torch.ones(hselect.size()).cuda()
        scale = 1.0
        weights[hselect>0.4*scale] = 0.8
        weights[hselect>0.5*scale] = 0.9
        weights[hselect>0.6*scale] = 1.0

        return weights/weights.sum(dim=1).unsqueeze(-1) # B, num_feat
        
    def optmatch(self, costs, src_masks, trg_masks, epsilon, exp2):
        
        mus = self.return_weight(src_masks) # normalize weight
        nus = self.return_weight(trg_masks)

        ## ---- <Run Optimal Transport Algorithm> ----
        cnt = 0
        votes = []

        for cost, mu, nu in zip(costs, mus, nus):
            while True: # see Algorithm 1
                # PI is optimal transport plan or transport matrix.
                PI = perform_sinkhorn(cost, epsilon, mu.unsqueeze(-1), nu.unsqueeze(-1)) # 4x4096x4096
                
                if not torch.isnan(PI).any():
                    if cnt>0:
                        print(cnt)
                    break
                else: # Nan encountered caused by overflow issue is sinkhorn
                    epsilon *= 2.0
                    #print(epsilon)
                    cnt += 1

            #exp2 = 1.0 for spair-71k, TSS
            #exp2 = 0.5 # for pf-pascal and pfwillow
            PI = torch.pow(torch.clamp(self.num_feat*PI, min=0), exp2)
            
            votes.append(PI.unsqueeze(0))

        return torch.cat(votes, dim=0)

    def compute_bin_id(self, src_imsize, src_box, trg_box, hs_cellsize, nbins_x):
        r"""Computes Hough space bin ids for voting"""
        src_ptref = torch.tensor(src_imsize, dtype=torch.float).to(src_box.device)
        src_trans = geometry.center(src_box)
        trg_trans = geometry.center(trg_box)
        xy_vote = (src_ptref.unsqueeze(0).expand_as(src_trans) - src_trans).unsqueeze(2).\
                      repeat(1, 1, len(trg_box)) + trg_trans.t().unsqueeze(0).repeat(len(src_box), 1, 1)

        bin_ids = (xy_vote / hs_cellsize).long()
        del src_ptref

        return bin_ids[:, 0, :] + bin_ids[:, 1, :] * nbins_x

    def build_hspace(self, src_imsize, trg_imsize, ncells):
        r"""Build Hough space"""
        hs_width = src_imsize[0] + trg_imsize[0]
        hs_height = src_imsize[1] + trg_imsize[1]
        hs_cellsize = math.sqrt((hs_width * hs_height) / ncells)
        nbins_x = int(hs_width / hs_cellsize) + 1
        nbins_y = int(hs_height / hs_cellsize) + 1

        return nbins_x, nbins_y, hs_cellsize






def perform_sinkhorn(C,epsilon,mu,nu,a=[],warm=False,niter=1,tol=10e-9):
    """Main Sinkhorn Algorithm"""
    if not warm:
        a = torch.ones((C.shape[0],1)).cuda() / C.shape[0]
    
    
    K = torch.exp(-C/epsilon)
    # print(K.size(), nu.size(), a.size())

    Err = torch.zeros((niter,2)).cuda()
    for i in range(niter):
        b = nu/torch.mm(K.t(), a)

        if i%2==0:
            Err[i,0] = torch.norm(a*(torch.mm(K, b)) - mu, p=1)
            if i>0 and (Err[i,0]) < tol:
                break

        a = mu / torch.mm(K, b)

        if i%2==0:
            Err[i,1] = torch.norm(b*(torch.mm(K.t(), a)) - nu, p=1)
            if i>0 and (Err[i,1]) < tol:
                break

        PI = torch.mm(torch.mm(torch.diag(a[:,-1]),K), torch.diag(b[:,-1]))

    del a; del b; del K
    return PI


























