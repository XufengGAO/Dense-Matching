r"""Different strategies of weak/strong supervisions"""
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .norm import unit_gaussian_normalize, l1normalize, linearnormalize
import re
from model import rhm_map
from model import geometry

relu = nn.ReLU(inplace=True)
hsfilter = geometry.gaussian2d(7).unsqueeze(0).unsqueeze(0).cuda()

class Corr_calculator:
    """Compute correleation"""
    def __init__(self, args, use_selffeat=False) -> None:
        self.epsilon = args.epsilon
        self.exp2 = args.exp2
        self.loss_stage = args.loss_stage
        self.use_selffeat = use_selffeat
        self.src_size = None
        self.trg_size = None
        self.hpgeometry = None


    def forward(self, cross_sim, src_sim, trg_sim, src_weights, trg_weights, src_imsize, trg_imsize, bsz):
        
        if 'votes' in self.loss_stage:
            self.src_size = (src_feats.size()[0], src_feats.size()[1])
            self.trg_size = (trg_feats.size()[0], src_feats.size()[1])
            cross_sim, src_sim, trg_sim = self.calculate_votes(
                cross_sim, src_sim, trg_sim, src_weights, trg_weights, bsz
            )
            
        if 'geo' in self.loss_stage:
            cross_sim, src_sim, trg_sim = self.calculate_votesGeo(
                cross_sim, src_sim, trg_sim, src_imsize, trg_imsize
            )

        return cross_sim, src_sim, trg_sim

    def calculate_sim(self, src_feats, trg_feats, bsz=1):

        src_feats = F.normalize(src_feats, p=2, dim=2) # [B, HW, C]
        trg_feats = F.normalize(trg_feats, p=2, dim=2)

        cross_sim = relu(torch.bmm(src_feats, trg_feats.transpose(1, 2))) # s->t, [B, HW, HW]
        
        if self.use_selffeat:
            src_sim = relu(torch.bmm(src_feats[:bsz], src_feats[:bsz].transpose(1, 2)))
            trg_sim = relu(torch.bmm(trg_feats[:bsz], trg_feats[:bsz].transpose(1, 2)))   
        else:
            src_sim = None
            trg_sim = None         

        del src_feats, trg_feats

        return cross_sim, src_sim, trg_sim

    def optimal_matching(self, sim, src_size, trg_size, src_weights=None, trg_weights=None):
        costs = 1 - sim

        if src_weights is not None:
            mus = src_weights / src_weights.sum(dim=1).unsqueeze(-1) # normalize weights
        else:
            mus = (torch.ones((src_size[0],src_size[1]))/src_size[1])

        if trg_weights is not None:
            nus = trg_weights / trg_weights.sum(dim=1).unsqueeze(-1)
        else:
            nus = (torch.ones((src_size[0],trg_size[1]))/trg_size[1])

        del src_weights, trg_weights

        ## ---- <Run Optimal Transport Algorithm> ----
        cnt = 0
        votes = []
        for cost, mu, nu in zip(costs, mus, nus):
            while True: # see Algorithm 1
                # PI is optimal transport plan or transport matrix.
                PI = rhm_map.perform_sinkhorn2(cost, self.epsilon, mu.unsqueeze(-1), nu.unsqueeze(-1)) # 4x4096x4096
                
                if not torch.isnan(PI).any():
                    if cnt>0:
                        print(cnt)
                    break
                else: # Nan encountered caused by overflow issue is sinkhorn
                    self.epsilon *= 2.0
                    #print(epsilon)
                    cnt += 1

            #exp2 = 1.0 for spair-71k, TSS
            #exp2 = 0.5 # for pf-pascal and pfwillow
            PI = torch.pow(relu(src_size[1]*PI), self.exp2)

            votes.append(PI.unsqueeze(0))

        del mus, nus, sim, costs, PI

        votes = torch.cat(votes, dim=0)

        return votes
    
    def calculate_votes(self, cross_sim, src_sim, trg_sim, src_weights=None, trg_weights=None):
        cross_votes = self.optimal_matching(cross_sim, self.src_size, self.trg_size, src_weights, trg_weights)
        if self.use_selffeat:
            src_votes = self.optimal_matching(src_sim, self.src_size, self.src_size, src_weights, src_weights)
            trg_votes = self.optimal_matching(trg_sim, self.trg_size, self.trg_size, trg_weights, trg_weights)
        else:
            src_votes = None
            trg_votes = None

        return cross_votes, src_votes, trg_votes

    def calculate_votesGeo(self, cross_votes, src_votes, trg_votes, src_imsize, trg_imsize):
        
        cross_votes_geo = self.rhm(cross_votes, src_imsize, trg_imsize)
        src_votes_geo, trg_votes_geo = None, None
        if src_votes is not None:
            src_votes_geo = self.rhm(src_votes, src_imsize, src_imsize)
        if trg_votes is not None:
            trg_votes_geo = self.rhm(trg_votes, trg_imsize, trg_imsize)        

        return cross_votes_geo, src_votes_geo, trg_votes_geo

    def rhm(self, votes, src_imsize, trg_imsize):
        with torch.no_grad():
            ncells = 8192
            geometric_scores = []
            nbins_x, nbins_y, hs_cellsize = rhm_map.build_hspace(src_imsize, trg_imsize, ncells)
            bin_ids = rhm_map.hspace_bin_ids(src_imsize, self.hpgeometry, self.hpgeometry, hs_cellsize, nbins_x)
            hspace = self.hpgeometry.new_zeros((votes.size()[1], nbins_y * nbins_x))

            hbin_ids = bin_ids.add(torch.arange(0, votes.size()[1]).to(votes.device).
                                mul(hspace.size(1)).unsqueeze(1).expand_as(bin_ids))
            for vote in votes:
                new_hspace = hspace.view(-1).index_add(0, hbin_ids.view(-1), vote.view(-1)).view_as(hspace)
                new_hspace = torch.sum(new_hspace, dim=0).to(votes.device)

                # Aggregate the voting results
                new_hspace = F.conv2d(new_hspace.view(1, 1, nbins_y, nbins_x), hsfilter, padding=3).view(-1)

                geometric_scores.append((torch.index_select(new_hspace, dim=0, index=bin_ids.view(-1)).view_as(vote)).unsqueeze(0))

            geometric_scores = torch.cat(geometric_scores, dim=0)

        votes = votes * geometric_scores
        del nbins_x, nbins_y, hs_cellsize, bin_ids, hspace, hbin_ids, new_hspace, geometric_scores
        
        return votes


class StrongCrossEntropyLoss(nn.Module):
    r"""Strongly-supervised cross entropy loss"""
    def __init__(self, alpha=0.1) -> None:
        super(StrongCrossEntropyLoss, self).__init__()
        assert alpha > 0.0, "negative alpha is not allowed"
        self.softmax = torch.nn.Softmax(dim=1)
        self.alpha = alpha
        self.eps = 1e-30

    def forward(self, x: torch.Tensor, easy_match, hard_match, pckthres, n_pts) -> torch.Tensor:
        loss_buf = x.new_zeros(x.size(0))

        # normalize each row of coefficient, de-mean and unit-std
        x = unit_gaussian_normalize(x)

        for idx, (ct, thres, npt) in enumerate(zip(x, pckthres, n_pts)):

            # Hard (incorrect) match
            if len(hard_match['src'][idx]) > 0:
                cross_ent = self.cross_entropy(ct, hard_match['src'][idx], hard_match['trg'][idx])
                loss_buf[idx] += cross_ent.sum()

            # Easy (correct) match
            if len(easy_match['src'][idx]) > 0:
                cross_ent = self.cross_entropy(ct, easy_match['src'][idx], easy_match['trg'][idx])
                smooth_weight = (easy_match['dist'][idx] / (thres * self.alpha)).pow(2)
                loss_buf[idx] += (smooth_weight * cross_ent).sum()

            loss_buf[idx] /= npt

        return loss_buf.mean()
    
    def cross_entropy(self, correlation_matrix, src_match, trg_match):
        r"""Cross-entropy between predicted pdf and ground-truth pdf (one-hot vector)"""        
        pdf = self.softmax(correlation_matrix.index_select(0, src_match))
        # print("pdf", pdf.size(), trg_match)
        prob = pdf[range(len(trg_match)), trg_match.long()]
        cross_ent = -torch.log(prob + self.eps)

        return cross_ent
    

def mutual_nn_filter(correlation_matrix):
    r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
    corr_src_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
    corr_trg_max = torch.max(correlation_matrix, dim=1, keepdim=True)[0]
    corr_src_max[corr_src_max == 0] += 1e-30
    corr_trg_max[corr_trg_max == 0] += 1e-30

    corr_src = correlation_matrix / corr_src_max
    corr_trg = correlation_matrix / corr_trg_max

    return correlation_matrix * (corr_src * corr_trg)
    
class WeakDiscMatchLoss(nn.Module):
    r"""Weakly-supervised discriminative and maching loss"""
    def __init__(self, temp=1.0, weak_lambda=None) -> None:
        super(WeakDiscMatchLoss, self).__init__()
        self.temp = temp
        weak_lambda = list(map(float, re.findall(r"[-+]?(?:\d*\.*\d+)", weak_lambda)))
        self.weak_lambda = [i>0.0 for i in weak_lambda]

    def forward(self, corr: torch.Tensor, src_feats: torch.Tensor, trg_feats: torch.Tensor, bsz, num_negatives=0) -> torch.Tensor:

        if self.weak_lambda[0]:
            entropy_loss_pos = self.information_entropy(corr[:bsz])
            if num_negatives > 0:
                entropy_loss_neg = self.information_entropy(corr[bsz:])
            else:
                entropy_loss_neg = 1.0

            entropy_loss = entropy_loss_pos / entropy_loss_neg
        else:
            entropy_loss = torch.zeros(1).cuda()

        if self.weak_lambda[1]:
            match_loss = self.information_match(corr[:bsz], src_feats[:bsz], trg_feats[:bsz])
        else:
            match_loss = torch.zeros(1).cuda()
                    
        loss = torch.stack([entropy_loss, match_loss])
        return loss


    def information_entropy(self, corr: torch.Tensor):
        r"""Computes information entropy of all candidate matches"""

        #TODO: other norm is possible, softmax with t, [B, HW, HW]
        B, HW0, HW1 = corr.size()[0], corr.size()[1], corr.size()[2]
        src_pdf = l1normalize(corr, dim=2)
        trg_pdf = l1normalize(corr.transpose(1,2), dim=2)

        # [B, HW, HW]
        entropy_loss = torch.sum((-1.*src_pdf) * torch.log(src_pdf+1e-8))/(B*HW0) + \
                       torch.sum((-1.*trg_pdf) * torch.log(trg_pdf+1e-8))/(B*HW1)

        return entropy_loss

    
    def information_match(self, corr: torch.Tensor, src_feats: torch.Tensor, trg_feats: torch.Tensor):

        B, HW0, HW1 = src_feats.size()[0], src_feats.size()[1], trg_feats.size()[2]
        src_feats = F.normalize(src_feats, p=2, dim=2) # [B, HW0, C]
        trg_feats = F.normalize(trg_feats, p=2, dim=2)
        
        # corr = [B, HW0, HW1]
        s_corr = F.softmax(corr/self.temp, dim=2)
        t_corr = F.softmax(corr.permute(0,2,1)/self.temp, dim=2)

        src2trg = src_feats - torch.bmm(s_corr, trg_feats) # [B, HW0, C]
        trg2src = trg_feats - torch.bmm(t_corr, src_feats) # [B, HW1, C]
        
        match_loss = torch.sum(torch.norm(src2trg, dim=2))/(B*HW0) + \
                     torch.sum(torch.norm(trg2src, dim=2))/(B*HW1)

        return match_loss


class StrongFlowLoss(nn.Module):
    r"""Strongly-supervised flow loss"""
    def __init__(self) -> None:
        super(StrongFlowLoss, self).__init__()
        self.eps = 1e-30

    def forward(self, x: torch.Tensor, flow_gt, feat_size):
        r"""Strongly-supervised matching loss (L_{match})"""
        
        B = x.size()[0]
        grid_x, grid_y = soft_argmax(x.view(B, -1, feat_size, feat_size), feature_size=feat_size)

        pred_flow = torch.cat((grid_x, grid_y), dim=1)
        pred_flow = unnormalise_and_convert_mapping_to_flow(pred_flow)

        loss_flow = EPE(pred_flow, flow_gt)
        return loss_flow
    
def soft_argmax(self, corr, beta=0.02, feature_size=64):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    x_normal = nn.Parameter(torch.tensor(np.linspace(-1,1,feature_size), dtype=torch.float, requires_grad=False)).cuda()
    y_normal = nn.Parameter(torch.tensor(np.linspace(-1,1,feature_size), dtype=torch.float, requires_grad=False)).cuda()

    b,_,h,w = corr.size()
    
    corr = softmax_with_temperature(corr, beta=beta, d=1)
    corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

    grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
    x_normal = x_normal.expand(b,w)
    x_normal = x_normal.view(b,w,1,1)
    grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    
    grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
    y_normal = y_normal.expand(b,h)
    y_normal = y_normal.view(b,h,1,1)
    grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    return grid_x, grid_y

def EPE(input_flow, target_flow, sparse=True, mean=True, sum=False):

    EPE_map = torch.norm(target_flow-input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/torch.sum(~mask)

def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:,0,:,:] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
    mapping[:,1,:,:] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow

def softmax_with_temperature(x, beta, d = 1):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M # subtract maximum value for stability
    exp_x = torch.exp(x/beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum

    

