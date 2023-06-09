r"""Different strategies of weak/strong supervisions"""
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .norm import unit_gaussian_normalize, l1normalize
import re



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
    
class WeakCEMatchLoss(nn.Module):
    r"""Weakly-supervised discriminative and maching loss"""
    def __init__(self, loss) -> None:
        super(WeakCEMatchLoss, self).__init__()
        self.t = loss.temp
        self.ce_w = loss.ce_loss_weight
        self.match_w = loss.match_loss_weight

    def forward(self, corr: torch.Tensor, src_feats: torch.Tensor, trg_feats: torch.Tensor, bsz, num_negatives=0) -> torch.Tensor:

        if self.ce_w>1e-15:
            entropy_loss_pos = self.information_entropy(corr[:bsz])
            if num_negatives > 0:
                entropy_loss_neg = self.information_entropy(corr[bsz:])
            else:
                entropy_loss_neg = 1.0

            entropy_loss = entropy_loss_pos / entropy_loss_neg
        else:
            entropy_loss = torch.zeros(1).cuda()

        if self.match_w>1e-15:
            match_loss = self.information_match(corr[:bsz], src_feats[:bsz], trg_feats[:bsz])
        else:
            match_loss = torch.zeros(1).cuda()
 
        loss = self.ce_w * entropy_loss + self.match_w * match_loss

        return loss, entropy_loss.item(), match_loss.item()


    def information_entropy(self, corr: torch.Tensor):
        r"""Computes information entropy of all candidate matches"""

        #TODO: other norm is possible, softmax with t, [B, HW, HW]
        B, HW0, HW1 = corr.size()[0], corr.size()[1], corr.size()[2]
        src_pdf = l1normalize(corr, dim=2)
        trg_pdf = l1normalize(corr.transpose(1,2), dim=2)

        # [B, HW, HW]
        entropy_loss = torch.sum((-1.*src_pdf) * torch.log(src_pdf+1e-8))/(B*HW0) + \
                       torch.sum((-1.*trg_pdf) * torch.log(trg_pdf+1e-8))/(B*HW1)
     
        return entropy_loss.unsqueeze(dim=0)

    
    def information_match(self, corr: torch.Tensor, src_feats: torch.Tensor, trg_feats: torch.Tensor):

        B, HW0, HW1 = src_feats.size()[0], src_feats.size()[1], trg_feats.size()[2]
        src_feats = F.normalize(src_feats, p=2, dim=2) # [B, HW0, C]
        trg_feats = F.normalize(trg_feats, p=2, dim=2)
        
        # corr = [B, HW0, HW1]
        s_corr = F.softmax(corr/self.t[0], dim=2)
        t_corr = F.softmax(corr.permute(0,2,1)/self.t[0], dim=2)

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
        grid_x, grid_y = self.soft_argmax(x.view(B, -1, feat_size, feat_size), feature_size=feat_size)

        pred_flow = torch.cat((grid_x, grid_y), dim=1)
        pred_flow = self.unnormalise_and_convert_mapping_to_flow(pred_flow)

        loss_flow = self.EPE(pred_flow, flow_gt)
        return loss_flow
    
    def soft_argmax(self, corr, beta=0.02, feature_size=64):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        x_normal = nn.Parameter(torch.tensor(np.linspace(-1,1,feature_size), dtype=torch.float, requires_grad=False)).cuda()
        y_normal = nn.Parameter(torch.tensor(np.linspace(-1,1,feature_size), dtype=torch.float, requires_grad=False)).cuda()

        b,_,h,w = corr.size()
        
        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
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

    def EPE(self, input_flow, target_flow, sparse=True, mean=True, sum=False):

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

    def unnormalise_and_convert_mapping_to_flow(self, map):
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

    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    
class ASYMLoss(nn.Module):
    r"""ASYM, 'Demistfying Unsupervised Semantic Correspondence Estimation' paper"""
    def __init__(self, temp=[10, 5], loss_type='ce') -> None:
        super(ASYMLoss, self).__init__()
        self.t1 = temp[0]
        self.t2 = temp[1]
        self.loss_type = loss_type

    def forward(self, src_f, trg_f, src_pf, trg_pf):

        src_f = F.normalize(src_f, p=2, dim=1) * self.t1
        trg_f = F.normalize(trg_f, p=2, dim=1) * self.t1
        
        src_pf = F.normalize(src_pf, p=2, dim=1) * self.t2
        trg_pf = F.normalize(trg_pf, p=2, dim=1) * self.t2
        
        B, C, H, W = src_f.shape
        _, Cp, _, _ = src_pf.shape
        
        corr = torch.bmm(src_f.view(B,C,H*W).permute(0,2,1), trg_f.view(B,C,H*W)).view(B,H,W,H,W)
        s_corr = F.softmax(corr.view(B,H,W,-1), dim=3).view(B,H,W,H,W)

        corr_p = torch.bmm(src_pf.view(B,Cp,H*W).permute(0,2,1), trg_pf.view(B,Cp,H*W)).view(B,H,W,H,W)
        s_corr_p = F.softmax(corr_p.view(B,H,W,-1), dim=3).view(B,H,W,H,W)

        if self.loss_type == 'ce':
            loss  = torch.sum((-1.*s_corr) * torch.log(s_corr_p+1e-8))/ (B*H*W)
            return loss
        else:
            loss = torch.sum(torch.pow(s_corr - s_corr_p, 2)) 
            return loss*100 / (H*W*B)
        
class LeadLoss(nn.Module):
    r"""LEAD, ' Self-Supervised Landmark Estimation by Aligning Distributions of Feature Similarity' paper"""
    def __init__(self, temp=[10], loss_type='ce') -> None:
        super(ASYMLoss, self).__init__()
        self.t1 = temp[0]
        self.loss_type = loss_type

    def forward(self, src_f, trg_f, src_pf, trg_pf):

        #LEAD loss, Karmali et.al 2022
        src_f = F.normalize(src_f, p=2, dim=1)
        trg_f = F.normalize(trg_f, p=2, dim=1)
        
        src_pf = F.normalize(src_pf, p=2, dim=1)
        trg_pf = F.normalize(trg_pf, p=2, dim=1)
        
        B, C, H, W = src_f.shape
        _, Cp, _, _ = src_pf.shape
        
        corr = torch.bmm(src_f.view(B,C,H*W).permute(0,2,1), trg_f.view(B,C,H*W)).view(B,H,W,H,W)
        s_corr = F.softmax(corr.view(B,H,W,-1)*self.t1, dim=3).view(B,H,W,H,W)

        corr_p = torch.bmm(src_pf.view(B,Cp,H*W).permute(0,2,1), trg_pf.view(B,Cp,H*W)).view(B,H,W,H,W)
        s_corr_p = F.softmax(corr_p.view(B,H,W,-1)*self.t1, dim=3).view(B,H,W,H,W)

        if self.loss_type == 'ce':
            loss  = torch.sum((-1.*s_corr) * torch.log(s_corr_p+1e-8))/ (B*H*W)
            return loss
        else:
            loss = torch.sum(torch.pow(s_corr - s_corr_p, 2)) / (B*H*W)
            return loss * 100