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
    
    def optmatch(self, corr):
        pass

    def compute_bin_id(self, src_imsize, src_box, trg_box, hs_cellsize, nbins_x):
        r"""Computes Hough space bin ids for voting"""
        src_ptref = src_imsize.float()
        src_trans = geometry.center(src_box)
        trg_trans = geometry.center(trg_box)
        xy_vote = (src_ptref.unsqueeze(0).expand_as(src_trans) - src_trans).unsqueeze(2).\
                      repeat(1, 1, len(trg_box)) + trg_trans.t().unsqueeze(0).repeat(len(src_box), 1, 1)

        bin_ids = (xy_vote / hs_cellsize).long()

        return bin_ids[:, 0, :] + bin_ids[:, 1, :] * nbins_x

    def build_hspace(self, src_imsize, trg_imsize, ncells):
        r"""Build Hough space"""
        hs_width = src_imsize[0] + trg_imsize[0]
        hs_height = src_imsize[1] + trg_imsize[1]
        hs_cellsize = math.sqrt((hs_width * hs_height) / ncells)
        nbins_x = int(hs_width / hs_cellsize) + 1
        nbins_y = int(hs_height / hs_cellsize) + 1

        return nbins_x, nbins_y, hs_cellsize

































