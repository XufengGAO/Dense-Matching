"""For quantitative evaluation of DHPF"""
from skimage import draw
import numpy as np
import torch
from . import utils
from model import geometry, rhm_map
import torch.nn.functional as F

class Evaluator:
    r"""Computes evaluation metrics of PCK, LT-ACC, IoU"""

    def __init__(self, criterion, alpha=0.1):
        self.alpha = alpha
        self.hpos = None
        self.hpgeometry = None
        self.supervision = criterion
        self.hsfilter = geometry.gaussian2d(7).unsqueeze(0).unsqueeze(0).cuda()

    def evaluate(self, src_kps, trg_kps, n_pts, corr, pckthres, pck_only):
        self.src_kpidx = match_idx(src_kps, n_pts, self.hpos)
        self.trg_kpidx = match_idx(trg_kps, n_pts, self.hpos)
        prd_kps = geometry.predict_kps(self.hpgeometry, src_kps, n_pts, corr)

        easy_match = {'src': [], 'trg': [], 'dist': []}
        hard_match = {'src': [], 'trg': []}

        pck = []
        pck_ids = torch.zeros((prd_kps.size()[0], prd_kps.size()[-1]), dtype=torch.uint8) # Bx40, default incorrect points
        for idx, (pk, tk, thres, npt) in enumerate(zip(prd_kps, trg_kps, pckthres, n_pts)):
            correct_dist, correct_ids, incorrect_ids, ncorrt = self.classify_prd(pk[:, :npt], tk[:, :npt], thres)
            pck_ids[idx, correct_ids] = 1
            # Collect easy and hard match feature index & store pck to buffer
            if self.supervision == "strong_ce" and not pck_only:
                easy_match['dist'].append(correct_dist)
                # for each keypoint, we find its nearest neighbour of center of receptive field
                # then kpidx is the id of hyperpixel
                easy_match['src'].append(self.src_kpidx[idx][:npt][correct_ids])
                easy_match['trg'].append(self.trg_kpidx[idx][:npt][correct_ids])
                hard_match['src'].append(self.src_kpidx[idx][:npt][incorrect_ids])
                hard_match['trg'].append(self.trg_kpidx[idx][:npt][incorrect_ids])
            pck.append(int(ncorrt)/int(npt))
            # print(int(ncorrt)/int(npt))
            del correct_dist, correct_ids, incorrect_ids, ncorrt
        
        eval_result = {'easy_match': easy_match,
                       'hard_match': hard_match,
                       'pck': pck, 
                       'pck_ids': pck_ids}
        del pck_ids
        
        return eval_result

    def classify_prd(self, prd_kps, trg_kps, pckthres):
        r"""Compute the number of correctly transferred key-points"""
        l2dist = (prd_kps - trg_kps).pow(2).sum(dim=0).pow(0.5)
        thres = pckthres.expand_as(l2dist).float() * self.alpha
        correct_pts = torch.le(l2dist, thres)

        # print("l2", l2dist)
        # print("thres", thres)

        correct_ids = utils.where(correct_pts == 1)
        incorrect_ids = utils.where(correct_pts == 0)
        correct_dist = l2dist[correct_pts]

        del l2dist, thres

        return correct_dist, correct_ids, incorrect_ids, int(torch.sum(correct_pts))
    
    def rhm(self, corr, src_imsize, trg_imsize):
        with torch.no_grad():
            ncells = 8192
            geometric_scores = []
            nbins_x, nbins_y, hs_cellsize = rhm_map.build_hspace(src_imsize, trg_imsize, ncells)
            bin_ids = rhm_map.hspace_bin_ids(src_imsize, self.hpgeometry, self.hpgeometry, hs_cellsize, nbins_x)
            hspace = self.hpgeometry.new_zeros((corr.size()[1], nbins_y * nbins_x))

            hbin_ids = bin_ids.add(torch.arange(0, corr.size()[1]).to(corr.device).
                                mul(hspace.size(1)).unsqueeze(1).expand_as(bin_ids))
            for cor in corr:
                new_hspace = hspace.view(-1).index_add(0, hbin_ids.view(-1), cor.view(-1)).view_as(hspace)
                new_hspace = torch.sum(new_hspace, dim=0).to(corr.device)

                # Aggregate the voting results
                new_hspace = F.conv2d(new_hspace.view(1, 1, nbins_y, nbins_x), self.hsfilter, padding=3).view(-1)

                geometric_scores.append((torch.index_select(new_hspace, dim=0, index=bin_ids.view(-1)).view_as(cor)).unsqueeze(0))

            geometric_scores = torch.cat(geometric_scores, dim=0)

            corr *= geometric_scores
        
        return corr


def find_knn(db_vectors, qr_vectors):
    r"""Finds K-nearest neighbors (Euclidean distance)"""
    # print("knn", db_vectors.unsqueeze(1).size(), qr_vectors.size())
    # print("knn", db_vectors[-3])
    # (3600, 40, 2), repeated centers for each rep field of each hyperpixel
    db = db_vectors.unsqueeze(1).repeat(1, qr_vectors.size(0), 1)

    # (3600, 40, 2), repeated 40 keypoints
    qr = qr_vectors.unsqueeze(0).repeat(db_vectors.size(0), 1, 1)
    dist = (db - qr).pow(2).sum(2).pow(0.5).t() # (40, 3600)
    # keypoint to each center
    # print("dist", dist.size())
    _, nearest_idx = dist.min(dim=1) #  hyperpixel idx for each keypoint
    # print("nea_idx", nearest_idx.size())
    return nearest_idx

def match_idx(kpss, n_ptss, rf_center):
    r"""Samples the nearst feature (receptive field) indices"""
    max_pts = 40
    batch = len(kpss)
    nearest_idxs = torch.zeros((batch, max_pts), dtype=torch.int32).to(rf_center.device)
    for idx, (kps, n_pts) in enumerate(zip(kpss, n_ptss)):
        nearest_idx = find_knn(rf_center, kps[:,:n_pts].t())
        nearest_idxs[idx, :n_pts] = nearest_idx

    return nearest_idxs
