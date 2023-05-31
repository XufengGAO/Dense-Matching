"""For quantitative evaluation of DHPF"""
from skimage import draw
import numpy as np
import torch
from . import utils
from model import geometry, rhm_map
import torch.nn.functional as F

class Evaluator:
    r"""Computes evaluation metrics of PCK, LT-ACC, IoU"""

    def __init__(self, criterion, device, alpha=0.1):
        self.alpha = alpha
        self.rf_center = None
        self.rf = None
        self.supervision = criterion
        self.device = device

    def set_geo(self, rfsz, jsz, feat_h, feat_w):
        self.rf = geometry.receptive_fields(rfsz, jsz, feat_h, feat_w).to(self.device)
        self.rf_center = geometry.center(self.rf)

    def evaluate(self, src_kps, trg_kps, n_pts, corr, pckthres, pck_only):
        if not pck_only:
            src_kpidx = match_idx(src_kps, n_pts, self.rf_center)
            trg_kpidx = match_idx(trg_kps, n_pts, self.rf_center)
            easy_match = {'src': [], 'trg': [], 'dist': []}
            hard_match = {'src': [], 'trg': []}
        
        prd_kps = geometry.predict_kps(self.rf, src_kps, n_pts, corr)

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
                easy_match['src'].append(src_kpidx[idx][:npt][correct_ids])
                easy_match['trg'].append(trg_kpidx[idx][:npt][correct_ids])
                hard_match['src'].append(src_kpidx[idx][:npt][incorrect_ids])
                hard_match['trg'].append(trg_kpidx[idx][:npt][incorrect_ids])
            
            pck.append(int(ncorrt)/int(npt))
     
            del correct_dist, correct_ids, incorrect_ids, ncorrt

        if pck_only:
            return pck
        else:
            eval_result = {'easy_match': easy_match,
                        'hard_match': hard_match,
                        'pck': pck, 
                        'pck_ids': pck_ids}
            
            
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