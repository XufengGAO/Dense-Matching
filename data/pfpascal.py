r"""PF-PASCAL dataset"""
import os

import scipy.io as sio
import pandas as pd
import numpy as np
import torch

from .dataset import CorrespondenceDataset
from PIL import Image


class PFPascalDataset(CorrespondenceDataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, benchmark, datapath, thres, split, img_size):
        r"""PF-PASCAL dataset constructor"""
        super(PFPascalDataset, self).__init__(benchmark, datapath, thres, split, img_size)

        self.train_data = pd.read_csv(self.spt_path) # dataframe
        self.src_imnames = np.array(self.train_data.iloc[:, 0])
        self.trg_imnames = np.array(self.train_data.iloc[:, 1])
        self.cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.cls_ids = self.train_data.iloc[:, 2].values.astype('int') - 1

        if split == 'trn': # trn.csv file inclues column 'flip'
            self.flip = self.train_data.iloc[:, 3].values.astype('int')
        self.src_kps = [] # list of tensor keypoints
        self.trg_kps = [] 
        self.src_bbox = [] # list of tensor bbx (x11, x12, x21, x22)
        self.trg_bbox = []
        # loop over each pair
        for src_imname, trg_imname, cls in zip(self.src_imnames, self.trg_imnames, self.cls_ids):
            # read annotation files
            src_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(src_imname))[:-4] + '.mat'
            trg_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(trg_imname))[:-4] + '.mat'

            src_kp = torch.tensor(read_mat(src_anns, 'kps')).float()
            trg_kp = torch.tensor(read_mat(trg_anns, 'kps')).float()
            src_box = torch.tensor(read_mat(src_anns, 'bbox')[0].astype(float))
            trg_box = torch.tensor(read_mat(trg_anns, 'bbox')[0].astype(float))

            src_kps = []
            trg_kps = []
            for src_kk, trg_kk in zip(src_kp, trg_kp):
                # if kp is nan, just ignore
                if torch.isnan(src_kk).sum() > 0 or torch.isnan(trg_kk).sum() > 0:
                    continue
                else:
                    src_kps.append(src_kk)
                    trg_kps.append(trg_kk)
            self.src_kps.append(torch.stack(src_kps).t()) # stacked kp for one image, size = (2, num_kp)
            self.trg_kps.append(torch.stack(trg_kps).t())
            self.src_bbox.append(src_box) # bbx consists of 4 numbers
            self.trg_bbox.append(trg_box)

        self.src_imnames = list(map(lambda x: os.path.basename(x), self.src_imnames))
        self.trg_imnames = list(map(lambda x: os.path.basename(x), self.trg_imnames))



    def __getitem__(self, idx):
        r"""Construct and return a batch for PF-PASCAL dataset"""
        batch = super(PFPascalDataset, self).__getitem__(idx)

        # Object bounding-box (list of tensors)
        batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize'])
        batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize'])
        batch['pckthres'] = self.get_pckthres(batch) # rescaled pckthres
            
        # Horizontal flip of key-points when training (no training in HyperpixelFlow)
        if self.split == 'trn' and self.flip[idx]: # width - current x-axis
            # sample['src_kps'][0] = sample['src_img'].size()[2] - sample['src_kps'][0]
            # sample['trg_kps'][0] = sample['trg_img'].size()[2] - sample['trg_kps'][0]
            self.horizontal_flip(batch)
            batch['flip'] = 1
        else:
            batch['flip'] = 0

        return batch
    
    def horizontal_flip(self, sample):
        tmp = sample['src_bbox'][0].clone()
        sample['src_bbox'][0] = sample['src_img'].size(2) - sample['src_bbox'][2]
        sample['src_bbox'][2] = sample['src_img'].size(2) - tmp

        tmp = sample['trg_bbox'][0].clone()
        sample['trg_bbox'][0] = sample['trg_img'].size(2) - sample['trg_bbox'][2]
        sample['trg_bbox'][2] = sample['trg_img'].size(2) - tmp

        sample['src_kps'][0] = sample['src_img'].size(2) - sample['src_kps'][0]
        sample['trg_kps'][0] = sample['trg_img'].size(2) - sample['trg_kps'][0]

        sample['src_img'] = torch.flip(sample['src_img'], dims=(2,))
        sample['trg_img'] = torch.flip(sample['trg_img'], dims=(2,))
   

    def get_mask(self, img_names, idx, flip):
        r"""Return image mask"""
        f = 'f1' if flip==1 else 'f0'
        img_name = os.path.join(self.img_path, f, "%s.pt"%(img_names[idx][:-4]))
        mask_name = img_name.replace('JPEGImages', self.cam) # TODO: prepared cam folder

        if os.path.exists(mask_name):
            # mask = np.array(Image.open(mask_name)) # WxH
            # print(mask_name)
            mask = torch.load(mask_name)
        else:
            #print(img_name,mask_name)
            mask = None
        
        return mask
    
    def get_bbox(self, bbox_list, idx, imsize):
        r""" Returns object bounding-box """
        bbox = bbox_list[idx].clone()
        if self.img_size is not None:
            bbox[0::2] *= (self.img_size[1] / imsize[0]) # w
            bbox[1::2] *= (self.img_size[0] / imsize[1]) # H
        return bbox

def read_mat(path, obj_name):
    r"""Read specified objects from Matlab data file, (.mat)"""
    mat_contents = sio.loadmat(path)
    mat_obj = mat_contents[obj_name]

    return mat_obj
