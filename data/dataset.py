r"""Superclass for semantic correspondence datasets"""
import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F

class CorrespondenceDataset(Dataset):
    r"""Parent class of PFPascal, PFWillow, Caltech, and SPair""" # img_size = (H, W)
    def __init__(self, benchmark, datapath, thres, split, img_size):
        r"""CorrespondenceDataset constructor"""
        super(CorrespondenceDataset, self).__init__()

        # {Directory name, Layout path, Image path, Annotation path, PCK threshold}
        self.metadata = {
            'pfwillow': ('PF-WILLOW',
                         'test_pairs.csv',
                         '',
                         '',
                         'bbox'),
            'pfpascal': ('PF-PASCAL',
                         '_pairs.csv',
                         'JPEGImages',
                         'Annotations',
                         'img'),
            'spair':   ('SPair-71k',
                        'Layout/large',
                        'JPEGImages',
                        'PairAnnotation',
                        'bbox')
        }

        # Directory path for train, val, or test splits
        base_path = os.path.join(os.path.abspath(datapath), self.metadata[benchmark][0])
        if benchmark == 'pfpascal':
            self.spt_path = os.path.join(base_path, split+'_pairs.csv')
        elif benchmark == 'spair':
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1], split+'.txt')
        else:
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1])

        # Directory path for images
        self.img_path = os.path.join(base_path, self.metadata[benchmark][2])

        # Directory path for annotations
        if benchmark == 'spair':
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3], split)
        else:
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3])

        # Miscellaneous
        self.max_pts = 40
        self.split = split

        if img_size is not None:
            trans = [transforms.Resize(img_size)]
        else:
            trans = []
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                          std=[0.229, 0.224, 0.225]))

        self.img_size = img_size # rescale image
        self.benchmark = benchmark
        self.range_ts = torch.arange(self.max_pts)
        self.thres = self.metadata[benchmark][4] if thres == 'auto' else thres
        self.transform = transforms.Compose(trans)

        # To get initialized in subclass constructors
        self.train_data = []
        self.src_imnames = []
        self.trg_imnames = []
        self.cls = []
        self.cls_ids = []
        self.src_kps = []
        self.trg_kps = []

    def __len__(self):
        r"""Returns the number of pairs"""
        return len(self.train_data)

    def __getitem__(self, idx):
        r"""Construct and return a batch"""

        # Image names
        batch = dict()
        batch['src_imname'] = self.src_imnames[idx]
        batch['trg_imname'] = self.trg_imnames[idx]

        # Class of instances in the images
        batch['category_id'] = self.cls_ids[idx]
        batch['pair_class'] = self.cls[batch['category_id']]

        # Image as PIL (original height, original width)
        src_pil = self.get_image(self.src_imnames, idx)
        trg_pil = self.get_image(self.trg_imnames, idx)
        batch['src_imsize'] = src_pil.size # (W,H)
        batch['trg_imsize'] = trg_pil.size

        # Image (original size) as tensor
        batch['src_img'] = self.transform(src_pil) # totensor, CxHxW
        batch['trg_img'] = self.transform(trg_pil)

        # Key-points (original) as tensor
        batch['src_kps'], num_pts = self.get_points(self.src_kps, idx, src_pil.size)
        batch['trg_kps'], _ = self.get_points(self.trg_kps, idx, trg_pil.size)
        batch['n_pts'] = torch.tensor(num_pts)

        # The number of pairs in training split
        batch['datalen'] = len(self.train_data)

        return batch

    def get_image(self, imnames, idx):
        r""" Reads PIL image from path """
        path = os.path.join(self.img_path, imnames[idx])
        return Image.open(path).convert('RGB')
    
    def get_pckthres(self, batch):
        r""" Computes PCK threshold """
        if self.thres == 'bbox':
            bbox = batch['trg_bbox'].clone()
            bbox_w = (bbox[2] - bbox[0])
            bbox_h = (bbox[3] - bbox[1])
            pckthres = torch.max(bbox_w, bbox_h)
        elif self.thres == 'img':
            imsize_t = batch['trg_img'].size()
            pckthres = torch.tensor(max(imsize_t[1], imsize_t[2]))
        else:
            raise Exception('Invalid pck threshold type: %s' % self.thres)
        return pckthres.float()

    def get_points(self, pts_list, idx, org_imsize):
        r"""Returns key-points of an image"""

        xy, n_pts = pts_list[idx].size()
        if self.img_size is not None:
            pad_pts = torch.zeros((xy, self.max_pts - n_pts)) - 2
            x_crds = pts_list[idx][0] * (self.img_size / org_imsize[0])
            y_crds = pts_list[idx][1] * (self.img_size / org_imsize[1])
            kps = torch.cat([torch.stack([x_crds, y_crds]), pad_pts], dim=1)
        else:
            # only for test case, we use original size here
            kps = torch.cat([torch.stack([pts_list[idx][0], pts_list[idx][1]])], dim=1)

        return kps, n_pts




