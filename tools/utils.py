r"""Some helper functions"""
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from enum import Enum
from torch import distributed as dist
from .logger import Logger
import re
from collections import OrderedDict

def fix_randseed(seed):
    r"""Fixes random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mean(x):
    r"""Computes average of a list"""
    return sum(x) / len(x) if len(x) > 0 else 0.0

def where(predicate):
    r"""Predicate must be a condition on nd-tensor"""
    matching_indices = predicate.nonzero()
    if len(matching_indices) != 0:
        matching_indices = matching_indices.t().squeeze(0)
    return matching_indices

def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"

# Draw class pck
# if False and (epoch % 2)==0:
#     draw_class_pck_path = os.path.join(Logger.logpath, "draw_class_pck")
#     os.makedirs(draw_class_pck_path, exist_ok=True)
#     class_pth = utils.draw_class_pck(
#         average_meter.sel_buffer["votes_geo"], draw_class_pck_path, epoch, step
#     )
#     if args.use_wandb and dist.get_rank() == 0:
#         wandb.log(
#             {
#                 "class_pck": wandb.Image(Image.open(class_pth).convert("RGB")),
#             }
#         )

def draw_class_pck(sel_buffer, class_pck_path, epoch=0, step=0):

    mean_sel_buffer = {}
    for (key, value) in sel_buffer.items():
        # print(key, "%.3f" % list_mean(value))
        mean_sel_buffer[key] = mean(value)

    names = list(mean_sel_buffer.keys())
    values = list(mean_sel_buffer.values())

    fig, axs = plt.subplots(1, 1, figsize=(8, 3), sharey=True)
    pps = axs.bar(names, values)

    plt.setp(axs.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    axs.set_ylabel('Avg pck')

    for p in pps:
        height = p.get_height()
        axs.annotate('%.2f'%(height),
                        xy=(p.get_x()+p.get_width()/2, height),
                        xytext=(0,3),
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=45)
    axs.set_title("Per-class pck, e={}, s={}".format(epoch, step))
    fig.tight_layout()

    class_pth = os.path.join(class_pck_path, "e{}_s{}.png".format(epoch, step))

    fig.savefig(class_pth)
    plt.close('all')
    
    return class_pth

# 6. draw weight map
# if dist.get_rank() == 0:
#     # 1. Draw weight map
#     weight_map_path = os.path.join(Logger.logpath, "weight_map")
#     os.makedirs(weight_map_path, exist_ok=True)
#     weight_pth = draw_weight_map(
#         model.module.learner.layerweight.detach().clone(),
#         epoch, weight_map_path)
#     if args.use_wandb:
#         wandb.log({"weight_map": wandb.Image(Image.open(weight_pth).convert("RGB"))})

def draw_weight_map(weight, epoch, weight_map_path):

    num_weight = weight.numel()
    pad_width = 20 if num_weight == 17 else 35
    pad_weight = torch.zeros(pad_width, dtype=weight.dtype)
    pad_weight[:num_weight] = weight
    pad_weight[num_weight:] = -1000

    pad_weight = pad_weight.sigmoid().view(-1, 5)

    y, x = pad_weight.size()[0], pad_weight.size()[1]
    fig, ax = plt.subplots()
    im = ax.imshow(pad_weight)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(x))
    ax.set_yticks(np.arange(y))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(y):
        for j in range(x):
            text = ax.text(j, i, "%.3f"%pad_weight[i, j].item(),
                        ha="center", va="center", color="w")

    ax.set_title("Sigmoid weight, epoch={}".format(epoch))
    fig.tight_layout()

    weight_pth = os.path.join(weight_map_path, "e{}.png".format(epoch))

    fig.savefig(weight_pth)
    plt.close('all')
    
    del weight, pad_weight
    
    return weight_pth

def get_concat_h(im1, im2):
    max_height = max(im1.height, im2.height)
    dst = Image.new('RGB', (im1.width + im2.width, max_height), color="white")
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def draw_matches_on_image(epoch, step, match_idx, match_pck, src_img, trg_img, batch, pred_trg_kps=None, origin=True, color_ids=None, draw_match_path=None):
    r"""Draw keypoints on image
    
    Args:
        match_idx: sample id in one batch
        src_img, trg_img: The original PIL.Image object

    """
    
    # 1. Check flip
    if batch['flip'][match_idx].item() == 1:
        src_img = src_img.transpose(Image.FLIP_LEFT_RIGHT)
        trg_img = trg_img.transpose(Image.FLIP_LEFT_RIGHT)
    n_pts = batch['n_pts'][match_idx]

    # 2. Check image rescale
    if origin:
        src_ratio = batch['src_intratio'][match_idx].flip(dims=(0,)).view(2,-1).cpu().numpy() # wxh
        trg_ratio = batch['trg_intratio'][match_idx].flip(dims=(0,)).view(2,-1).cpu().numpy()
    else:
        src_ratio = torch.ones((2,1)).cpu().numpy() 
        trg_ratio = torch.ones((2,1)).cpu().numpy()

    # 3. Check kps
    src_kps = (batch['src_kps'][match_idx][:,:n_pts.item()].cpu().numpy() / src_ratio)

    if pred_trg_kps is not None:
        trg_kps = (pred_trg_kps[:,:n_pts.item()].cpu().numpy() / trg_ratio)
    else:
        trg_kps = (batch['trg_kps'][match_idx][:,:n_pts.item()].cpu().numpy() / trg_ratio).cpu().numpy()
    
    # 4. Check bounding box
    src_bbox = batch['src_bbox'][match_idx].cpu().numpy()
    trg_bbox = batch['trg_bbox'][match_idx].cpu().numpy()

    src_bbox_start = (src_bbox[0]/src_ratio[0], src_bbox[1]/src_ratio[1])
    src_bbox_w, src_bbox_h = (src_bbox[2] - src_bbox[0])/src_ratio[0], (src_bbox[3] - src_bbox[1])/src_ratio[1]

    trg_bbox_start = ((trg_bbox[0]/trg_ratio[0] + src_img.width), trg_bbox[1]/trg_ratio[1])
    trg_bbox_w, trg_bbox_h = (trg_bbox[2] - trg_bbox[0])/trg_ratio[0], (trg_bbox[3] - trg_bbox[1])/trg_ratio[1]

    src_rect = patches.Rectangle(src_bbox_start, src_bbox_w, src_bbox_h, linewidth=2, edgecolor='b', facecolor='none')
    trg_rect = patches.Rectangle(trg_bbox_start, trg_bbox_w, trg_bbox_h, linewidth=2, edgecolor='b', facecolor='none')

    # 5. Concatenate images horinzontally
    con_img = get_concat_h(src_img, trg_img)
    con_img = np.array(con_img)

    # 6. Draw the matches
    fig, ax = plt.subplots()
    ax.imshow(con_img)

    colors = ['red', 'green']
    if color_ids is None:
        color_ids = torch.ones(n_pts, dtype=torch.uint8)

    for pt_idx in range(n_pts):
        ax.plot(src_kps[0,pt_idx], src_kps[1,pt_idx], marker='o', color=colors[color_ids[pt_idx]])
        ax.plot(trg_kps[0,pt_idx] + src_img.width, trg_kps[1,pt_idx], marker='o', color=colors[color_ids[pt_idx]])
        ax.plot([src_kps[0,pt_idx], trg_kps[0,pt_idx] + src_img.width], [src_kps[1,pt_idx], trg_kps[1,pt_idx]], color=colors[color_ids[pt_idx]], linewidth=2)

    ax.add_patch(src_rect)
    ax.add_patch(trg_rect)

    img_name = "{}-{}.png".format(batch["src_imname"][match_idx][:-4], batch["src_imname"][match_idx][:-4])
    ax.set_title("Pair=%s, epoch=%d, step=%d, pck=%.3f" % (img_name[:-4], epoch, step, match_pck))
    fig.tight_layout()

    match_pth = os.path.join(draw_match_path, img_name)

    fig.savefig(match_pth)
    plt.close('all')

    return match_pth

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def parse_string(string):
    r"""Parse given hyperpixel list (string -> int)"""
    string = list(map(int, re.findall(r'\d+', string)))
    if len(string) == 2:
        string = tuple(string)
    else:
        string = int(string[0])
        
    return string

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
    VAL = 4

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        if self.summary_type is Summary.VAL:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'

        return fmtstr.format(**self.__dict__)


    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        Logger.info(',  '.join(entries))
        
    def display_summary(self):
        entries = [self.prefix]
        entries += [meter.summary() for meter in self.meters]
        # print(' '.join(entries))
        Logger.info(',  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def reduce_results(results):
    """Coalesced mean all reduce over a dictionary of 0-dimensional tensors"""
    names, values = [], []
    for k, v in results.items():
        names.append(k)
        values.append(v)

    # Peform the actual coalesced all_reduce
    values = torch.stack([torch.tensor(v) for v in values], dim=0).cuda()
    dist.all_reduce(values, dist.ReduceOp.SUM)
    values.div_(dist.get_world_size())
    values = torch.chunk(values, values.size(0), dim=0)

    # Reconstruct the dictionary
    return OrderedDict((k, v.item()) for k, v in zip(names, values))

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def get_input(data, use_negative):
    if use_negative:
        shifted_idx = np.roll(np.arange(data['src_img'].size(0)), -1)
        trg_img_neg = data['trg_img'][shifted_idx].clone()
        trg_cls_neg = data['category_id'][shifted_idx].clone()
        neg_subidx = (data['category_id'] - trg_cls_neg) != 0

        src_img = torch.cat([data['src_img'], data['src_img'][neg_subidx]], dim=0)
        trg_img = torch.cat([data['trg_img'], trg_img_neg[neg_subidx]], dim=0)

        del trg_img_neg

        if 'src_mask' in data.keys():
            trg_mask_neg = data['trg_mask'][shifted_idx].clone()
            src_mask = torch.cat([data['src_mask'], data['src_mask'][neg_subidx]], dim=0)
            trg_mask = torch.cat([data['trg_mask'], trg_mask_neg[neg_subidx]], dim=0)

            del trg_mask_neg
        else:
            src_mask, trg_mask = None, None
        
        num_negatives = neg_subidx.sum()

    else:
        num_negatives = 0
        src_img = data['src_img']
        trg_img = data['trg_img']

        if 'src_mask' in data.keys():
            src_mask = data['src_mask']
            trg_mask = data['trg_mask']
        else:
            src_mask, trg_mask = None, None

    imgs = torch.cat([src_img, trg_img], dim=0).cuda(non_blocking=True)
    if src_mask and trg_mask:
        masks = torch.cat([src_mask, trg_mask], dim=0).cuda(non_blocking=True)
    else:
        masks = None

    return imgs, masks, num_negatives

def get_Meters(criterion, collect_grad, len_dataloader, epoch):
    meters = {}
    loss_meter = AverageMeter('loss', ':4.2f')
    meters['loss'] = loss_meter
    progress_list = [loss_meter]
    
    if criterion == "weak":
        entropy_meter = AverageMeter('EntropyLoss', ':4.2f') 
        match_meter = AverageMeter('MatchLoss', ':4.2f') 
        meters['EntropyLoss'] = entropy_meter
        meters['MatchLoss'] = match_meter
        progress_list += [entropy_meter, match_meter]

        if collect_grad:
            entropyG_meter = AverageMeter('EntropyG', ':4.2f', summary_type=Summary.VAL) 
            matchG_meter = AverageMeter('MatchG', ':4.2f', summary_type=Summary.VAL) 
            meters['EntropyG'] = entropyG_meter
            meters['MatchG'] = matchG_meter
            progress_list += [entropyG_meter, matchG_meter]
    
    progress = ProgressMeter(len_dataloader, progress_list, prefix="Epoch[{}]".format(epoch))

    return progress, meters

#     if args.criterion == "weak" and (iter%10 == 0) and args.collect_grad:
#         GW_t = []
#         for i in range(3):
#             # get the gradient of this task loss with respect to the shared parameters
#             GiW_t = torch.autograd.grad(
#                 task_loss[i], model.module.learner.parameters(), retain_graph=True)[0]
                            
#             # GiW_t is tuple
#             # compute the norm
#             dist.barrier()     
#             dist.all_reduce(GiW_t, op=dist.ReduceOp.SUM)
#             GiW_t /= dist.get_world_size()
#             GW_t.append(torch.norm(GiW_t).item())

#         del GiW_t
#         Loss['SelfG'].update(GW_t[0], bsz)
#         Loss['CrossG'].update(GW_t[1], bsz)
#         Loss['matchG'].update(GW_t[2], bsz)
#         if args.use_wandb and dist.get_rank() == 0:
#             wandb.log({"discSelfGrad": GW_t[0], "discCrossGrad": GW_t[1], "matchGrad": GW_t[2]})














