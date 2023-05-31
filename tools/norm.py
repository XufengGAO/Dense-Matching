r"""Normalization functions"""
import torch

def l1normalize(x, dim):
    r"""L1-normalization"""
    vector_sum = torch.sum(x, dim=dim, keepdim=True)
    vector_sum[vector_sum == 0] = 1.0
    return x / vector_sum

def linearnormalize(x, dim):
    r"""linear normalization"""
    vectore_max = torch.max(x, dim=dim, keepdim=True)
    vectore_min = torch.max(x, dim=dim, keepdim=True)

    return (x - vectore_min)/(vectore_max - vectore_min + 1e-30)
