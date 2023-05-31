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

def unit_gaussian_normalize(x):
    r"""Make each (row) distribution into unit gaussian"""
    correlation_matrix = x - x.mean(dim=2).unsqueeze(2).expand_as(x)

    with torch.no_grad():
        standard_deviation = correlation_matrix.std(dim=2)
        standard_deviation[standard_deviation == 0] = 1.0
    correlation_matrix /= standard_deviation.unsqueeze(2).expand_as(correlation_matrix)

    return correlation_matrix