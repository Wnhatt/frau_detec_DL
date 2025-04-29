from pathlib import Path
import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import logging


def extract_features(model, dataset, batch_size=128, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features = []

    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            feats = model.forward_embeddings(batch_x)  # feats là torch.Tensor
            all_features.append(feats.cpu())  # Đưa về CPU, không chuyển sang numpy

    # Nối lại theo chiều batch
    return torch.cat(all_features, dim=0)


def compute_average_activations(activations):
    """
    Given a 2D activation matrix of shape [num_samples, num_dimensions],
    returns a 1D tensor of shape [num_dimensions] containing the average
    absolute activation across all samples.

    Args:
        activations (Tensor): shape [total_samples, total_dims]

    Returns:
        mean_acts (Tensor): shape [total_dims], mean absolute activation
    """
    # Compute mean absolute activation for each dimension
    # across all samples (dim=0).
    mean_acts = activations.abs().mean(dim=0)  # [total_dims]
    return mean_acts



def get_prune_indices(mean_activations, prune_ratio):
    """
    From a vector of mean activations, get the indices of the lowest
    prune_ratio fraction of dimensions.

    Args:
        mean_activations (Tensor): shape [total_dims]
        prune_ratio (float): fraction of dimensions to prune, e.g. 0.2

    Returns:
        prune_inds (LongTensor): indices of the dimensions to prune
    """
    total_dims = mean_activations.shape[0]
    num_prune = int(prune_ratio * total_dims)

    # Sort dimensions by ascending activation
    sorted_inds = torch.argsort(mean_activations)  # ascending
    # Select the lowest-activation indices
    prune_inds = sorted_inds[:num_prune]
    return prune_inds

def create_pruning_mask(total_dims, prune_inds, device):
    """
    Create a 1D pruning mask of length 'total_dims' that is 1.0
    for unpruned dimensions and 0.0 for pruned dimensions.

    Args:
        total_dims (int): total number of dimensions in the final Transformer output
        prune_inds (Tensor): indices of dimensions to prune
        device (torch.device)

    Returns:
        mask (nn.Parameter): shape [total_dims], with 1/0 entries
    """
    mask = torch.ones(total_dims, dtype=torch.float32, device=device)
    mask[prune_inds] = 0.0
    # We wrap this in nn.Parameter so it's easy to use in forward, but
    # we set requires_grad=False because we don't want to train it.
    # mask = nn.Parameter(mask, requires_grad=False)
    return mask


class PrunedFC(nn.Module):
    def __init__(self, original_fc, pruning_mask: torch.Tensor):
        super().__init__()
        self.fc = original_fc
        self.register_buffer("pruning_mask", pruning_mask)

    def forward(self, x):
        # Apply pruning mask
        x = x * self.pruning_mask
        return self.fc(x)
