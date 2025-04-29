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


def spectral_signature_defense(
    features: np.ndarray,
    labels: np.ndarray,
    class_list,
    expected_poison_fraction: float = 0.05,
    extra_multiplier: float = 1.5,
):
    """
    Implements the Spectral Signature defense on extracted features.

    For each class c:
      1) Gather features of that class.
      2) Center them.
      3) Perform SVD -> top singular vector.
      4) Compute squared projection along that vector as score.
      5) Remove top K samples (K ~ expected_poison_fraction * extra_multiplier).
    
    Returns:
      - suspicious_indices (List[int]): indices flagged as suspicious
      - scores_by_class (Dict[int, np.ndarray]): arrays of scores for each class
    """
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must match in length.")

    suspicious_indices = []
    scores_by_class = {}
    all_indices = np.arange(len(labels))

    for c in class_list:
        # Indices for class c
        class_idxs = all_indices[labels == c]
        if len(class_idxs) < 2:
            # Not enough samples to do meaningful SVD
            continue

        class_feats = features[class_idxs]
        mean_feat = class_feats.mean(axis=0)
        centered = class_feats - mean_feat

        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        top_vec = Vt[0]

        # Score = squared projection
        scores = (centered @ top_vec) ** 2
        scores_by_class[c] = scores

        # Number of suspicious to remove
        K = int(len(class_feats) * expected_poison_fraction * extra_multiplier)
        K = min(K, len(class_feats))
        if K < 1:
            continue

        # The top-K scoring samples are suspicious
        suspicious_local = np.argsort(scores)[-K:]  # largest scores
        suspicious_global = class_idxs[suspicious_local]
        suspicious_indices.extend(suspicious_global)

    return suspicious_indices, scores_by_class


def plotCorrelationScores(
    class_id: int,
    scores_for_class: np.ndarray,
    mask_for_class: np.ndarray,
    nbins: int = 100,
    label_clean: str = "Clean",
    label_poison: str = "Poisoned",
    save_path: Path = None
):
    """
    Plots a histogram of the correlation (spectral) scores for clean vs. poisoned
    or suspicious samples in a single class.

    Args:
      - class_id (int): The class label for the plot title
      - scores_for_class (np.ndarray): Array of shape (N_class_samples,) with the spectral scores
      - mask_for_class (np.ndarray): Boolean array of same shape, True means "poisoned" or "suspicious"
      - nbins (int): Number of bins for the histogram
      - label_clean (str): Legend label for clean distribution
      - label_poison (str): Legend label for poison distribution
    """
    plt.figure(figsize=(10, 6))  # Increase figure size for better readability
    sns.set_style("white")
    sns.set_palette("tab10")

    scores_clean = scores_for_class[~mask_for_class]
    scores_poison = scores_for_class[mask_for_class]

    if len(scores_poison) == 0:
        # No poison => just plot clean
        if len(scores_clean) == 0:
            return  # no data
        bins = np.linspace(0, scores_clean.max(), nbins)
        plt.hist(scores_clean, bins=bins, color="green", alpha=0.75, label=label_clean)
    else:
        # We have both categories
        combined_max = max(scores_clean.max(), scores_poison.max())
        bins = np.linspace(0, combined_max, nbins)
        plt.hist(scores_clean, bins=bins, color="green", alpha=0.75, label=label_clean)
        plt.hist(scores_poison, bins=bins, color="red", alpha=0.75, label=label_poison)
        plt.legend(loc="upper right")

    plt.xlabel("Spectral Signature Score", fontsize=12)  # Increase font size
    plt.ylabel("Count", fontsize=12)  # Increase font size
    plt.title(f"Class {class_id}: Score Distribution", fontsize=14)  # Increase font size
    plt.xticks(rotation=45)  # Rotate x-axis ticks to prevent overlap
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()
    
    # Save the plot
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.savefig(f"plots/ss_class_{class_id}.png")


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
            feats = model.forward_embeddings(batch_x)
            all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)


