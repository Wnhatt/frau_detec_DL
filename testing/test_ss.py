from defences.ss_for_backdoor import spectral_signature_defense, plotCorrelationScores

from pathlib import Path
import sys
import os

# Add the project root to the Python path
# Get the absolute path of the current file
current_file = Path(__file__).resolve()
# Get the project root (two directories up from the current file)
project_root = current_file.parent.parent.parent
# Add the project root to sys.path
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from defences.ss_for_backdoor import spectral_signature_defense, plotCorrelationScores, extract_features

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from attacks.universal_backdoor_attacks import universal_backdoor_attack
from utils.dataset import simDataset, load_model
from models.models import LSTM

def main():
    # --- Hyperparameters ---
    learning_rate = 0.001
    num_epochs = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = '/home/anhkhoa/ml4secu/save_path'
    datapath = '/home/anhkhoa/ml4secu/data.csv'

    # --- Load Dataset ---
    train_dataset = simDataset(datapath=datapath, device=device, split='train')
    test_dataset = simDataset(datapath=datapath, device=device, split='test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # --- Initialize Model ---
    model = LSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    attack = universal_backdoor_attack(
        device=device,
        model=model,
        data_obj=(train_loader, test_loader),
        target_label=0,
        mu=0.2,
        beta=0.1,
        lambd=0.1,
        epsilon=0.02,
        save_path=save_path
    )

    # load poisend dataset

    attack.load_poisoned_dataset(
        filepath='/home/anhkhoa/ml4secu/save_path/saved_datasets/poisoned_dataset.pt'
    )

    poisoned_trainset, poisoned_testset = attack.poisoned_dataset
    poisoned_train_samples, poisoned_test_samples = attack.poisoned_samples
    attack.poisoned_dataset = (poisoned_trainset, poisoned_testset)
    attack.poisoned_samples = (poisoned_train_samples, poisoned_test_samples)

    poisoned_indices = []
    # 
    poisoned_samples_set = set(tuple(sample.tolist()) for sample, _ in poisoned_train_samples)

    # Iterate over poisoned_trainset to find indices of poisoned samples
    for idx, (sample, _) in enumerate(poisoned_trainset):
        # Convert the sample to a tuple for comparison
        sample_tuple = tuple(sample.tolist())
        
        # Check if the sample is in the poisoned_samples_set
        if sample_tuple in poisoned_samples_set:
            poisoned_indices.append(idx)


    features = extract_features(model, poisoned_trainset)

    labels = poisoned_trainset.tensors[-1].cpu().numpy()

    class_list = [num for num in range(len(set(labels)))]

    suspicious_idx, scores_dict = spectral_signature_defense(
        features,
        labels=labels,
        class_list=class_list,
        expected_poison_fraction=0.02,
        extra_multiplier=1.5
    )

    poison_mask = np.zeros(len(labels), dtype=bool)
    poison_mask[poisoned_indices] = True

    save_path = Path("/home/anhkhoa/ml4secu/plots")
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    dataset_name = "simDataset"
    target_label = 0
    mu = 0.2
    beta = 0.1
    lambd = 0.1
    epsilon = 0.02

    # Now we can plot the distribution for each class
    for c in class_list:
        # The scores for class c are in scores_dict[c].
        # The associated indices are where y_demo == c.
        class_idxs = np.where(labels == c)[0]
        class_scores = scores_dict[c]
        class_poison_mask = poison_mask[class_idxs]
        save_address = save_path / Path(f"ss_class{c}_{model.model_name}_{dataset_name}_{target_label}_{mu}_{beta}_{lambd}_{epsilon}.png")

        plotCorrelationScores(
            class_id=c,
            scores_for_class=class_scores,
            mask_for_class=class_poison_mask,
            nbins=50,
            label_clean="Clean",
            label_poison="Poisoned",
            save_path=save_address
        )

    suspicious_idx_np = np.array(suspicious_idx)
    poisoned_indices_np = np.array(poisoned_indices)

    # Calculate metrics
    true_positives = np.intersect1d(suspicious_idx_np, poisoned_indices_np).size
    false_positives = suspicious_idx_np.size - true_positives
    total_poisoned = poisoned_indices_np.size
    
    recall = true_positives / total_poisoned if total_poisoned > 0 else 0
    precision = true_positives / suspicious_idx_np.size if suspicious_idx_np.size > 0 else 0

    print("\n=== Defense Evaluation Metrics ===")
    print(f"Total Poisoned Samples: {total_poisoned}")
    print(f"Detected Suspicious Samples: {suspicious_idx_np.size}")
    print(f"True Positives (TP): {true_positives}")
    print(f"False Positives (FP): {false_positives}")
    print(f"Recall (TP Rate): {recall:.2%}")
    print(f"Precision: {precision:.2%}")

if __name__ == "__main__":
    main()


