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
from defences.pruning_for_backdoor import extract_features, compute_average_activations, get_prune_indices, create_pruning_mask, PrunedFC

def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    return 100 * correct / total
from utils.dataset import load_model_state


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

    model, optimizer = load_model_state(model, optimizer, os.path.join(save_path, f'{model.model_name}_falfa'))
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

    poisoned_samples_set = set(tuple(sample.tolist()) for sample, _ in poisoned_train_samples)

    poisoned_indices = []
    # Iterate over poisoned_trainset to find indices of poisoned samples
    for idx, (sample, _) in enumerate(poisoned_trainset):
        # Convert the sample to a tuple for comparison
        sample_tuple = tuple(sample.tolist())
        
        # Check if the sample is in the poisoned_samples_set
        if sample_tuple in poisoned_samples_set:
            poisoned_indices.append(idx)

    print("==> [Step 0] Testing model BEFORE pruning...")

    original_cda = test(model, test_loader, device)
    original_asr = test(model, DataLoader(poisoned_testset, batch_size=32), device)

    print(f"Original Clean Accuracy (CDA): {original_cda:.2f}%")
    print(f"Original Attack Success Rate (ASR): {original_asr:.2f}%")
    
    # gather activations on clean data
    print("==> [Step 1] Gathering Activations on Clean Data...")
    activations_all = extract_features(model, train_dataset, device = device)

    print("    Activations shape:", activations_all.shape)  # e.g., [N, nFeats*dim]

    print("==> [Step 2] Computing Mean Activations + Identifying Pruning Indices...")

    prune_rate = 0.9

    mean_acts = compute_average_activations(activations_all)
    prune_inds = get_prune_indices(mean_acts, prune_rate)

    print(f"    total dims: {mean_acts.numel()}, prune {len(prune_inds)} dims ({prune_rate*100}%).")

    print("==> [Step 3] inject pruning mask into the model ...")
           
    mask = create_pruning_mask(mean_acts.numel(), prune_inds, device)

    print("==> [Step 4] Inject pruning mask into LSTM model...")
    model.fc = PrunedFC(model.fc, mask)

    print("==> [Step 5] Fine-tuning the pruned model...")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Dùng lại train_loader đã có
    for epoch in range(15):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
    
        print(f"Epoch {epoch+1}/5, Loss: {total_loss/len(train_loader):.4f}")
    
    print("==> [Step 6] Testing pruned model on clean data...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    print(f"Accuracy after pruning: {100 * correct / total:.2f}%")

    pruned_cda = 100 * correct / total
    pruned_asr = test(model, DataLoader(poisoned_testset, batch_size=32), device)

    print("=" * 50)
    print("📊 Evaluation Comparison:")
    print(f"   Clean Accuracy Before: {original_cda:.2f}%   | After: {pruned_cda:.2f}%")
    print(f"   Attack Success Rate   Before: {original_asr:.2f}%   | After: {pruned_asr:.2f}%")
    print("=" * 50)
if __name__ == "__main__":
    main()
