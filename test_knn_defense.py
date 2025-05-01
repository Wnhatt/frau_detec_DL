import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from attacks.falfa import falfa
from utils.dataset import simDataset
from models.models import LSTM
from tqdm import tqdm
import numpy as np
from defences.knn_based_defense import knn_based_defense

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total

# --- Hyperparameters ---
learning_rate = 0.001
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datapath = '/home/anhkhoa/ml4secu/data.csv'

# --- Load Dataset ---
train_dataset = simDataset(datapath=datapath, device=device, split='train')
test_dataset = simDataset(datapath=datapath, device=device, split='test')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

# === 1. TRAIN MODEL BEFORE ATTACK ===
print("✅ Training model BEFORE attack...")
model_clean = LSTM().to(device)
optimizer_clean = optim.Adam(model_clean.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model_clean.train()
    for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model_clean(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer_clean.zero_grad()
        loss.backward()
        optimizer_clean.step()

acc_clean = evaluate_model(model_clean, test_loader, device)
print(f"🎯 Accuracy BEFORE attack (clean labels): {acc_clean:.4f}")

# === 2. FALFA ATTACK ===
print("🚨 Running FALFA attack...")
model_attack = LSTM().to(device)
optimizer_attack = optim.Adam(model_attack.parameters(), lr=learning_rate)

falfa_runner = falfa(
    model=model_attack,
    device=device,
    epochs=num_epochs,
    epsilon=0.1,           # Lật 10% nhãn
    max_iter=1,
    criterion=criterion,
    optimizer=optimizer_attack
)

poisoned_labels = falfa_runner.falfa_attack_dl_dataloader(
    dataloader=train_loader,
    X_np=train_dataset.x,
    y_np=train_dataset.y
)

# === 3. TRAIN MODEL AFTER ATTACK ===
X_tensor = torch.tensor(train_dataset.x, dtype=torch.float)
y_poisoned_tensor = torch.tensor(poisoned_labels, dtype=torch.int64)
poisoned_dataset = TensorDataset(X_tensor, y_poisoned_tensor)
poisoned_loader = DataLoader(poisoned_dataset, batch_size=32, shuffle=True)

model_poisoned = LSTM().to(device)
optimizer_poisoned = optim.Adam(model_poisoned.parameters(), lr=learning_rate)
falfa_poisoned = falfa(
    model=model_poisoned,
    device=device,
    epochs=num_epochs,
    epsilon=0.1,
    max_iter=1,
    criterion=criterion,
    optimizer=optimizer_poisoned
)
falfa_poisoned.poisoned_labels = poisoned_labels
falfa_poisoned.train_model_on_loader(poisoned_loader)

acc_poisoned = evaluate_model(model_poisoned, test_loader, device)

# === 4. STATS ===
num_flipped = (poisoned_labels != train_dataset.y).sum()
print(f"☠️  Accuracy AFTER attack: {acc_poisoned:.4f}")
print(f"🔢 Final flipped labels: {num_flipped} / {len(train_dataset)}")

# === 5. APPLY KNN-BASED DEFENSE ===
print("🛡️ Running kNN-based defense...")
X_np = train_dataset.x
y_poisoned_np = poisoned_labels

# Apply defense
y_defended_np = knn_based_defense(X_np, y_poisoned_np, k=10, eta=0.9)
num_relabel = (y_defended_np != y_poisoned_np).sum()
print(f"🔁 Number of relabeled samples by kNN defense: {num_relabel}")

# === 6. TRAIN AFTER DEFENSE ===
X_tensor_defended = torch.tensor(X_np, dtype=torch.float32)
y_tensor_defended = torch.tensor(y_defended_np, dtype=torch.long)

cleaned_dataset = TensorDataset(X_tensor_defended, y_tensor_defended)
cleaned_loader = DataLoader(cleaned_dataset, batch_size=32, shuffle=True)

model_defended = LSTM().to(device)
optimizer_defended = optim.Adam(model_defended.parameters(), lr=learning_rate)

print("📦 Training model on cleaned data...")
for epoch in range(num_epochs):
    model_defended.train()
    for x_batch, y_batch in tqdm(cleaned_loader, desc=f"Epoch {epoch+1}"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model_defended(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer_defended.zero_grad()
        loss.backward()
        optimizer_defended.step()

acc_defended = evaluate_model(model_defended, test_loader, device)
print(f"✅ Accuracy AFTER kNN defense: {acc_defended:.4f}")
