import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from attacks.falfa import falfa
from utils.dataset import simDataset
from models.models import LSTM
from tqdm import tqdm


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
num_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = '/home/anhkhoa/ml4secu/save_path'
datapath = '/home/anhkhoa/ml4secu/data.csv'

# --- Load Dataset ---
train_dataset = simDataset(datapath=datapath, device=device, split='train')
test_dataset = simDataset(datapath=datapath, device=device, split='test')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

# --- Clean Model Accuracy ---
model_clean = LSTM().to(device)
optimizer_clean = optim.Adam(model_clean.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

falfa_clean = falfa(
    model=model_clean,
    device=device,
    epochs=num_epochs,
    epsilon=0.1,
    max_iter=0,
    criterion=criterion,
    optimizer=optimizer_clean
)

falfa_clean.poisoned_labels = None
falfa_clean.train_model_on_loader(train_loader)
acc_clean = evaluate_model(model_clean, test_loader, device)
print(f"🎯 Accuracy BEFORE attack: {acc_clean:.4f}")

# --- FALFA Attack ---
model_attack = LSTM().to(device)
optimizer_attack = optim.Adam(model_attack.parameters(), lr=learning_rate)
falfa_runner = falfa(
    model=model_attack,
    device=device,
    epochs=num_epochs,
    epsilon=0.1,
    max_iter=1,
    criterion=criterion,
    optimizer=optimizer_attack
)

print("🚨 Running FALFA attack...")
poisoned_labels = falfa_runner.falfa_attack_dl_dataloader(
    dataloader=train_loader,
    X_np=train_dataset.x,
    y_np=train_dataset.y
)

# --- Poisoned Model Accuracy ---
# Make new poisoned dataset
X_tensor = torch.tensor(train_dataset.x, dtype = torch.float)
y_poisoned_tensor = torch.tensor(poisoned_labels, dtype=torch.int64)
poisoned_dataset = TensorDataset(X_tensor, y_poisoned_tensor)
poisoned_loader = DataLoader(poisoned_dataset, batch_size=32, shuffle=False)

model_poisoned = LSTM().to(device)
optimizer_poisoned = optim.Adam(model_poisoned.parameters(), lr=learning_rate)
falfa_poisoned = falfa(
    model=model_poisoned,
    device=device,
    epochs=num_epochs,
    epsilon=0.1,
    max_iter=0,
    criterion=criterion,
    optimizer=optimizer_poisoned
)

falfa_poisoned.poisoned_labels = poisoned_labels
falfa_poisoned.train_model_on_loader(poisoned_loader)
acc_poisoned = evaluate_model(model_poisoned, test_loader, device)
print(f"☠️  Accuracy AFTER attack:  {acc_poisoned:.4f}")
