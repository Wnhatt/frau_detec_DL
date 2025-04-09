import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from attacks.universal_backdoor_attacks import universal_backdoor_attack
from utils.dataset import simDataset
from models.models import LSTM

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

# --- Initialize Attack ---
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

# --- Train Clean Model ---
attack.train((train_loader, test_loader), num_epochs=15, criterion=criterion, optimizer=optimizer)
attack.test(test_loader)

# --- Select & Rank ---
attack.select_non_target_samples(train_dataset.get_data())
attack.confidence_based_sample_ranking(batch_size=32)

# --- Trigger Creation ---
attack.define_trigger(train_dataset.get_data())
attack.compute_mode(train_dataset.get_data())
attack.optimize_trigger(num_epochs=500)

print("✅ Trigger optimized:\n", attack.delta)
print("✅ Mode vector:\n", attack.mode_vector)

# --- Poison Datasets ---
poisoned_trainset, poisoned_train_samples = attack.construct_poisoned_dataset(train_dataset.get_data(), epsilon=0.1)
poisoned_testset, poisoned_test_samples = attack.construct_poisoned_dataset(test_dataset.get_data(), epsilon=1.0)

# --- Save Poisoned Sets ---
attack.poisoned_dataset = (poisoned_trainset, poisoned_testset)
attack.poisoned_samples = (poisoned_train_samples, poisoned_test_samples)

# --- Reload DataLoaders ---
poisoned_train_loader = DataLoader(poisoned_trainset, batch_size=32, shuffle=True)
poisoned_test_loader = DataLoader(poisoned_testset, batch_size=32, shuffle=False)

# --- Train on Poisoned Dataset ---
attack.train((poisoned_train_loader, poisoned_test_loader), num_epochs, criterion, optimizer)

# --- Evaluate Model ---
print("\n📊 Evaluation:")
print("Clean Accuracy (CDA):")
cda = attack.test(test_loader)

print("\nAttack Success Rate (ASR):")
asr = attack.compute_asr()

# --- Optional: Save trigger & poisoned data ---
attack.save_trigger()
attack.save_poisoned_dataset()
