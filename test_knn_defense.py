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
from utils.dataset import save_model, load_model_state
import numpy as np
save_path = '/home/anhkhoa/ml4secu/save_path'
datapath = '/home/anhkhoa/ml4secu/data.csv'
device = 'cuda'
# --- Load Dataset ---
train_dataset = simDataset(datapath=datapath, device=device, split='train')
test_dataset = simDataset(datapath=datapath, device=device, split='test')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


learning_rate = 0.001
batch_size = 32

model = LSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Others loss function use
from torch.utils.data import DataLoader, TensorDataset
num_epochs = 3

accu_poisend_list = []
accu_defense_list = []

range_attack = [0.45]
import os 

for eps in range_attack:
    model_attack , optimizer_attack = load_model_state(model, optimizer, os.path.join(save_path, f'{model.model_name}_falfa'))

    falfa_runner = falfa(
        model=model_attack,
        device=device,
        epochs=num_epochs,
        epsilon=eps,           
        max_iter=1,
        criterion=criterion,
        optimizer=optimizer_attack
    )
    # --- Train the model on clean data ---
    print(falfa_runner.evaluate_model(
        dataloader=test_loader,
        device = 'cuda'
    ))

    poisoned_labels = falfa_runner.falfa_attack_dl_dataloader(
        dataloader=train_loader,
        X_np=train_dataset.x,
        y_np=train_dataset.y
    )


    X_tensor = torch.tensor(train_dataset.x, dtype=torch.float)
    y_poisoned_tensor = torch.tensor(poisoned_labels, dtype=torch.int64)
    poisoned_dataset = TensorDataset(X_tensor, y_poisoned_tensor)
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=32, shuffle=True)
    
    falfa_runner.poisoned_labels = poisoned_labels
    falfa_runner.train_model_on_loader(poisoned_loader)

    accu_poisned = falfa_runner.evaluate_model(
        dataloader = test_loader,
        device = 'cuda'
    )

    accu_poisend_list.append(accu_poisned)
    print(f"Accuracy after attack: {accu_poisned:.4f}")

    # --- KNN-based Defense ---
    
    knn_defense = knn_based_defense(k = 5, eta = 0.6) # 5, 0.8
    cleaned_train_loader = knn_defense.apply_defense(
        X  = train_dataset.x,
        y_poisoned = poisoned_labels,
        y_clean = train_dataset.y,
    )

    model_clean , optimizer_clean = load_model_state(model, optimizer, os.path.join(save_path, f'{model.model_name}_falfa'))

    model_defensed = knn_defense.train_model_after_defense(
        model = model_clean,
        optimizer = optimizer_clean,
        cleaned_loader = cleaned_train_loader,
        num_epochs = num_epochs,
        criterion = criterion
    )

    # Evaluate the model after defense
    accuracy_defense = knn_defense.evaluate_model(
        model = model_defensed,
        test_loader = test_loader,
        device = 'cuda'
    )

    accu_defense_list.append(accuracy_defense)
    
    print(f"Accuracy after defense: {accuracy_defense:.4f}")
