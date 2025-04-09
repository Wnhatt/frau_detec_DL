from torch.utils.data import Dataset, DataLoader, random_split

import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataset import simDataset
from models.models import neuralNetwork, LSTM

learning_rate = 0.001
num_epochs = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datapath = '/home/anhkhoa/ml4secu/data.csv'  # Replace with your actual file path
print(device) 

# 1. Initialize Dataset
train_dataset = simDataset(datapath=datapath, device=device, split='train')
test_dataset = simDataset(datapath=datapath, device=device, split='test')

# 2. Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("Độ dài training: ",len(train_dataset))
print("Độ dài testing: ",len(test_dataset))
# We initialize the model
model = LSTM().to(device)

# Loss function and optimizer
# criterion = nn.BCEWithLogitsLoss().to(device)
criterion = nn.CrossEntropyLoss() # Others loss function use
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

from tqdm import tqdm

# Train the network
model.train()  # Set model to training mode
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(train_loader, 
                       desc=f'Epoch {epoch+1}/{num_epochs}', 
                       unit='batch')
    
    for batch_id, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.type_as(outputs)
        loss = criterion(outputs.squeeze(), labels.long())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': running_loss/(batch_id+1)})
    
    print(f'Epoch {epoch+1} - Avg Loss: {running_loss/len(train_loader):.4f}')

print('Training ended')

# Evaluation
model.eval()  # Set model to evaluation mode
correct_pred = 0
total_pred = 0

with torch.no_grad():
    test_bar = tqdm(test_loader, desc='Testing', unit='batch')
    for inputs, labels in test_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        total_pred += labels.size(0)
        correct_pred += (pred == labels).sum().item()
        test_bar.set_postfix({'acc': f'{100*correct_pred/total_pred:.2f}%'})

print(f'\nFinal Test Accuracy: {100 * correct_pred / total_pred:.2f}%')


