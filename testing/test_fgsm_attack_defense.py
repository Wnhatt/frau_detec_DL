import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from attacks.universal_backdoor_attacks import universal_backdoor_attack
from utils.dataset import simDataset
from models.models import LSTM
from attacks.fgsm import Attack , Solver
from utils.dataset import load_model
import os

# --- Hyperparameters ---
learning_rate = 0.001
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = '/home/anhkhoa/ml4secu/save_path'
datapath = '/home/anhkhoa/ml4secu/data.csv'

# --- Load Dataset ---
train_dataset = simDataset(datapath=datapath, device=device, split='train')
test_dataset = simDataset(datapath=datapath, device=device, split='test')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

# --- Initialize Model ---
model = LSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


# model = load_model(model, os.path.join(save_path, f'{model.model_name}.pth'))

# --- Gói các tham số vào args giả lập ---
class Args:
    def __init__(self):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cuda = torch.cuda.is_available()
        self.epoch = num_epochs
        self.batch_size = 32
        self.eps = 0.03
        self.lr = learning_rate
        self.y_dim = 2
        self.dataset = 'fraud'
        self.data_loader = {
            'train': train_loader,
            'test': test_loader
        }
        self.save_path = save_path

args = Args()

# --- Khởi tạo Solver ---
solver = Solver(args)


# # --- Huấn luyện ---
# solver.train(train_loader, test_loader, num_epochs=num_epochs)


# # --- Sinh mẫu tấn công adversarial ---
# solver.generate(
#     num_sample=100,  # bạn có thể tăng số lượng
#     target=0,     # -1 nghĩa là untargeted
#     epsilon=50, # threshold 
#     alpha=0.03, # step 
#     iteration=100
# )


solver.adv_train(train_loader=train_loader, val_loader=test_loader, num_epochs=num_epochs, alpha = 0.03, eps = 0.03)
# solver.adv_test(test_loader, 'fgsm', 0.03)
solver.generate(
    num_sample=100,  # bạn có thể tăng số lượng
    epsilon=0.04, # threshold 
    alpha=0.03, # step 
    iteration=100
)


# accuracy 0.79 - 0.76 
