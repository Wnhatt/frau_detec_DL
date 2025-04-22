import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from attacks.universal_backdoor_attacks import universal_backdoor_attack
from utils.dataset import simDataset
from models.models import LSTM
from attacks.fgsm import Attack , Solver

# --- Hyperparameters ---
learning_rate = 0.001
num_epochs = 1
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

# --- Huấn luyện ---
solver.train(train_loader, test_loader, num_epochs=num_epochs)

# --- Đánh giá ---
solver.test(test_loader)

# --- Sinh mẫu tấn công adversarial ---
solver.generate(
    num_sample=100,  # bạn có thể tăng số lượng
    target=0,     # -1 nghĩa là untargeted
    epsilon=0.1,
    alpha=0.1,
    iteration=1000
)

# abc
