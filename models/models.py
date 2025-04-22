import torch.nn as nn
import torch
import torch.nn.functional as F

class neuralNetwork(nn.Module):
    def __init__(self):
        super(neuralNetwork, self).__init__()
        self.fc1 = nn.Linear(7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)
        self.fc4 = nn.Linear(32, 10)
        self.fc5 = nn.Linear(10, 5)
        self.fc6 = nn.Linear(5,1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        x = F.relu(self.fc6(x))
        return x



class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=4, num_classes=2, bidirectional=True):
        
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(2 * hidden_size if bidirectional else hidden_size, num_classes)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.model_name = 'LSTM'
        self.bidirectional = bidirectional
    def forward(self, x):
        # Khởi tạo hidden state và cell state
        # x shape: (B, T, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  # out shape: (B, T, hidden_size)
        out = out[:, -1, :]              # Lấy output ở time step cuối
        out = self.fc(out)

        
        return out

    def forward_embeddings(self, x):
        """
        Trả về đặc trưng từ tầng LSTM, dùng cho Spectral Signature Defense.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]  # (batch_size, hidden_size * num_directions)