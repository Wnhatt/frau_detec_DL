import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve

# class simDataset(Dataset):
#     def __init__(self, datapath, device, split):
#         super().__init__()
#         df = pd.read_csv(datapath)
#         self.x , self.y = self.transform(df, split)
#         self.device = device

        
#     def transform(self, df, split):
#         # X = df
#         # Y = X['isFraud']
#         # del X['isFraud']
#         X = df


#         Y = X['isFraud']
#         del X['isFraud']
        
#         train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.2)


#         if split == 'train':
#             return train_X.values, train_Y.values  # Return NumPy arrays for training
#         else:
#             return test_X.values, test_Y.values    # Return NumPy arrays for testing
        
#     def get_data(self):
#         return self.x, self.y

#     def __len__(self):
#         return len(self.x)
    
#     def __getitem__(self, idx):
        
#         x = torch.tensor(self.x[idx], dtype=torch.float).to(self.device) # 7
#         y = torch.tensor(self.y[idx], dtype=torch.float).to(self.device) # 1
        
#         return x.unsqueeze(-1), y


# class simDataset(Dataset):
#     def __init__(self, datapath = None, device = None, split = None, X=None, y=None):
#         super().__init__()
#         self.device = device

#         if datapath is not None:
#             df = pd.read_csv(datapath)
#             self.x, self.y = self.transform(df, split)
#         elif X is not None and y is not None:
#             self.x = X
#             self.y = y
#         else:
#             raise ValueError("Bạn phải cung cấp datapath hoặc cả X và y.")

#     def transform(self, df, split):
#         X = df
#         Y = X['isFraud']
#         del X['isFraud']

#         train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)

#         if split == 'train':
#             return train_X.values, train_Y.values
#         else:
#             return test_X.values, test_Y.values

#     def get_data(self):
#         return self.x, self.y

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         x = torch.tensor(self.x[idx], dtype=torch.float).to(self.device)
#         y = torch.tensor(self.y[idx], dtype=torch.int16).to(self.device)
        

#         return x, y


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
from torch.utils.data import Dataset

class simDataset(Dataset):
    scaler = None  # Static scaler, shared between train and test

    def __init__(self, datapath=None, device=None, split=None, X=None, y=None):
        super().__init__()
        self.device = device

        if datapath is not None:
            df = pd.read_csv(datapath)
            self.x, self.y = self.transform(df, split)
        elif X is not None and y is not None:
            self.x = X
            self.y = y
        else:
            raise ValueError("Bạn phải cung cấp datapath hoặc cả X và y.")

    def transform(self, df, split):
        X = df.copy()
        Y = X['isFraud']
        X.drop(columns=['isFraud'], inplace=True)

        # Train/test split
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

        if split == 'train':
            if simDataset.scaler is None:
                simDataset.scaler = MinMaxScaler()
                train_X_scaled = simDataset.scaler.fit_transform(train_X)
            else:
                train_X_scaled = simDataset.scaler.transform(train_X)
            return train_X_scaled, train_Y.values
        else:
            if simDataset.scaler is None:
                raise ValueError("Scaler chưa được fit. Vui lòng load tập train trước.")
            test_X_scaled = simDataset.scaler.transform(test_X)
            return test_X_scaled, test_Y.values

    def get_data(self):
        return self.x, self.y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float).to(self.device)
        y = torch.tensor(self.y[idx], dtype=torch.int64).to(self.device)
        return x, y

import torch.serialization

# Allowlist TensorDataset globally
torch.serialization.add_safe_globals([torch.utils.data.dataset.TensorDataset])

def load_model(model, model_path):
    """
    Load a model from a specified path.
    
    Args:
        model (torch.nn.Module): The model to load the state into.
        model_path (str): Path to the saved model file.
        
    Returns:
        torch.nn.Module: The model with loaded state.
    """
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    return model