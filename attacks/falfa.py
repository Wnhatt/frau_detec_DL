import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import cvxpy as cp
from tqdm import tqdm

def predict_proba_dl(model, X_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()


def solve_eq5_lp(alpha, beta, lam, y_train, epsilon):
    n = len(y_train)
    y_var = cp.Variable(n)
    objective = cp.Minimize((alpha - beta) @ y_var)
    constraints = [
        lam @ y_var <= epsilon * n + lam @ y_train,
        y_var >= 0,
        y_var <= 1
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    if y_var.value is None:
        raise ValueError("❌ Solver failed")
    return np.round(y_var.value).astype(int)


from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import cvxpy as cp

class falfa:
    def __init__(self, model, device, epochs, epsilon, max_iter, criterion, optimizer, poisoned_labels=None):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.criterion = criterion
        self.optimizer = optimizer
        self.poisoned_labels = poisoned_labels

    def train_model_on_loader(self, dataloader, epochs=None):
        self.model.train()
        if epochs is None:
            epochs = self.epochs
        
        for _ in range(epochs):
            for i, (x_batch, y_batch) in tqdm(enumerate(dataloader), desc='Training', leave=False):
                x_batch = x_batch.to(self.device)

                if self.poisoned_labels is not None:
                    y_batch = self.poisoned_labels[i * len(y_batch):(i+1) * len(y_batch)]
                    y_batch = torch.tensor(y_batch, dtype=torch.long).to(self.device)
                else:
                    y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

    def predict_proba(self, dataloader):
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for x_batch, _ in tqdm(dataloader, desc='Predicting Probability', leave=False):
                x_batch = x_batch.to(self.device)
                logits = self.model(x_batch)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def falfa_attack_dl_dataloader(self, dataloader, X_np, y_np):
        n = len(y_np)
        d = X_np.shape[1]
        poisoned_labels = y_np.copy()
        n_flip = int(self.epsilon * n)

        # Step 1: Train clean model
        print("📌 Training clean model...")
        self.poisoned_labels = None
        self.train_model_on_loader(dataloader)

        p = self.predict_proba(dataloader)
        beta = -np.log(p[:, 1]) + np.log(1 - p[:, 1])
        lam = np.where(y_np == 1, 1, -1)

        # Step 2: Initialize label flips
        flip_idx = np.random.choice(n, n_flip, replace=False)
        poisoned_labels[flip_idx] = 1 - poisoned_labels[flip_idx]
        
        for it in range(self.max_iter):
            print(f"🔁 Iteration {it + 1}/{self.max_iter}")

            # Train model with poisoned labels
            self.poisoned_labels = poisoned_labels
            self.train_model_on_loader(dataloader)

            # Predict probabilities
            p_poisoned = self.predict_proba(dataloader)
            alpha = -np.log(p_poisoned[:, 1]) + np.log(1 - p_poisoned[:, 1])

            # Solve LP
            new_labels = self.solve_eq5_lp(alpha, beta, lam, y_np, self.epsilon)

            if np.array_equal(new_labels, poisoned_labels):
                print("✅ Converged early.")
                break

            poisoned_labels = new_labels

        return poisoned_labels

    @staticmethod
    def solve_eq5_lp(alpha, beta, lam, y_train, epsilon):
        n = len(y_train)
        y_var = cp.Variable(n)
        objective = cp.Minimize((alpha - beta) @ y_var)
        constraints = [
            cp.sum(cp.abs(y_var - y_train)) <= epsilon * n,  # số lượng nhãn bị đổi
            lam @ y_var <= epsilon * n + lam @ y_train,
            y_var >= 0,
            y_var <= 1
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        if y_var.value is None:
            raise ValueError("❌ LP Solver failed.")
        return np.round(y_var.value).astype(int)

    