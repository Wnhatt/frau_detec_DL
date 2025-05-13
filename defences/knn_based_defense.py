import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset

class knn_based_defense:
    def __init__(self, k = 10, eta = 0.9):
        """
        Initialize kNN-based label sanitization defense.

        Parameters:
            k (int): Number of neighbors
            eta (float): Confidence threshold
        """
        self.k = k
        self.eta = eta
        self.device = 'cuda'
    
    def defense_mechanic(self, X, y):
        m = len(y)
        y_defended = np.copy(y)

        # Fit kNN model
        knn = NearestNeighbors(n_neighbors=self.k+1, metric='euclidean')  # +1 because the point itself is included
        knn.fit(X)

        for i in range(m):
            # Get k nearest neighbors (excluding the point itself)
            distances, indices = knn.kneighbors(X[i].reshape(1, -1), n_neighbors=self.k+1)
            neighbors_idx = indices[0][1:]  # exclude itself
            neighbor_labels = y[neighbors_idx]

            # Compute label distribution and confidence
            label_counts = Counter(neighbor_labels)
            majority_label, majority_count = label_counts.most_common(1)[0]
            confidence = majority_count / self.k

            if confidence >= self.eta:
                y_defended[i] = majority_label
            # else: keep original label (already set)
        
        return y_defended

    def apply_defense(self, X, y_poisoned, y_clean=None):
        y_defended_np = self.defense_mechanic(X, y_poisoned)
        num_relabel = (y_defended_np != y_poisoned).sum()
        print(f"🔁 Number of relabeled samples by kNN defense: {num_relabel}")

        if y_clean is not None:
            relabel_idx = (y_defended_np != y_poisoned)
            correct_relabel = (y_defended_np[relabel_idx] == y_clean[relabel_idx]).sum()
            precision = correct_relabel / relabel_idx.sum() if relabel_idx.sum() > 0 else 0
            print(f"✅ Precision of relabeled samples: {precision:.4f} ({correct_relabel}/{relabel_idx.sum()})")

        X_tensor_defended = torch.tensor(X, dtype=torch.float32)
        y_tensor_defended = torch.tensor(y_defended_np, dtype=torch.long)

        cleaned_dataset = TensorDataset(X_tensor_defended, y_tensor_defended)
        cleaned_loader = DataLoader(cleaned_dataset, batch_size=32, shuffle=True)

        return cleaned_loader


    def train_model_after_defense(self, model, cleaned_loader, optimizer, num_epochs, criterion):
        """
        Train the model on the cleaned data.

        Parameters:
            model: The model to be trained.
            cleaned_loader: DataLoader for the cleaned dataset.
            optimizer: Optimizer for the model.
            num_epochs (int): Number of epochs to train.
        """
        model.train()
        for epoch in range(num_epochs):
            for x_batch, y_batch in cleaned_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
        
        return model
    
    def evaluate_model(self, model, test_loader, device):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        return accuracy



# def knn_based_defense(X, y, k=5, eta=0.5):
#     """
#     Apply kNN-based label sanitization defense.

#     Parameters:
#         X (ndarray): Feature vectors, shape (m, d)
#         y (ndarray): Labels, shape (m,)
#         k (int): Number of neighbors
#         eta (float): Confidence threshold

#     Returns:
#         y_defended (ndarray): Possibly relabeled training labels
#     """
#     m = len(y)
#     y_defended = np.copy(y)

#     # Fit kNN model
#     knn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')  # +1 because the point itself is included
#     knn.fit(X)

#     for i in range(m):
#         # Get k nearest neighbors (excluding the point itself)
#         distances, indices = knn.kneighbors(X[i].reshape(1, -1), n_neighbors=k+1)
#         neighbors_idx = indices[0][1:]  # exclude itself
#         neighbor_labels = y[neighbors_idx]

#         # Compute label distribution and confidence
#         label_counts = Counter(neighbor_labels)
#         majority_label, majority_count = label_counts.most_common(1)[0]
#         confidence = majority_count / k

#         if confidence >= eta:
#             y_defended[i] = majority_label
#         # else: keep original label (already set)
    



#     return y_defended