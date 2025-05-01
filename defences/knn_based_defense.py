import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter

def knn_based_defense(X, y, k=5, eta=0.5):
    """
    Apply kNN-based label sanitization defense.

    Parameters:
        X (ndarray): Feature vectors, shape (m, d)
        y (ndarray): Labels, shape (m,)
        k (int): Number of neighbors
        eta (float): Confidence threshold

    Returns:
        y_defended (ndarray): Possibly relabeled training labels
    """
    m = len(y)
    y_defended = np.copy(y)

    # Fit kNN model
    knn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')  # +1 because the point itself is included
    knn.fit(X)

    for i in range(m):
        # Get k nearest neighbors (excluding the point itself)
        distances, indices = knn.kneighbors(X[i].reshape(1, -1), n_neighbors=k+1)
        neighbors_idx = indices[0][1:]  # exclude itself
        neighbor_labels = y[neighbors_idx]

        # Compute label distribution and confidence
        label_counts = Counter(neighbor_labels)
        majority_label, majority_count = label_counts.most_common(1)[0]
        confidence = majority_count / k

        if confidence >= eta:
            y_defended[i] = majority_label
        # else: keep original label (already set)
        
    return y_defended