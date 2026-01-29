import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def compute_silhouette(embeddings, labels):
    """
    embeddings: np.ndarray (cells x features)
    labels: array-like (cells, cluster/perturbation labels)
    Returns: float (silhouette score)
    """
    # Se embeddings è una lista di layer, calcola per ciascuno
    if isinstance(embeddings, dict):
        scores = {}
        for layer, emb in tqdm(embeddings.items(), desc="Silhouette score per layer"):
            scores[layer] = silhouette_score(emb, labels)
        return scores
    return silhouette_score(embeddings, labels)


def compute_knn_purity(embeddings, labels, k=10):
    """
    embeddings: np.ndarray (cells x features)
    labels: array-like (cells, cluster/perturbation labels)
    k: int, number of neighbors
    Returns: float (mean purity)
    """
    # Se embeddings è una lista di layer, calcola per ciascuno
    if isinstance(embeddings, dict):
        purities = {}
        for layer, emb in tqdm(embeddings.items(), desc="kNN purity per layer"):
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(emb)
            _, indices = nbrs.kneighbors(emb)
            purity_list = []
            for i, neighbors in enumerate(indices):
                neighbor_labels = labels[neighbors[1:]]
                purity = np.mean(neighbor_labels == labels[i])
                purity_list.append(purity)
            purities[layer] = np.mean(purity_list)
        return purities
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)
    purity_list = []
    for i, neighbors in enumerate(indices):
        neighbor_labels = labels[neighbors[1:]]
        purity = np.mean(neighbor_labels == labels[i])
        purity_list.append(purity)
    return np.mean(purity_list)
