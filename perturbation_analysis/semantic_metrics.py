"""
Metriche semantiche di perturbazione per valutazione embedding.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

def global_perturbation_semantic_alignment(de_matrix, emb_centroids, method='spearman'):
    """
    Calcola la correlazione tra la matrice di similarità biologica (DE) e quella embedding.
    de_matrix: (n_pert, n_genes) DataFrame o ndarray
    emb_centroids: (n_pert, n_dim) DataFrame o ndarray
    method: 'spearman' o 'pearson'
    Ritorna: correlazione tra upper triangle delle due matrici di similarità
    """
    # Similarità biologica (correlazione tra profili DE)
    bio_sim = np.corrcoef(de_matrix) if method == 'pearson' else spearmanr(de_matrix, axis=1).correlation
    # Similarità embedding (coseno)
    emb_sim = cosine_similarity(emb_centroids)
    # Upper triangle
    iu = np.triu_indices_from(bio_sim, k=1)
    corr, _ = spearmanr(bio_sim[iu], emb_sim[iu])
    return corr

def save_metric_per_layer(results_dict, out_csv):
    """
    Salva un dizionario layer:metrica in un csv.
    """
    df = pd.DataFrame(list(results_dict.items()), columns=['Layer', 'SemanticAlignment'])
    df.to_csv(out_csv, index=False)
