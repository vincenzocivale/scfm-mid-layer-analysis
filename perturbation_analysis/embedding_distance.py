import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def embedding_distance_analysis(embeddings, labels):
    """
    Calcola la separabilità tra perturbazioni usando le distanze tra embeddings.
    Args:
        embeddings: np.ndarray (cells x features)
        labels: array-like (cells, gene perturbation)
    Returns:
        pd.DataFrame con dist_within, dist_between, separability per ciascun gene perturbato
    """
    unique_genes = np.unique(labels)
    results = []
    for gene in tqdm(unique_genes, desc="Embedding distance analysis"):
        idx_gene = np.where(labels == gene)[0]
        idx_other = np.where(labels != gene)[0]
        emb_gene = embeddings[idx_gene]
        emb_other = embeddings[idx_other]
        # GPU
        if GPU_AVAILABLE:
            emb_gene_gpu = cp.asarray(emb_gene)
            emb_other_gpu = cp.asarray(emb_other)
            dist_within = float(cp.mean(cp.linalg.norm(emb_gene_gpu[:, None] - emb_gene_gpu[None, :], axis=-1))) if len(idx_gene) > 1 else np.nan
            dist_between = float(cp.mean(cp.linalg.norm(emb_gene_gpu[:, None] - emb_other_gpu[None, :], axis=-1)))
        else:
            dist_within = np.mean(cdist(emb_gene, emb_gene)) if len(idx_gene) > 1 else np.nan
            dist_between = np.mean(cdist(emb_gene, emb_other))
        separability = dist_between / dist_within if dist_within > 0 else np.nan
        results.append({
            'gene': gene,
            'dist_within': dist_within,
            'dist_between': dist_between,
            'separability': separability
        })
    return pd.DataFrame(results)
