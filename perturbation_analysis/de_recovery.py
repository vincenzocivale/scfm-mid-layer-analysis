import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def compute_de_recovery(true_de, pred_de, method='pearson'):
    """
    Calcola la correlazione tra DE ground truth e DE predetto dagli embeddings.
    Args:
        true_de: array-like, valori DE reali
        pred_de: array-like, valori DE predetti dagli embeddings
        method: 'pearson' o 'spearman'
    Returns:
        correlation: float
        p_value: float
    """
    # Se pred_de è un dict di layer, calcola per ciascuno
    from tqdm import tqdm
    if isinstance(pred_de, dict):
        results = {}
        for layer, pred in tqdm(pred_de.items(), desc="DE recovery per layer"):
            if method == 'pearson':
                corr, p = pearsonr(true_de, pred)
            else:
                corr, p = spearmanr(true_de, pred)
            results[layer] = (corr, p)
        return results
    if method == 'pearson':
        corr, p = pearsonr(true_de, pred_de)
    else:
        corr, p = spearmanr(true_de, pred_de)
    return corr, p
