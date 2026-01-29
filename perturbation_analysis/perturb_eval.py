"""
Optimized evaluation with parallelizable held-out splits.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from typing import Dict, Literal


class SimilarityEvaluator:
    """Evaluate embedding similarity with parallel support."""
    
    def __init__(self, reference_similarity: pd.DataFrame, 
                 embedding_similarities: Dict[int, pd.DataFrame]):
        self.reference = reference_similarity
        self.perturbations = list(reference_similarity.index)
        
        # Reindex embeddings
        self.embeddings = {
            layer: sim.loc[self.perturbations, self.perturbations]
            for layer, sim in embedding_similarities.items()
        }
        
        # Cache reference upper triangle
        self._ref_mask = np.triu_indices_from(self.reference.values, k=1)
        self._ref_values = self.reference.values[self._ref_mask]
    
    def compute_layer_correlations(self, method: Literal['spearman', 'pearson'] = 'spearman') -> pd.DataFrame:
        """Vectorized correlation computation."""
        corr_func = spearmanr if method == 'spearman' else pearsonr
        
        results = []
        for layer in sorted(self.embeddings.keys()):
            sim = self.embeddings[layer]
            emb_values = sim.values[self._ref_mask]
            corr, p = corr_func(self._ref_values, emb_values)
            results.append({'layer': layer, 'correlation': corr, 'p_value': p})
        
        return pd.DataFrame(results)
    
    def evaluate_single_split(self, split_idx: int, n_splits: int = 5, 
                             random_state: int = 42) -> pd.DataFrame:
        """
        Evaluate ONE held-out split (parallelizable).
        
        Returns:
            DataFrame with [layer, split, correlation]
        """
        np.random.seed(random_state)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Get this specific split
        for i, (_, test_idx) in enumerate(kf.split(range(len(self.perturbations)))):
            if i == split_idx:
                break
        
        test_perturbs = [self.perturbations[i] for i in test_idx]
        ref_subset = self.reference.loc[test_perturbs, test_perturbs].values
        mask = np.triu_indices_from(ref_subset, k=1)
        ref_vals = ref_subset[mask]
        
        results = []
        for layer, sim in self.embeddings.items():
            emb_vals = sim.loc[test_perturbs, test_perturbs].values[mask]
            if len(ref_vals) > 1:
                corr, _ = spearmanr(ref_vals, emb_vals)
                results.append({
                    'layer': layer,
                    'split': split_idx,
                    'correlation': corr
                })
        
        return pd.DataFrame(results)
    
    def evaluate_held_out_perturbations(self, n_splits: int = 5, 
                                       random_state: int = 42) -> pd.DataFrame:
        """
        Original method - sequential evaluation of all splits.
        
        Returns:
            DataFrame with aggregated held-out results
        """
        all_results = []
        
        for split_idx in range(n_splits):
            split_df = self.evaluate_single_split(split_idx, n_splits, random_state)
            all_results.append(split_df)
        
        combined = pd.concat(all_results, ignore_index=True)
        
        # Aggregate
        aggregated = combined.groupby('layer')['correlation'].agg(['mean', 'std']).reset_index()
        aggregated.columns = ['layer', 'mean_correlation', 'std_correlation']
        
        return aggregated
    
    def rank_layers_by_quality(self, correlation_df: pd.DataFrame, 
                               held_out_df: pd.DataFrame,
                               alpha: float = 0.7) -> pd.DataFrame:
        """Rank layers by combined score."""
        df = correlation_df.copy()
        
        # Merge with held-out
        held_out_dict = held_out_df.set_index('layer')['mean_correlation'].to_dict()
        df['held_out_correlation'] = df['layer'].map(held_out_dict)
        
        # Combined score
        df['full_correlation'] = df['correlation']
        df['combined_score'] = alpha * df['correlation'] + (1 - alpha) * df['held_out_correlation']
        
        return df.sort_values('combined_score', ascending=False)
    
    def compute_top_k_accuracy(self, k: int = 5) -> pd.DataFrame:
        """Vectorized top-k accuracy."""
        results = []
        
        for layer in sorted(self.embeddings.keys()):
            emb_sim = self.embeddings[layer]
            
            ref_topk = {p: set(self.reference.loc[p].nlargest(k+1).index[1:]) 
                       for p in self.perturbations}
            emb_topk = {p: set(emb_sim.loc[p].nlargest(k+1).index[1:]) 
                       for p in self.perturbations}
            
            accuracies = [len(ref_topk[p] & emb_topk[p]) / k 
                         for p in self.perturbations]
            
            results.append({
                'layer': layer, 
                f'top_{k}_accuracy': np.mean(accuracies)
            })
        
        return pd.DataFrame(results)