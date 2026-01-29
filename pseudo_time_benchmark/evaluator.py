import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import issparse

class EmbeddingEvaluator:
    def __init__(self, adata, root_cell_index):
        self.adata = adata
        self.adata.uns['iroot'] = root_cell_index
        self.ref_pseudotime = None
        self.ref_neighbors_indices = None
        self.k_neighbors = 15

    def setup_reference(self, use_rep='X_pca', n_neighbors=30):
        """Calcola DPT e Neighbors sul riferimento (Ground Truth)."""
        print(f"--- Building Ground Truth Graph (k={n_neighbors}, rep={use_rep}) ---")
        self.k_neighbors = n_neighbors
        
        # Neighbors & Diffmap & DPT
        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, use_rep=use_rep, key_added='ref')
        sc.tl.diffmap(self.adata, neighbors_key='ref')
        sc.tl.dpt(self.adata, neighbors_key='ref')
        
        self.ref_pseudotime = self.adata.obs['dpt_pseudotime'].values
        
        # Cache neighbors indices for Metric 2
        distances = self.adata.obsp['ref_distances']
        self.ref_neighbors_indices = self._get_knn_indices(distances, n_neighbors)

    def _get_knn_indices(self, adjacency_matrix, k):
        """Estrae indici per overlap veloce."""
        n_samples = adjacency_matrix.shape[0]
        indices = np.zeros((n_samples, k), dtype=int)
        if issparse(adjacency_matrix):
            for i in range(n_samples):
                row_indices = adjacency_matrix[i].indices
                limit = min(len(row_indices), k)
                indices[i, :limit] = row_indices[:limit]
        return indices

    def _metric_1_ordering(self, layer_key):
        """Spearman correlation Reference vs Embedding Pseudotime."""
        sc.pp.neighbors(self.adata, use_rep=layer_key, n_neighbors=self.k_neighbors, key_added='temp')
        sc.tl.diffmap(self.adata, neighbors_key='temp')
        sc.tl.dpt(self.adata, neighbors_key='temp')
        
        layer_dpt = self.adata.obs['dpt_pseudotime'].values
        valid = np.isfinite(self.ref_pseudotime) & np.isfinite(layer_dpt)
        corr, _ = spearmanr(self.ref_pseudotime[valid], layer_dpt[valid])
        return corr

    def _metric_2_continuity(self, layer_key):
        """Neighborhood Overlap."""
        if 'temp_distances' not in self.adata.obsp:
             sc.pp.neighbors(self.adata, use_rep=layer_key, n_neighbors=self.k_neighbors, key_added='temp')
        
        layer_indices = self._get_knn_indices(self.adata.obsp['temp_distances'], self.k_neighbors)
        
        overlaps = []
        for i in range(self.adata.shape[0]):
            intersect = np.intersect1d(self.ref_neighbors_indices[i], layer_indices[i])
            overlaps.append(len(intersect) / self.k_neighbors)
        return np.mean(overlaps)

    def _metric_3_geometry(self, layer_key, n_pairs=5000):
        """Correlation Delta-Time vs Embedding Distance."""
        n_cells = self.adata.shape[0]
        idx1 = np.random.randint(0, n_cells, n_pairs)
        idx2 = np.random.randint(0, n_cells, n_pairs)
        
        delta_time = np.abs(self.ref_pseudotime[idx1] - self.ref_pseudotime[idx2])
        emb = self.adata.obsm[layer_key]
        emb_dists = np.linalg.norm(emb[idx1] - emb[idx2], axis=1)
        
        valid = np.isfinite(delta_time)
        corr, _ = spearmanr(delta_time[valid], emb_dists[valid])
        return corr

    def evaluate_layer(self, layer_key):
        print(f"Testing: {layer_key}")
        # Controllo NaN nell'embedding
        arr = self.adata.obsm[layer_key]
        from scipy.sparse import issparse
        if issparse(arr):
            arr = arr.toarray()
        n_nans = np.isnan(arr).sum()
        if n_nans > 0:
            print(f"WARNING: Layer {layer_key} contiene {n_nans} valori NaN. Layer saltato.")
            return None
        return {
            "Layer": layer_key,
            "Pseudotime_Corr": self._metric_1_ordering(layer_key),
            "Neighborhood_Overlap": self._metric_2_continuity(layer_key),
            "Global_Geom_Corr": self._metric_3_geometry(layer_key)
        }