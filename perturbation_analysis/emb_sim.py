"""
Optimized embedding similarity for large datasets.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from typing import Dict, Optional


class EmbeddingSimilarity:
    """Compute embedding similarity layer-by-layer to save memory."""
    
    def __init__(self, adata: sc.AnnData, perturb_key: str = 'perturbed_gene_name', min_cells: int = 5):
        self.adata = adata
        self.perturb_key = perturb_key
        self.min_cells = min_cells
        
        # Get valid perturbations
        perturb_counts = adata.obs[perturb_key].value_counts()
        self.perturbations = sorted([
            p for p in perturb_counts.index 
            if pd.notna(p) and perturb_counts[p] >= min_cells
        ])
        
        self.labels = adata.obs[perturb_key].values
        self.layer_embeddings = None
    
    def compute_single_layer_similarity(self, layer_name: str, 
                                       perturbations: Optional[list] = None, 
                                       device: str = 'cuda',
                                       dtype: str = 'float32') -> pd.DataFrame:
        """Compute similarity for ONE layer only, using PyTorch and GPU if available. dtype: 'float32' or 'float16'"""
        if layer_name not in self.adata.obsm:
            raise ValueError(f"Layer {layer_name} not found")

        embeddings = self.adata.obsm[layer_name]
        if perturbations is None:
            perturbations = self.perturbations

        # Compute centroids
        centroids = []
        valid_perturbs = []
        for perturb in perturbations:
            mask = self.labels == perturb
            if mask.sum() >= self.min_cells:
                centroids.append(embeddings[mask].mean(axis=0))
                valid_perturbs.append(perturb)

        if not centroids:
            raise ValueError("No valid perturbations with enough cells.")

        # Convert to torch tensor and move to GPU if available
        torch_dtype = torch.float16 if dtype == 'float16' else torch.float32
        centroids_torch = torch.tensor(np.array(centroids), dtype=torch_dtype)
        if device == 'cuda' and torch.cuda.is_available():
            centroids_torch = centroids_torch.cuda()

        # Compute cosine similarity matrix using torch
        sim_matrix = F.cosine_similarity(centroids_torch.unsqueeze(1), centroids_torch.unsqueeze(0), dim=-1)
        sim_matrix = sim_matrix.detach().cpu().numpy().astype(np.float16 if dtype == 'float16' else np.float32)

        return pd.DataFrame(sim_matrix, index=valid_perturbs, columns=valid_perturbs)
    
    def compute_similarity_matrices(self) -> Dict[int, pd.DataFrame]:
        """Compute similarity matrices for all layers (memory intensive)."""
        # Accetta sia 'X_scgpt_' che 'X_layer_'
        layer_names = sorted([k for k in self.adata.obsm.keys() if k.startswith('X_scgpt_') or k.startswith('X_layer_')])

        from sklearn.metrics.pairwise import cosine_similarity

        self.layer_embeddings = {
            int(name.split('_')[-1]): self.adata.obsm[name]
            for name in layer_names
        }

        similarities = {}
        for layer_num, embeddings in self.layer_embeddings.items():
            centroids = []
            for perturb in self.perturbations:
                mask = self.labels == perturb
                centroids.append(embeddings[mask].mean(axis=0))
            centroids = np.array(centroids)
            sim_matrix = cosine_similarity(centroids)
            similarities[layer_num] = pd.DataFrame(
                sim_matrix, index=self.perturbations, columns=self.perturbations
            )
        return similarities