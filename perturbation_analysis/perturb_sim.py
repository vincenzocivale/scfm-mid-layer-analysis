"""
Biological similarity computation for perturbation analysis.
Supports DE-based and pathway-based similarity metrics.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from typing import Dict, Literal, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm


class PerturbationSimilarity:
    """
    Compute biological similarity between perturbations using:
    1. Differential expression profiles
    2. Pathway/GO annotations (optional)
    """
    
    def __init__(self, adata: sc.AnnData, perturb_key: str = 'perturbation', 
        control_label: str = 'control', min_cells: int = 5):
        self.adata = adata
        self.perturb_key = perturb_key
        self.control_label = control_label
        self.min_cells = min_cells
        
        if perturb_key not in adata.obs.columns:
            raise ValueError(f"'{perturb_key}' not in adata.obs")
        
        perturb_counts = adata.obs[perturb_key].value_counts()
        self.perturbations = sorted([
            p for p in perturb_counts.index 
            if pd.notna(p) and p != control_label and perturb_counts[p] >= min_cells
        ])
        
        self.de_profiles = None
        
        # Cache control expression (AGGIUNTO)
        control_mask = self.adata.obs[self.perturb_key] == self.control_label
        if control_mask.sum() == 0:
            raise ValueError(f"No control cells with label '{self.control_label}'")
        
        control_expr = self.adata[control_mask].X
        if hasattr(control_expr, 'toarray'):
            control_expr = control_expr.toarray()
        self.control_expr = control_expr

    def _compute_single_de_profile(self, perturb: str, n_top_genes: int = 500, device: str = 'cpu') -> pd.Series:
        """Fast DE using vectorized t-test, CPU only."""
        perturb_mask = self.adata.obs[self.perturb_key] == perturb
        perturb_expr = self.adata[perturb_mask].X
        if hasattr(perturb_expr, 'toarray'):
            perturb_expr = perturb_expr.toarray()

        # Move to torch tensor (CPU only)
        control_tensor = torch.tensor(self.control_expr, dtype=torch.float32)
        perturb_tensor = torch.tensor(perturb_expr, dtype=torch.float32)

        control_mean = control_tensor.mean(dim=0)
        control_std = control_tensor.std(dim=0)
        perturb_mean = perturb_tensor.mean(dim=0)
        perturb_std = perturb_tensor.std(dim=0)
        n1, n2 = control_tensor.shape[0], perturb_tensor.shape[0]
        pooled_std = torch.sqrt(((n1-1)*control_std**2 + (n2-1)*perturb_std**2) / (n1+n2-2))
        denom = pooled_std * torch.sqrt(torch.tensor(1.0/n1 + 1.0/n2)) + 1e-10
        t_stat = torch.abs((perturb_mean - control_mean) / denom)

        # Top genes by |t-stat|
        top_idx = torch.topk(t_stat, n_top_genes).indices.cpu().numpy()
        gene_mask = np.zeros(self.adata.n_vars, dtype=bool)
        gene_mask[top_idx] = True
        # Z-scores sui top genes
        z_scores = ((perturb_mean - control_mean) / (control_std + 1e-8))[top_idx].detach().cpu().numpy()
        return pd.Series(z_scores, index=self.adata.var_names[top_idx])
    
    def compute_de_profiles(self, n_top_genes: int = 500, per_perturb: bool = True) -> pd.DataFrame:
        """Optimized with optional parallelization."""
        # Cache control
        control_mask = self.adata.obs[self.perturb_key] == self.control_label
        self.control_expr = self.adata[control_mask].X
        if hasattr(self.control_expr, 'toarray'):
            self.control_expr = self.control_expr.toarray()
        
        if per_perturb:
            # Sequential is often faster for <100 perturbations
            profiles = {}
            for perturb in tqdm(self.perturbations, desc="DE profiles"):
                profiles[perturb] = self._compute_single_de_profile(perturb, n_top_genes, device='cpu')

            # Union genes
            all_genes = sorted(set.union(*[set(p.index) for p in profiles.values()]))
            self.de_profiles = pd.DataFrame({
                p: prof.reindex(all_genes, fill_value=0)
                for p, prof in profiles.items()
            }).T
        else:
            # HVG fallback...
            pass
        
        return self.de_profiles
    
    def compute_similarity_matrix(self, normalize: bool = False, chunk_size: int = 100, device: str = 'cpu') -> pd.DataFrame:
        """
        Compute Spearman correlation between DE profiles con chunking.
        
        Args:
            normalize: normalizza in [0,1]
            chunk_size: processa N perturbazioni alla volta
            device: 'cpu' o 'cuda' (usa cpu per dataset grandi)
        
        Returns:
            Symmetric similarity matrix (n_perturbations × n_perturbations)
        """
        if self.de_profiles is None:
            raise ValueError("Run compute_de_profiles() first")
        
        n_perturbs = len(self.perturbations)
        
        # Precomputa ranks (una volta sola)
        de_mat = self.de_profiles.values
        from scipy.stats import rankdata
        ranked = np.apply_along_axis(rankdata, 1, de_mat)
        ranked = ranked - ranked.mean(axis=1, keepdims=True)
        
        # Matrice di output
        sim_matrix = np.zeros((n_perturbs, n_perturbs), dtype=np.float32)
        
        # Processa a chunk
        for i in tqdm(range(0, n_perturbs, chunk_size), desc="Computing similarity"):
            end_i = min(i + chunk_size, n_perturbs)
            chunk_i = ranked[i:end_i]
            
            # Converte a torch
            chunk_i_tensor = torch.tensor(chunk_i, dtype=torch.float32)
            ranked_tensor = torch.tensor(ranked, dtype=torch.float32)
            
            if device == 'cuda' and torch.cuda.is_available():
                chunk_i_tensor = chunk_i_tensor.cuda()
                ranked_tensor = ranked_tensor.cuda()
            
            # Cosine similarity tra chunk e tutti
            sim = F.cosine_similarity(
                chunk_i_tensor.unsqueeze(1),  # (chunk_size, 1, n_genes)
                ranked_tensor.unsqueeze(0),   # (1, n_perturbs, n_genes)
                dim=-1
            )
            
            sim_matrix[i:end_i, :] = sim.detach().cpu().numpy()
        
        if normalize:
            sim_matrix = (sim_matrix + 1) / 2
        
        return pd.DataFrame(sim_matrix, index=self.perturbations, columns=self.perturbations)

    def compute_control_direction_consistency(
        self, 
        emb_sim: 'EmbeddingSimilarity',
        device: str = 'cuda'
    ) -> pd.DataFrame:
        """
        Measure alignment of perturbation→control direction with DE direction.
        
        Returns:
            DataFrame: [layer, avg_cosine_de, avg_distance_to_control]
        """
        if self.de_profiles is None:
            raise ValueError("Run compute_de_profiles() first")
        control_mask = self.adata.obs[self.perturb_key] == self.control_label
        results = []
        for layer, embeddings in emb_sim.layer_embeddings.items():
            # Move to torch
            emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
            if device == 'cuda' and torch.cuda.is_available():
                emb_tensor = emb_tensor.cuda()
            control_centroid = emb_tensor[control_mask].mean(dim=0)
            distances = []
            for perturb in self.perturbations:
                perturb_mask = torch.tensor(emb_sim.labels == perturb)
                if device == 'cuda' and torch.cuda.is_available():
                    perturb_mask = perturb_mask.cuda()
                perturb_centroid = emb_tensor[perturb_mask].mean(dim=0)
                emb_direction = perturb_centroid - control_centroid
                distance = torch.norm(emb_direction).item()
                distances.append(distance)
            results.append({
                'layer': layer,
                'avg_distance_to_control': float(np.mean(distances)),
                'std_distance': float(np.std(distances))
            })
        return pd.DataFrame(results)


    def compute_direction_consistency_across_layers(
        self,
        emb_sim: 'EmbeddingSimilarity',
        device: str = 'cuda'
    ) -> pd.DataFrame:
        """
        Measure how consistent perturbation→control direction is across layers.
        
        Returns:
            DataFrame: [layer, avg_direction_consistency]
        """
        control_mask = self.adata.obs[self.perturb_key] == self.control_label
        # Compute directions for all layers
        all_directions = {}
        for layer, embeddings in emb_sim.layer_embeddings.items():
            emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
            if device == 'cuda' and torch.cuda.is_available():
                emb_tensor = emb_tensor.cuda()
            control_centroid = emb_tensor[control_mask].mean(dim=0)
            layer_directions = {}
            for perturb in self.perturbations:
                perturb_mask = torch.tensor(emb_sim.labels == perturb)
                if device == 'cuda' and torch.cuda.is_available():
                    perturb_mask = perturb_mask.cuda()
                perturb_centroid = emb_tensor[perturb_mask].mean(dim=0)
                direction = perturb_centroid - control_centroid
                direction_norm = direction / (torch.norm(direction) + 1e-8)
                layer_directions[perturb] = direction_norm.detach().cpu().numpy()
            all_directions[layer] = layer_directions
        # Measure consistency: cosine between consecutive layers
        layers = sorted(all_directions.keys())
        results = []
        for i, layer in enumerate(layers[1:], 1):
            prev_layer = layers[i-1]
            cosines = []
            for perturb in self.perturbations:
                dir_prev = all_directions[prev_layer][perturb]
                dir_curr = all_directions[layer][perturb]
                cosine = np.dot(dir_prev, dir_curr)
                cosines.append(cosine)
            results.append({
                'layer': layer,
                'direction_consistency': float(np.mean(cosines)),
                'std_consistency': float(np.std(cosines))
            })
        return pd.DataFrame(results)
    
    def save_similarity(self, output_path: Path, matrix: pd.DataFrame):
        """Save similarity matrix to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(output_path)