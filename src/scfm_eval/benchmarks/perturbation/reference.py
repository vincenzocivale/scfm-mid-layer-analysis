"""
Computes the biological reference similarity matrix based on differential expression.
"""
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from typing import Dict, Literal
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scanpy')


class ReferenceBuilder:
    """
    Builds a biological similarity reference between perturbations using
    their differential expression (DE) profiles.
    """
    
    def __init__(self, adata: sc.AnnData, perturb_key: str, control_label: str = 'control'):
        """
        Args:
            adata: AnnData object with raw counts in .X or .raw.
            perturb_key: Column in adata.obs with perturbation labels.
            control_label: The label for control cells.
        """
        if perturb_key not in adata.obs.columns:
            raise ValueError(f"'{perturb_key}' not in adata.obs")

        self.adata = adata
        self.perturb_key = perturb_key
        self.control_label = control_label
        self.de_profiles = None

        # Get perturbations with enough cells
        perturb_counts = adata.obs[perturb_key].value_counts()
        self.perturbations = sorted([
            p for p in perturb_counts.index 
            if pd.notna(p) and p != control_label and perturb_counts[p] >= 5
        ])

        # Cache control expression
        control_mask = self.adata.obs[self.perturb_key] == self.control_label
        if control_mask.sum() == 0:
            raise ValueError(f"No control cells with label '{self.control_label}' found.")
        
        # Use .raw if available, otherwise .X
        if self.adata.raw is not None:
            print("Using .raw for DE calculation.")
            self.rna_adata = self.adata.raw.to_adata()
        else:
            print("Using .X for DE calculation.")
            self.rna_adata = self.adata

        self.control_expr = self.rna_adata[control_mask].X.toarray() if hasattr(self.rna_adata[control_mask].X, 'toarray') else self.rna_adata[control_mask].X


    def _compute_single_de_profile(self, perturb: str, n_top_genes: int = 500) -> pd.Series:
        """Computes DE profile for a single perturbation vs control."""
        perturb_mask = self.rna_adata.obs[self.perturb_key] == perturb
        perturb_expr = self.rna_adata[perturb_mask].X
        if hasattr(perturb_expr, 'toarray'):
            perturb_expr = perturb_expr.toarray()

        # Using torch for speed
        control_tensor = torch.tensor(self.control_expr, dtype=torch.float32)
        perturb_tensor = torch.tensor(perturb_expr, dtype=torch.float32)

        control_mean = control_tensor.mean(dim=0)
        perturb_mean = perturb_tensor.mean(dim=0)
        
        # Calculate log fold change and variance for ranking
        log_fold_change = perturb_mean - control_mean
        
        # Simple t-statistic for ranking genes
        control_var = control_tensor.var(dim=0)
        perturb_var = perturb_tensor.var(dim=0)
        n_control = control_tensor.shape[0]
        n_perturb = perturb_tensor.shape[0]
        
        pooled_std = torch.sqrt(control_var / n_control + perturb_var / n_perturb)
        t_stat = torch.abs(log_fold_change / (pooled_std + 1e-8))

        # Select top genes based on t-statistic
        top_indices = torch.topk(t_stat, min(n_top_genes, len(t_stat))).indices.numpy()
        
        # Get DE values (log fold change) for top genes
        de_values = log_fold_change[top_indices].numpy()
        gene_names = self.rna_adata.var_names[top_indices].to_list()
        
        return pd.Series(de_values, index=gene_names)

    def compute_de_profiles(self, n_top_genes: int = 500) -> pd.DataFrame:
        """
        Computes DE profiles for all perturbations.
        Each profile is a vector of log-fold-changes for the top DE genes.
        """
        print(f"Computing DE profiles for {len(self.perturbations)} perturbations...")
        profiles = {}
        for perturb in tqdm(self.perturbations, desc="Computing DE profiles"):
            profiles[perturb] = self._compute_single_de_profile(perturb, n_top_genes)

        # Combine profiles into a single DataFrame
        all_genes = sorted(set.union(*[set(p.index) for p in profiles.values()]))
        self.de_profiles = pd.DataFrame({
            p: prof.reindex(all_genes, fill_value=0)
            for p, prof in profiles.items()
        }).T
        
        print("DE profiles computed.")
        return self.de_profiles

    def compute_similarity_matrix(self, method: Literal['spearman', 'pearson'] = 'spearman') -> pd.DataFrame:
        """
        Computes the similarity matrix from DE profiles using correlation.
        """
        if self.de_profiles is None:
            raise ValueError("DE profiles not computed. Run compute_de_profiles() first.")

        print(f"Computing {method} correlation matrix...")
        if method == 'spearman':
            # Rank data for Spearman correlation
            ranked_de = self.de_profiles.rank(axis=1, method='average')
            # Cosine similarity on ranked data is equivalent to Spearman
            ranked_tensor = torch.tensor(ranked_de.values, dtype=torch.float32)
            sim_matrix = F.cosine_similarity(ranked_tensor.unsqueeze(1), ranked_tensor.unsqueeze(0), dim=-1)
            sim_matrix = sim_matrix.cpu().numpy()
        else: # Pearson
            de_tensor = torch.tensor(self.de_profiles.values, dtype=torch.float32)
            # Center data for Pearson correlation
            de_tensor_centered = de_tensor - de_tensor.mean(dim=1, keepdim=True)
            sim_matrix = F.cosine_similarity(de_tensor_centered.unsqueeze(1), de_tensor_centered.unsqueeze(0), dim=-1)
            sim_matrix = sim_matrix.cpu().numpy()

        return pd.DataFrame(sim_matrix, index=self.de_profiles.index, columns=self.de_profiles.index)

    def build(self, de_top_n: int = 500, corr_method: Literal['spearman', 'pearson'] = 'spearman'):
        """
        Main method to build the reference similarity matrix.
        """
        self.compute_de_profiles(n_top_genes=de_top_n)
        similarity_matrix = self.compute_similarity_matrix(method=corr_method)
        return similarity_matrix
