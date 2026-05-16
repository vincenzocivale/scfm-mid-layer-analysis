"""Biological reference similarity matrix from per-perturbation DE profiles.

Uses scanpy's Wilcoxon rank-sum (`sc.tl.rank_genes_groups`) for the per-perturbation
DE call instead of a custom t-statistic. Profiles are vectors of log-fold-change
over the *union* of all genes (no top-N truncation, so the similarity is computed
on a complete profile rather than zero-filled holes).
"""
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore', category=UserWarning, module='scanpy')


class ReferenceBuilder:
    def __init__(self, adata: sc.AnnData, perturb_key: str, control_label: str = 'control',
                 min_cells_per_perturb: int = 10):
        if perturb_key not in adata.obs.columns:
            raise ValueError(f"'{perturb_key}' not in adata.obs")
        self.adata = adata
        self.perturb_key = perturb_key
        self.control_label = control_label
        self.min_cells_per_perturb = min_cells_per_perturb
        self.de_profiles = None

        counts = adata.obs[perturb_key].value_counts()
        self.perturbations = sorted([
            p for p in counts.index
            if pd.notna(p) and p != control_label and counts[p] >= min_cells_per_perturb
        ])
        if len(self.perturbations) < 2:
            raise ValueError(f"Need ≥2 perturbations with ≥{min_cells_per_perturb} cells; got {len(self.perturbations)}")

        if (adata.obs[perturb_key] == control_label).sum() == 0:
            raise ValueError(f"No control cells with label '{control_label}'")

        self.rna_adata = adata.raw.to_adata() if adata.raw is not None else adata.copy()
        # Ensure log-normalised for Wilcoxon stability
        if 'log1p' not in self.rna_adata.uns:
            sc.pp.normalize_total(self.rna_adata, target_sum=1e4)
            sc.pp.log1p(self.rna_adata)
        self.rna_adata.obs[perturb_key] = adata.obs[perturb_key].values

    def compute_de_profiles(self) -> pd.DataFrame:
        """Run Wilcoxon per perturbation vs control. Returns (n_perturb x n_genes) LFC matrix."""
        groups = self.perturbations
        print(f"Running Wilcoxon DE for {len(groups)} perturbations vs control...")
        sub_mask = self.rna_adata.obs[self.perturb_key].isin(groups + [self.control_label])
        sub = self.rna_adata[sub_mask].copy()
        sc.tl.rank_genes_groups(
            sub,
            groupby=self.perturb_key,
            groups=groups,
            reference=self.control_label,
            method='wilcoxon',
            use_raw=False,
            n_genes=sub.n_vars,
        )
        names = sub.uns['rank_genes_groups']['names']
        logfc = sub.uns['rank_genes_groups']['logfoldchanges']

        all_genes = sub.var_names.to_numpy()
        profile_map = {}
        for g in groups:
            gene_arr = np.asarray(names[g])
            lfc_arr = np.asarray(logfc[g], dtype=float)
            s = pd.Series(lfc_arr, index=gene_arr)
            profile_map[g] = s.reindex(all_genes).fillna(0.0).values

        self.de_profiles = pd.DataFrame(profile_map, index=all_genes).T
        return self.de_profiles

    def compute_similarity_matrix(self, method: Literal['spearman', 'pearson'] = 'spearman') -> pd.DataFrame:
        if self.de_profiles is None:
            raise ValueError("compute_de_profiles() first")
        print(f"Computing {method} similarity between DE profiles...")
        if method == 'spearman':
            mat = self.de_profiles.rank(axis=1, method='average').values
        else:
            mat = self.de_profiles.values - self.de_profiles.values.mean(axis=1, keepdims=True)
        t = torch.tensor(mat, dtype=torch.float32)
        sim = F.cosine_similarity(t.unsqueeze(1), t.unsqueeze(0), dim=-1).cpu().numpy()
        return pd.DataFrame(sim, index=self.de_profiles.index, columns=self.de_profiles.index)

    def build(self, corr_method: Literal['spearman', 'pearson'] = 'spearman', **_):
        # **_ swallows legacy `de_top_n` kwarg from older callers.
        self.compute_de_profiles()
        return self.compute_similarity_matrix(method=corr_method)
