"""Per-layer perturbation evaluator.

CAVEAT (data leakage by construction): the biological reference is the DE matrix
computed on the *same* counts the FM saw at inference. The semantic-similarity
score therefore measures how well the embedding *preserves* the DE structure
derivable from those same counts, not predictive biology. For a stricter
zero-shot test, supply an external reference (e.g. CMap, MSigDB pathway sim).
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

_SEED = 42


def _cosine_sim_matrix(M: np.ndarray) -> np.ndarray:
    t = torch.tensor(M, dtype=torch.float32)
    return F.cosine_similarity(t.unsqueeze(1), t.unsqueeze(0), dim=-1).cpu().numpy()


def _upper_triu(M: np.ndarray) -> np.ndarray:
    return M[np.triu_indices_from(M, k=1)]


class PerturbationEvaluator:
    def __init__(self, adata: sc.AnnData, perturb_key: str, control_label: str = 'control',
                 min_cells_per_perturb: int = 10, seed: int = _SEED):
        self.adata = adata
        self.perturb_key = perturb_key
        self.control_label = control_label
        self.seed = seed

        self.layer_keys = sorted(
            [k for k in adata.obsm.keys() if k.startswith('X_layer_') or k.startswith('X_scgpt_')],
            key=lambda x: int(x.split('_')[-1]),
        )
        if not self.layer_keys:
            raise ValueError("No layer embeddings (X_layer_* or X_scgpt_*) in adata.obsm")

        counts = adata.obs[perturb_key].value_counts()
        self.perturbations = sorted([
            p for p in counts.index
            if pd.notna(p) and p != control_label and counts[p] >= min_cells_per_perturb
        ])

    def _compute_centroids(self, layer_key: str) -> pd.DataFrame:
        emb = self.adata.obsm[layer_key]
        labels = self.adata.obs[self.perturb_key].values
        rows = [emb[labels == p].mean(axis=0) for p in self.perturbations]
        return pd.DataFrame(np.vstack(rows), index=self.perturbations)

    def evaluate_semantic_similarity(self, reference_sim_matrix: pd.DataFrame,
                                     n_permutations: int = 100) -> pd.DataFrame:
        """Spearman(centroid-cosine-sim, reference-sim) plus a label-shuffle null distribution.

        Returns one row per layer with: correlation, p_value (analytic),
        null_mean, null_std, z_score (vs shuffle null), n_common_perturbations.
        """
        print("Evaluating semantic similarity...")
        common = sorted(set(self.perturbations) & set(reference_sim_matrix.index))
        if len(common) < 3:
            print(f"WARNING: only {len(common)} perturbations in common with reference; results unreliable.")
        ref_aligned = reference_sim_matrix.loc[common, common]
        ref_vec = _upper_triu(ref_aligned.values)

        rng = np.random.default_rng(self.seed)
        results = []
        for layer_key in tqdm(self.layer_keys, desc="Semantic Similarity"):
            centroids = self._compute_centroids(layer_key).loc[common]
            emb_sim = _cosine_sim_matrix(centroids.values)
            emb_vec = _upper_triu(emb_sim)
            corr, p_val = spearmanr(ref_vec, emb_vec)

            null_corrs = np.empty(n_permutations)
            n = len(common)
            for i in range(n_permutations):
                perm = rng.permutation(n)
                shuffled = emb_sim[np.ix_(perm, perm)]
                null_corrs[i], _ = spearmanr(ref_vec, _upper_triu(shuffled))
            null_mean = float(np.mean(null_corrs))
            null_std = float(np.std(null_corrs))
            z = (corr - null_mean) / null_std if null_std > 0 else np.nan

            results.append({
                'layer': int(layer_key.split('_')[-1]),
                'correlation': float(corr),
                'p_value': float(p_val),
                'null_mean': null_mean,
                'null_std': null_std,
                'z_score': float(z),
                'n_common_perturbations': len(common),
            })
        return pd.DataFrame(results).sort_values('layer').reset_index(drop=True)

    def evaluate_dose_response(self, dose_key: str) -> Optional[pd.DataFrame]:
        """Cosine-distance from control centroid vs dose, Spearman per perturbation, mean per layer."""
        if dose_key not in self.adata.obs:
            print(f"Warning: dose key '{dose_key}' not found. Skipping.")
            return None
        print("Evaluating dose response (cosine distance)...")
        control_mask = (self.adata.obs[self.perturb_key] == self.control_label).values

        results = []
        for layer_key in tqdm(self.layer_keys, desc="Dose Response"):
            emb = self.adata.obsm[layer_key]
            ctrl = emb[control_mask].mean(axis=0)
            ctrl_t = torch.tensor(ctrl, dtype=torch.float32).unsqueeze(0)
            layer_corrs = []
            for p in self.perturbations:
                mask = (self.adata.obs[self.perturb_key] == p).values
                if mask.sum() == 0:
                    continue
                doses = self.adata.obs.loc[mask, dose_key]
                if len(doses.unique()) < 2 or not pd.api.types.is_numeric_dtype(doses):
                    continue
                pe = torch.tensor(emb[mask], dtype=torch.float32)
                cos = F.cosine_similarity(pe, ctrl_t.expand_as(pe), dim=-1).cpu().numpy()
                dist = 1.0 - cos
                corr, _ = spearmanr(doses.values, dist)
                if not np.isnan(corr):
                    layer_corrs.append(corr)
            if layer_corrs:
                results.append({
                    'layer': int(layer_key.split('_')[-1]),
                    'mean_dose_correlation': float(np.mean(layer_corrs)),
                    'std_dose_correlation': float(np.std(layer_corrs)),
                    'n_perturbations_with_dose': len(layer_corrs),
                })
        if not results:
            print("Dose-response: no usable perturbations.")
            return None
        return pd.DataFrame(results).sort_values('layer').reset_index(drop=True)

    def evaluate_pathway_clustering(self, pathway_dict: Dict[str, List[str]]) -> Optional[pd.DataFrame]:
        if not pathway_dict:
            return None
        print("Evaluating pathway clustering...")
        results = []
        for layer_key in tqdm(self.layer_keys, desc="Pathway Clustering"):
            centroids = self._compute_centroids(layer_key)
            sim = pd.DataFrame(_cosine_sim_matrix(centroids.values),
                               index=centroids.index, columns=centroids.index)
            within, between = [], []
            perturbs = list(sim.index)
            for i in range(len(perturbs)):
                for j in range(i + 1, len(perturbs)):
                    p1, p2 = perturbs[i], perturbs[j]
                    a = set(pathway_dict.get(p1, []))
                    b = set(pathway_dict.get(p2, []))
                    if not a or not b:
                        continue
                    s = sim.loc[p1, p2]
                    (within if a & b else between).append(s)
            if within and between:
                results.append({
                    'layer': int(layer_key.split('_')[-1]),
                    'pathway_gap': float(np.mean(within) - np.mean(between)),
                    'n_within': len(within),
                    'n_between': len(between),
                })
        if not results:
            return None
        return pd.DataFrame(results).sort_values('layer').reset_index(drop=True)

    def evaluate(self, reference_sim_matrix: pd.DataFrame,
                 dose_key: Optional[str] = None,
                 pathway_dict: Optional[Dict[str, List[str]]] = None,
                 n_permutations: int = 100) -> Dict[str, pd.DataFrame]:
        out = {'semantic_similarity': self.evaluate_semantic_similarity(
            reference_sim_matrix, n_permutations=n_permutations)}
        if dose_key:
            dr = self.evaluate_dose_response(dose_key)
            if dr is not None:
                out['dose_response'] = dr
        if pathway_dict:
            pc = self.evaluate_pathway_clustering(pathway_dict)
            if pc is not None:
                out['pathway_clustering'] = pc
        return out
