"""
Core evaluator for perturbation analysis.
"""
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm
from typing import Dict, Optional, List

class PerturbationEvaluator:
    """
    Evaluates the quality of embeddings based on how they represent
    biological perturbations.
    """
    def __init__(self, adata: sc.AnnData, perturb_key: str, control_label: str = 'control'):
        """
        Args:
            adata: AnnData object containing embeddings.
            perturb_key: Column in .obs with perturbation labels.
            control_label: Label for control cells.
        """
        self.adata = adata
        self.perturb_key = perturb_key
        self.control_label = control_label
        
        # Find layer embeddings
        self.layer_keys = sorted([
            k for k in self.adata.obsm.keys() 
            if k.startswith('X_layer_') or k.startswith('X_scgpt_')
        ], key=lambda x: int(x.split('_')[-1]))

        if not self.layer_keys:
            raise ValueError("No layer embeddings found in adata.obsm (e.g., 'X_layer_1').")

        # Get valid perturbations
        perturb_counts = self.adata.obs[self.perturb_key].value_counts()
        self.perturbations = sorted([
            p for p in perturb_counts.index 
            if pd.notna(p) and p != self.control_label and perturb_counts[p] >= 5
        ])

    def _compute_centroids(self, layer_key: str) -> pd.DataFrame:
        """Computes centroids for all perturbations in a given layer."""
        embeddings = self.adata.obsm[layer_key]
        labels = self.adata.obs[self.perturb_key].values
        
        centroids = []
        for perturb in self.perturbations:
            mask = (labels == perturb)
            centroids.append(embeddings[mask].mean(axis=0))
        
        return pd.DataFrame(np.array(centroids), index=self.perturbations)

    def _compute_centroid_similarity(self, centroids: pd.DataFrame) -> pd.DataFrame:
        """Computes cosine similarity between centroids."""
        centroids_tensor = torch.tensor(centroids.values, dtype=torch.float32)
        sim_matrix = F.cosine_similarity(centroids_tensor.unsqueeze(1), centroids_tensor.unsqueeze(0), dim=-1)
        return pd.DataFrame(sim_matrix.numpy(), index=centroids.index, columns=centroids.index)

    def evaluate_semantic_similarity(self, reference_sim_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Correlates embedding-based similarity with a biological reference similarity.
        This is the main metric for semantic alignment.
        """
        print("Evaluating semantic similarity...")
        results = []
        
        # Align reference matrix to the perturbations found in the adata
        common_perturbs = sorted(list(set(self.perturbations) & set(reference_sim_matrix.index)))
        ref_sim_aligned = reference_sim_matrix.loc[common_perturbs, common_perturbs]
        ref_values = ref_sim_aligned.values[np.triu_indices_from(ref_sim_aligned.values, k=1)]

        for layer_key in tqdm(self.layer_keys, desc="Semantic Similarity"):
            centroids = self._compute_centroids(layer_key)
            centroids_aligned = centroids.loc[common_perturbs]
            
            emb_sim = self._compute_centroid_similarity(centroids_aligned)
            emb_values = emb_sim.values[np.triu_indices_from(emb_sim.values, k=1)]
            
            corr, p_val = spearmanr(ref_values, emb_values)
            results.append({'layer': int(layer_key.split('_')[-1]), 'correlation': corr, 'p_value': p_val})
        
        return pd.DataFrame(results).sort_values('layer').reset_index(drop=True)

    def evaluate_dose_response(self, dose_key: str) -> Optional[pd.DataFrame]:
        """
        Evaluates if embedding distance from control scales with perturbation dose.
        """
        if dose_key not in self.adata.obs:
            print(f"Warning: Dose key '{dose_key}' not found. Skipping dose evaluation.")
            return None
        
        print("Evaluating dose response...")
        results = []
        control_mask = self.adata.obs[self.perturb_key] == self.control_label

        for layer_key in tqdm(self.layer_keys, desc="Dose Response"):
            layer_embeddings = self.adata.obsm[layer_key]
            control_centroid = layer_embeddings[control_mask].mean(axis=0)
            
            layer_correlations = []
            for perturb in self.perturbations:
                perturb_mask = self.adata.obs[self.perturb_key] == perturb
                if perturb_mask.sum() == 0: continue
                
                doses = self.adata.obs.loc[perturb_mask, dose_key]
                # Skip if only one dose level or doses are not numeric
                if len(doses.unique()) < 2 or not pd.api.types.is_numeric_dtype(doses):
                    continue
                    
                perturb_embeddings = layer_embeddings[perturb_mask]
                distances = np.linalg.norm(perturb_embeddings - control_centroid, axis=1)
                
                # Correlation between dose and distance from control
                corr, _ = spearmanr(doses, distances)
                if not np.isnan(corr):
                    layer_correlations.append(corr)
            
            if layer_correlations:
                results.append({
                    'layer': int(layer_key.split('_')[-1]),
                    'mean_dose_correlation': np.mean(layer_correlations),
                    'std_dose_correlation': np.std(layer_correlations)
                })
        
        if not results:
            print("Dose-response evaluation could not be completed (e.g., no numeric doses found).")
            return None
            
        return pd.DataFrame(results).sort_values('layer').reset_index(drop=True)

    def evaluate_pathway_clustering(self, pathway_dict: Dict[str, List[str]]) -> Optional[pd.DataFrame]:
        """
        Evaluates if perturbations of the same pathway are more similar in embedding space.
        Uses the "pathway gap" metric.
        """
        if not pathway_dict:
            print("Warning: No pathway dictionary provided. Skipping pathway evaluation.")
            return None
            
        print("Evaluating pathway clustering...")
        results = []
        
        for layer_key in tqdm(self.layer_keys, desc="Pathway Clustering"):
            centroids = self._compute_centroids(layer_key)
            emb_sim = self._compute_centroid_similarity(centroids)
            
            within_pathway_sims = []
            between_pathway_sims = []
            
            perturbs = list(emb_sim.index)
            for i in range(len(perturbs)):
                for j in range(i + 1, len(perturbs)):
                    p1, p2 = perturbs[i], perturbs[j]
                    p1_pathways = set(pathway_dict.get(p1, []))
                    p2_pathways = set(pathway_dict.get(p2, []))
                    
                    if not p1_pathways or not p2_pathways: continue
                    
                    similarity = emb_sim.loc[p1, p2]
                    if p1_pathways & p2_pathways: # Shared pathway
                        within_pathway_sims.append(similarity)
                    else:
                        between_pathway_sims.append(similarity)
            
            if within_pathway_sims and between_pathway_sims:
                gap = np.mean(within_pathway_sims) - np.mean(between_pathway_sims)
                results.append({'layer': int(layer_key.split('_')[-1]), 'pathway_gap': gap})

        if not results:
            print("Pathway evaluation could not be completed (e.g., no overlapping pathways found).")
            return None

        return pd.DataFrame(results).sort_values('layer').reset_index(drop=True)

    def evaluate(self, reference_sim_matrix: pd.DataFrame, dose_key: Optional[str] = None, pathway_dict: Optional[Dict[str, List[str]]] = None) -> Dict[str, pd.DataFrame]:
        """
        Runs all evaluations and returns a dictionary of results.
        """
        all_results = {}
        
        # 1. Semantic Similarity (core metric)
        all_results['semantic_similarity'] = self.evaluate_semantic_similarity(reference_sim_matrix)
        
        # 2. Dose Response (optional)
        if dose_key:
            dose_results = self.evaluate_dose_response(dose_key)
            if dose_results is not None:
                all_results['dose_response'] = dose_results

        # 3. Pathway Clustering (optional)
        if pathway_dict:
            pathway_results = self.evaluate_pathway_clustering(pathway_dict)
            if pathway_results is not None:
                all_results['pathway_clustering'] = pathway_results
                
        return all_results
