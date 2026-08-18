"""
Main pipeline for perturbation analysis.
"""
import scanpy as sc
import pandas as pd
import os
import json
from typing import Optional
from pathlib import Path

from .reference import ReferenceBuilder
from .evaluator import PerturbationEvaluator
from .save_metrics import save_metrics # Re-using the existing save utility

def run_perturbation_pipeline(
    adata_path: str,
    output_dir: str,
    perturb_key: str,
    control_label: str,
    dose_key: Optional[str] = None,
    pathway_json_path: Optional[str] = None,
    de_top_n: int = 500,
    corr_method: str = 'spearman'
):
    """
    Runs the full perturbation analysis pipeline.

    Args:
        adata_path: Path to the AnnData file with embeddings.
        output_dir: Directory to save the results.
        perturb_key: Column in .obs with perturbation labels.
        control_label: Label for control cells.
        dose_key: Optional column in .obs with dose information.
        pathway_json_path: Optional path to a JSON file mapping perturbations to pathways.
        de_top_n: DEPRECATED — reference now uses full gene set from Wilcoxon DE.
        corr_method: Correlation method for reference ('spearman' or 'pearson').
    """
    print("--- Starting Perturbation Analysis Pipeline ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the input filename stem to use in output files
    input_stem = Path(adata_path).stem

    # 1. Load Data
    print(f"Loading AnnData from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    
    # Load optional pathway data
    pathway_dict = None
    if pathway_json_path:
        print(f"Loading pathway data from {pathway_json_path}...")
        try:
            with open(pathway_json_path, 'r') as f:
                pathway_dict = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load or parse pathway JSON. Skipping pathway analysis. Error: {e}")

    # 2. Build Biological Reference
    print("\n--- Building Biological Reference Matrix ---")
    ref_builder = ReferenceBuilder(adata, perturb_key=perturb_key, control_label=control_label)
    reference_sim_matrix = ref_builder.build(corr_method=corr_method)
    
    # Save the reference matrix for inspection
    ref_matrix_path = os.path.join(output_dir, f"{input_stem}_reference_de_similarity.csv")
    reference_sim_matrix.to_csv(ref_matrix_path)
    print(f"Saved DE reference similarity matrix to {ref_matrix_path}")

    # 3. Run Evaluation
    print("\n--- Evaluating Embedding Layers ---")
    evaluator = PerturbationEvaluator(adata, perturb_key=perturb_key, control_label=control_label)
    all_results = evaluator.evaluate(
        reference_sim_matrix=reference_sim_matrix,
        dose_key=dose_key,
        pathway_dict=pathway_dict
    )

    # 4. Save Results
    print("\n--- Saving All Metrics ---")
    if all_results:
        save_metrics(all_results, output_dir, filename_prefix=input_stem)
        print(f"All metrics saved successfully to {output_dir}")
    else:
        print("No results were generated from the evaluation.")
        
    print("\n--- Pipeline Finished ---")
