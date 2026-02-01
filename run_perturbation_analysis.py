"""
Command-line launcher for the refactored perturbation analysis pipeline.
"""
import argparse
import sys
import os

# Add the project root to the Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from perturbation_analysis.pipeline import run_perturbation_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the perturbation analysis pipeline on an AnnData file with embeddings.")
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True, help="Path to the input AnnData file (.h5ad) containing embeddings.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the analysis results.")
    parser.add_argument('--perturb_key', type=str, required=True, help="Column name in .obs that contains perturbation labels (e.g., 'gene_name').")
    parser.add_argument('--control_label', type=str, required=True, help="The label used for control/unperturbed cells in the perturb_key column.")
    
    # Optional arguments
    parser.add_argument('--dose_key', type=str, default=None, help="Optional: Column name in .obs for dose information.")
    parser.add_argument('--pathway_json', type=str, default=None, help="Optional: Path to a JSON file mapping perturbations to pathways.")
    
    # Analysis parameters
    parser.add_argument('--de_top_n', type=int, default=500, help="Number of top DE genes to use for building the biological reference matrix.")
    parser.add_argument('--corr_method', type=str, default='spearman', choices=['spearman', 'pearson'], help="Correlation method for the reference similarity matrix.")
    
    args = parser.parse_args()

    # Run the pipeline with the parsed arguments
    run_perturbation_pipeline(
        adata_path=args.input,
        output_dir=args.output_dir,
        perturb_key=args.perturb_key,
        control_label=args.control_label,
        dose_key=args.dose_key,
        pathway_json_path=args.pathway_json,
        de_top_n=args.de_top_n,
        corr_method=args.corr_method
    )

if __name__ == "__main__":
    main()