"""
Script di lancio per la pipeline di analisi pseudo-time.
Permette di specificare input, colonna temporale, prefisso embedding e output.
Salva i risultati in data/pseudotime_results/<nomefile>_results.csv
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pseudo_time_benchmark.pipeline import run_evaluation_pipeline

def main():
    parser = argparse.ArgumentParser(description="Pseudo-time analysis pipeline")
    parser.add_argument('--input', type=str, required=True, help="File .h5ad con embedding")
    parser.add_argument('--time_column', type=str, default='week', help="Colonna temporale in obs")
    parser.add_argument('--embedding_prefix', type=str, default='X_layer', help="Prefisso embedding in obsm")
    parser.add_argument('--output_dir', type=str, default='data/pseudotime_results', help="Cartella di output")
    args = parser.parse_args()

    input_name = Path(args.input).stem
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{input_name}_results.csv")

    print(f"Eseguo pipeline su {args.input} (colonna tempo: {args.time_column})...")
    results = run_evaluation_pipeline(
        args.input,
        time_column=args.time_column,
        embedding_prefix=args.embedding_prefix,
        output_path=output_path
    )
    print(f"Risultati salvati in {output_path}")
    print("Plotting disabilitato. Solo metriche salvate.")

if __name__ == "__main__":
    main()
