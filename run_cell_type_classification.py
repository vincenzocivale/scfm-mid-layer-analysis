"""
Script di lancio per la pipeline di analisi di classificazione dei tipi di cellule.
Permette di specificare input, colonna dei tipi di cellule, prefisso embedding e output.
Salva i risultati in una cartella di output specificata.
"""
import argparse
from pathlib import Path
import os
from cell_type_classification_benchmark.pipeline import run_classification_pipeline

def main():
    parser = argparse.ArgumentParser(description="Cell type classification analysis pipeline")
    parser.add_argument('--input', type=str, required=True, help="File .h5ad con embedding")
    parser.add_argument('--cell_type_column', type=str, default='cell_type', help="Colonna con etichette dei tipi di cellule in .obs")
    parser.add_argument('--embedding_prefix', type=str, default='X_layer', help="Prefisso embedding in obsm")
    parser.add_argument('--output_dir', type=str, default='data/classification_results', help="Cartella di output per i risultati")
    args = parser.parse_args()

    # Crea la cartella di output se non esiste
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Definisci il percorso del file di output
    input_name = Path(args.input).stem
    output_path = os.path.join(args.output_dir, f"{input_name}_classification_results.csv")

    print(f"Eseguo pipeline di classificazione su {args.input} (colonna cell type: {args.cell_type_column})...")
    
    # Esegui la pipeline
    results = run_classification_pipeline(
        h5ad_path=args.input,
        cell_type_column=args.cell_type_column,
        embedding_prefix=args.embedding_prefix,
        output_path=output_path
    )
    
    if not results.empty:
        print(f"Pipeline completata. Risultati salvati in {output_path}")
    else:
        print("Pipeline completata senza generare risultati.")

if __name__ == "__main__":
    main()
