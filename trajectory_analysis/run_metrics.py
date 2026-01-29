"""
Script per eseguire il calcolo delle metriche trajectory layer-wise e salvare i risultati in CSV.
Non genera plot, solo salvataggio risultati.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import argparse
import scanpy as sc
import pandas as pd
import os
from trajectory_analysis.metrics import evaluate_regression, evaluate_classification, evaluate_pseudotime, evaluate_smoothness, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Valutazione traiettoria su embedding.")
    parser.add_argument('--input', type=str, required=True, help="File .h5ad con embedding")
    parser.add_argument('--time_col', type=str, required=True, help="Colonna temporale in obs")
    parser.add_argument('--output', type=str, default=None, help="File output csv")
    args = parser.parse_args()

    adata = sc.read_h5ad(args.input)
    layer_keys = [k for k in adata.obsm.keys() if k.startswith("X_layer_")]
    results = []
    for key in layer_keys:
        emb = adata.obsm[key]
        tp = adata.obs[args.time_col].values
        reg = evaluate_regression(emb, tp)
        clf = evaluate_classification(emb, tp)
        pt_res = evaluate_pseudotime(adata, key, args.time_col)
        smooth = evaluate_smoothness(adata, args.time_col)
        res = {"Layer": key, **reg, **clf, **pt_res, **smooth}
        results.append(res)
    df = pd.DataFrame(results)
    if args.output:
        output_path = args.output
    else:
        input_base = os.path.splitext(os.path.basename(args.input))[0]
        out_dir = os.path.join('data', 'trajectory_metrics')
        ensure_dir(out_dir)
        output_path = os.path.join(out_dir, f"{input_base}_trajectory_metrics.csv")
    df.to_csv(output_path, index=False)
    print(f"Risultati salvati in: {output_path}")
    print(df)

if __name__ == "__main__":
    main()
