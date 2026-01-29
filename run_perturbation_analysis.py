"""
Pipeline per analisi di perturbazione su embedding salvati.
Calcola e salva metriche per ogni layer, evitando ricalcoli inutili.
Risultati e grafici vengono salvati in data/perturbation_metrics/<nomefile>_<modello>/
"""
import os
import sys
import argparse
from pathlib import Path
import scanpy as sc
import pandas as pd
import pickle
from perturbation_analysis import emb_sim, perturb_eval, cluster_metrics, embedding_distance, de_recovery


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_output_dir(input_file, model_name):
    input_name = Path(input_file).stem
    return f"data/perturbation_metrics/{input_name}_{model_name}"


def save_metrics(metrics, out_path):
    with open(out_path, "wb") as f:
        pickle.dump(metrics, f)


def load_metrics(out_path):
    with open(out_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Pipeline di analisi perturbazione su embedding.")
    parser.add_argument('--input', type=str, required=True, help="File .h5ad con embedding")
    parser.add_argument('--model', type=str, required=True, help="Nome del modello (usato per output)")
    parser.add_argument('--force', action='store_true', help="Ricalcola metriche anche se già presenti")
    args = parser.parse_args()

    out_dir = get_output_dir(args.input, args.model)
    ensure_dir(out_dir)
    metrics_path = os.path.join(out_dir, "metrics.pkl")

    if os.path.exists(metrics_path) and not args.force:
        print(f"Carico metriche già calcolate da {metrics_path}")
        metrics = load_metrics(metrics_path)
    else:
        print(f"[START] Analisi perturbazione su {args.input}")
        adata = sc.read_h5ad(args.input)
        metrics = {}
        # Parametri
        CONTROL_LABEL = "NTC"
        MIN_CELLS = 20
        K_NEIGH = 5
        N_SPLITS = 5


        # Filtra perturbazioni con abbastanza cellule e limita a top N
        pert_counts = adata.obs['perturbed_gene_name'].value_counts()
        valid_perts = pert_counts[pert_counts >= MIN_CELLS].index.tolist()
        if CONTROL_LABEL not in valid_perts:
            print(f"[ERRORE] Nessuna cella di controllo trovata (label '{CONTROL_LABEL}' in perturbed_gene_name). Uscita.")
            return
        valid_perts = [p for p in valid_perts if p != CONTROL_LABEL]
        # Limita a top N perturbazioni più abbondanti
        TOP_N = 1000
        if len(valid_perts) > TOP_N:
            valid_perts = pert_counts[pert_counts.index.isin(valid_perts)].sort_values(ascending=False).index[:TOP_N].tolist()
            print(f"[LOG] Troppo numerose, uso solo le top {TOP_N} perturbazioni per abbondanza.")
        print(f"[LOG] Perturbazioni valide usate: {len(valid_perts)}")

        # Prepara moduli
        from perturbation_analysis.emb_sim import EmbeddingSimilarity
        from perturbation_analysis.perturb_sim import PerturbationSimilarity
        from perturbation_analysis.perturb_eval import SimilarityEvaluator
        from perturbation_analysis.semantic_metrics import global_perturbation_semantic_alignment, save_metric_per_layer

        # Calcola similarità embedding per tutti i layer
        emb_sim = EmbeddingSimilarity(adata, perturb_key='perturbed_gene_name', min_cells=MIN_CELLS)
        emb_sims = {}
        for key in sorted([k for k in adata.obsm.keys() if k.startswith('X_layer_')]):
            try:
                emb_sims[int(key.split('_')[-1])] = emb_sim.compute_single_layer_similarity(
                    key, perturbations=valid_perts, device='cpu', dtype='float16')
                print(f"[OK] Similarità embedding calcolata per {key}")
            except Exception as e:
                print(f"[WARN] Salto {key}: {e}")

        # Calcola similarità biologica (DE)
        pert_sim = PerturbationSimilarity(adata, perturb_key='perturbed_gene_name', control_label=CONTROL_LABEL, min_cells=MIN_CELLS)
        pert_sim.compute_de_profiles(n_top_genes=500)
        bio_sim = pert_sim.compute_similarity_matrix(device='cpu')
        print("[OK] Similarità biologica calcolata")

        # 1. Global Perturbation Semantic Alignment
        evaluator = SimilarityEvaluator(bio_sim, emb_sims)
        df_sem = evaluator.compute_layer_correlations(method='spearman')
        df_sem.to_csv(os.path.join(out_dir, "global_semantic_alignment.csv"), index=False)
        metrics['global_semantic_alignment'] = df_sem.set_index('layer')['correlation'].to_dict()
        print("[OK] Global Perturbation Semantic Alignment salvata")

        # 2. Held-Out Perturbation Generalization
        df_ho = evaluator.evaluate_held_out_perturbations(n_splits=N_SPLITS)
        df_ho.to_csv(os.path.join(out_dir, "heldout_generalization.csv"), index=False)
        metrics['heldout_generalization'] = df_ho.set_index('layer').to_dict(orient='index')
        print("[OK] Held-Out Perturbation Generalization salvata")

        # 3. Local Neighborhood Recovery (Top-K Overlap)
        df_topk = evaluator.compute_top_k_accuracy(k=K_NEIGH)
        df_topk.to_csv(os.path.join(out_dir, f"local_neigh_top{K_NEIGH}.csv"), index=False)
        metrics['local_neigh_topk'] = df_topk.set_index('layer')[f'top_{K_NEIGH}_accuracy'].to_dict()
        print("[OK] Local Neighborhood Recovery salvata")

        save_metrics(metrics, metrics_path)
        print(f"[DONE] Tutte le metriche salvate in {metrics_path}")

    print("[END] Analisi completata. Solo metriche salvate.")

if __name__ == "__main__":
    main()
