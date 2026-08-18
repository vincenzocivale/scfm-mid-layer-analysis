#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from scfm_eval.v2.results import ResultWriter
from scfm_eval.v2.specs import MetricRecord, RunSpec
from scfm_eval.v2.store import EmbeddingStore


def evaluate_linear_probe(X, y, folds: int, seed: int) -> dict:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs"),
    )
    pred = cross_val_predict(clf, X, y, cv=cv, method="predict")
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
    }


def main():
    p = argparse.ArgumentParser(
        description="Classification benchmark using canonical dataset annotations + sharded embeddings."
    )
    p.add_argument("--dataset-key", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--cell-type-column", required=True)
    p.add_argument("--results-root", type=Path, default=Path("artifacts/results"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    adata = sc.read_h5ad(args.dataset, backed="r")
    store = EmbeddingStore(args.embeddings)
    if not store.complete:
        raise ValueError(f"Embedding store is incomplete: {args.embeddings}")
    store.assert_alignment(adata.obs_names)

    labels = adata.obs[args.cell_type_column]
    valid = labels.notna().to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels.to_numpy()[valid])

    records = []
    for layer in store.layers:
        ids, X = store.load_layer(layer)
        metrics = evaluate_linear_probe(X[valid], y, args.folds, args.seed)
        for metric, value in metrics.items():
            records.append(MetricRecord(
                dataset=args.dataset_key,
                model=store.spec.model,
                model_size=store.spec.model_size,
                task="classification",
                representation="model_layer",
                layer=layer,
                n_layers_total=store.spec.n_layers_total,
                metric=metric,
                value=float(value),
                higher_is_better=True,
                n_obs=int(valid.sum()),
            ))

    run = RunSpec(
        dataset=args.dataset_key,
        model=store.spec.model,
        model_size=store.spec.model_size,
        task="classification",
        artifact_dir=str(args.embeddings),
        dataset_path=str(args.dataset),
    )
    path = ResultWriter(args.results_root, run).write(
        records,
        metadata={
            "cell_type_column": args.cell_type_column,
            "folds": args.folds,
            "seed": args.seed,
        },
    )
    print(path)


if __name__ == "__main__":
    main()
