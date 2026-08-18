"""
Humanize a non-human raw-count h5ad so every implemented foundation model can embed it.

Maps genes into human space (var_names = human ENSG, var['feature_name'] = human symbol)
via Ensembl BioMart (cached). One humanized file serves both the symbol-matching models
(scFoundation/scGPT/UCE, read feature_name) and the ENSG-matching models
(Tahoe/GeneCompass, read var_names) — no model-side change needed.

Usage:
    python scripts/humanize_dataset.py --input data/raw/pseudotime/GSE320427.h5ad \
        --output data/raw/pseudotime/GSE320427_human.h5ad --source-namespace human_symbol

    python scripts/humanize_dataset.py --input data/raw/pseudotime/GSE322764.h5ad \
        --output data/raw/pseudotime/GSE322764_human.h5ad --source-namespace mouse_ensembl
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import scanpy as sc

from scfm_eval.preprocessing import humanize_adata


def _match_preview(adata):
    """Cheap, GPU-free coverage check: how many genes each model would match."""
    n_ensg = int(sum(str(v).startswith("ENSG") for v in adata.var_names))
    print(f"  Tahoe/GeneCompass (var_names ENSG-pattern): {n_ensg}/{adata.n_vars}")

    # scFoundation gene list (human symbols) — also a good proxy for scGPT/UCE symbol coverage.
    gene_index = Path(
        os.environ.get(
            "SCFOUNDATION_GENE_INDEX",
            "data/checkpoints/scfoundation/OS_scRNA_gene_index.19264.tsv",
        )
    )
    if gene_index.exists():
        import pandas as pd

        scf = set(pd.read_csv(gene_index, sep="\t")["gene_name"].astype(str).str.upper())
        feats = adata.var["feature_name"].astype(str).str.upper()
        n_scf = int(feats.isin(scf).sum())
        print(f"  scFoundation/scGPT/UCE (feature_name ∩ scF 19264 vocab): {n_scf}/{adata.n_vars}")
    else:
        print(f"  (scFoundation gene index not found at {gene_index}; skipping symbol preview)")


def main():
    parser = argparse.ArgumentParser(description="Humanize a non-human h5ad for all FMs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-namespace", required=True, choices=["human_symbol", "mouse_ensembl"]
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/checkpoints/gene_maps"))
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    adata = sc.read_h5ad(args.input)
    print(f"  in: {adata.shape}  X dtype={adata.X.dtype}  var_names[:3]={list(adata.var_names[:3])}")

    out = humanize_adata(adata, args.source_namespace, cache_dir=args.cache_dir)

    # ── Assertions (fail fast before any GPU time is spent) ──────────────────
    assert out.n_obs == adata.n_obs, "cell count changed during humanization"
    assert out.var_names.is_unique, "duplicate human ENSG var_names after collapse"
    assert all(str(v).startswith("ENSG") for v in out.var_names), "var_names are not all ENSG"
    assert "feature_name" in out.var.columns, "missing feature_name (human symbols)"
    n_empty_sym = int(out.var["feature_name"].astype(str).str.len().eq(0).sum())
    if n_empty_sym:
        # Harmless: these genes still match the ENSG models (Tahoe/GeneCompass) via
        # var_names; they only skip the symbol models (scFoundation/scGPT/UCE).
        print(f"  note: {n_empty_sym}/{out.n_vars} genes have no human symbol "
              f"(ENSG-only; will be skipped by symbol-matching models)")

    print("Match preview (genes each model would match):")
    _match_preview(out)

    print(f"obs columns preserved: {list(out.obs.columns)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output, compression="gzip")
    print(f"Wrote {args.output}  ({out.shape})")


if __name__ == "__main__":
    main()
