"""Prepare GSE294098 (MOLM13 AML Perturb-seq, KAT6A-Menin-DOT1L screen)
for FM embedding extraction.

Unlike GSE294098's GEO description claims ("Filtered feature-barcode matrix"),
the supplied matrix is the *raw* (unfiltered) 10x barcode matrix — 3.2M
barcodes at ~40 UMIs/barcode on average — with "Gene Expression" and "CRISPR
Guide Capture" features interleaved in a single MatrixMarket file. No guide
calls are provided, so this script does both steps the authors would have run
upstream:

  1. Cell calling: keep barcodes with >= MIN_GENES detected genes. The paper
     reports 31,015 analyzed cells; min_genes=1000 yields 31,614 (closest
     simple threshold), so we use that as a defensible, documented choice
     rather than trying to exactly reproduce their (undocumented) QC.
  2. Guide assignment: for each called cell, take the CRISPR Guide Capture
     feature with the most UMIs. Keep the call only if it has >= MIN_GUIDE_UMI
     UMIs *and* a clear majority (> MIN_GUIDE_RATIO of total guide UMIs in
     that cell) — i.e. drop ambiguous / multi-guide cells, mirroring the
     "assigned_guide" convention used by the existing D1_* datasets.

The 16 target genes are encoded in the guide name prefix (e.g. 'KAT6A_4',
'sgMEN1_1' -> 'MEN1'); 'AAVS1_*' guides are the safe-harbor non-targeting
controls.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO / "data" / "raw" / "perturbation" / "GSE294098"
DEFAULT_OUTPUT = REPO / "data" / "raw" / "perturbation" / "GSE294098.h5ad"

MIN_GENES = 1000
MIN_GUIDE_UMI = 3
MIN_GUIDE_RATIO = 0.5
CONTROL_LABEL = "AAVS1"


def _target_gene(guide_id: str) -> str:
    """'KAT6A_4' -> 'KAT6A'; 'sgMEN1_1' -> 'MEN1'; 'AAVS1_2688' -> 'AAVS1'."""
    name = guide_id[2:] if guide_id.lower().startswith("sg") else guide_id
    return name.split("_")[0]


def prepare(raw_dir: Path, output_path: Path, force: bool = False) -> None:
    if output_path.exists() and not force:
        print(f"Skipping existing {output_path}", flush=True)
        return

    print("Reading 10x triplet ...", flush=True)
    features = pd.read_csv(raw_dir / "GSE294098_features.tsv.gz", header=None, sep="\t",
                           names=["gene_id", "gene_symbol", "feature_type"])
    barcodes = pd.read_csv(raw_dir / "GSE294098_barcodes.tsv.gz", header=None, sep="\t")[0].values
    mtx = sio.mmread(raw_dir / "GSE294098_matrix.mtx.gz").tocsr()  # features x barcodes
    if mtx.shape != (len(features), len(barcodes)):
        raise ValueError(f"mtx {mtx.shape} vs features={len(features)} barcodes={len(barcodes)}")

    gex_mask = (features["feature_type"] == "Gene Expression").values
    guide_mask = (features["feature_type"] == "CRISPR Guide Capture").values
    guide_names = features.loc[guide_mask, "gene_id"].values
    print(f"   {gex_mask.sum()} GEX features, {guide_mask.sum()} CRISPR guides, "
          f"{len(barcodes)} raw barcodes")

    print(f"Cell calling: keeping barcodes with >= {MIN_GENES} detected genes ...", flush=True)
    gex = mtx[gex_mask]
    n_genes = np.asarray((gex > 0).sum(axis=0)).ravel()
    called = n_genes >= MIN_GENES
    print(f"   called cells: {called.sum()} / {len(called)}")

    gex_called = gex[:, called].T.tocsr()              # cells x genes
    guide_called = mtx[guide_mask][:, called].toarray()  # guides x cells (small: 69 rows)
    barcodes_called = barcodes[called]
    n_genes_called = n_genes[called]
    total_counts = np.asarray(gex_called.sum(axis=1)).ravel()

    print(f"Guide assignment: argmax with >= {MIN_GUIDE_UMI} UMIs and "
          f"> {MIN_GUIDE_RATIO:.0%} majority ...", flush=True)
    top_idx = guide_called.argmax(axis=0)
    top_umi = guide_called.max(axis=0)
    total_umi = guide_called.sum(axis=0)
    ratio = np.divide(top_umi, total_umi, out=np.zeros(len(top_umi)), where=total_umi > 0)
    assigned = (top_umi >= MIN_GUIDE_UMI) & (ratio > MIN_GUIDE_RATIO)
    print(f"   assigned: {assigned.sum()} / {len(assigned)} ({100 * assigned.mean():.1f}%)")

    guide_id = np.array(["Unassigned"] * len(assigned), dtype=object)
    guide_id[assigned] = guide_names[top_idx[assigned]]
    target_gene = np.array([
        _target_gene(g) if a else None
        for g, a in zip(guide_id, assigned)
    ], dtype=object)

    obs = pd.DataFrame({
        "guide_id": guide_id,
        "target_gene": target_gene,
        "top_guide_umis": top_umi,
        "total_guide_umis": total_umi,
        "n_genes": n_genes_called,
        "total_counts": total_counts,
    }, index=pd.Index(barcodes_called, name="barcode"))

    var = pd.DataFrame({
        "gene_id": features.loc[gex_mask, "gene_id"].values,
        "gene_symbol": features.loc[gex_mask, "gene_symbol"].values,
        "feature_name": features.loc[gex_mask, "gene_symbol"].values,
    }).set_index("gene_id")
    var.index.name = None

    a = ad.AnnData(X=sp.csr_matrix(gex_called, dtype=np.int64), obs=obs, var=var)
    a.var_names_make_unique()

    # Keep only confidently assigned cells, mirroring the D1_* "assigned_guide" convention.
    a = a[a.obs["target_gene"].notna()].copy()
    a.obs["target_gene"] = a.obs["target_gene"].astype(str)
    a.obs["guide_id"] = a.obs["guide_id"].astype(str)
    print(f"   final: n_obs={a.n_obs}  n_vars={a.n_vars}")
    print(f"   target_gene value_counts:\n{a.obs['target_gene'].value_counts().to_string()}")

    a.uns["counts_state"] = {"X": "raw_counts"}
    a.uns["source"] = {
        "dataset": "GSE294098 (MOLM13 AML Perturb-seq)",
        "reference": (
            "Integrated Perturb-seq and Computational Modeling Identifies "
            "KAT6A-Menin-DOT1L Synergy and Associated Gene Program as "
            "Therapeutic Targets in Acute Myeloid Leukemia. GEO GSE294098."
        ),
        "notes": (
            f"Raw barcode matrix; cell-called with min_genes={MIN_GENES} "
            f"({int(called.sum())} of {len(called)} barcodes; "
            "paper reports 31,015 analyzed cells). Guide identity assigned "
            f"by per-cell argmax over CRISPR Guide Capture UMIs "
            f"(>= {MIN_GUIDE_UMI} UMIs, > {MIN_GUIDE_RATIO:.0%} majority); "
            "ambiguous/low-UMI cells dropped. perturb_key='target_gene', "
            f"control_label='{CONTROL_LABEL}' (AAVS1 safe-harbor guides)."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {output_path} ({a.n_obs} cells x {a.n_vars} genes)", flush=True)
    a.write_h5ad(output_path, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GSE294098 h5ad for FM embedding extraction.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Overwrite existing derived h5ad.")
    args = parser.parse_args()
    prepare(args.raw_dir, args.output, force=args.force)


if __name__ == "__main__":
    main()
