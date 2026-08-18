"""Prepare GSE303158 (hPSC pluripotency Perturb-seq, primed + naive) h5ads
for FM embedding extraction.

Source: two Seurat objects (post-Mixscape) supplied as double-gzipped RData,
exported via export_seurat_rdata.R into counts.mtx/genes.txt/barcodes.txt/meta.csv.
The exported metadata already carries the perturbation call from the authors'
Mixscape pipeline:
  - obs['target']               perturbed gene symbol, or 'CTRL' for controls
  - obs['mixscape_class.global']  CTRL / KO / NP (knockout vs non-perturbed)
  - obs['guide']                assigned sgRNA identity

We keep the authors' calls as-is (perturb_key='target', control_label='CTRL'),
restrict to genes detected in this assay, and humanize var_names from symbols
to human ENSG (var['feature_name']/['gene_symbol'] retain the symbol) so both
ENSG-keyed (Tahoe) and symbol-keyed (scFoundation) embedders can read it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse

from scfm_eval.preprocessing import humanize_adata

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO / "data" / "raw" / "perturbation" / "GSE303158"
DEFAULT_CACHE_DIR = REPO / "data" / "checkpoints" / "gene_maps"
RSCRIPT = REPO / "scripts" / "dataset_downloads" / "export_seurat_rdata.R"


def _export(rdata_gz: Path, out_dir: Path) -> None:
    if (out_dir / "counts.mtx").exists():
        print(f"   reusing existing export at {out_dir}", flush=True)
        return
    print(f"   running R export for {rdata_gz.name} ...", flush=True)
    subprocess.run(["Rscript", str(RSCRIPT), str(rdata_gz), str(out_dir)], check=True)


def _assemble(export_dir: Path, condition: str) -> ad.AnnData:
    genes = (export_dir / "genes.txt").read_text().splitlines()
    barcodes = (export_dir / "barcodes.txt").read_text().splitlines()
    meta = pd.read_csv(export_dir / "meta.csv", index_col=0)
    mtx = sio.mmread(export_dir / "counts.mtx").tocsr().T.tocsr()  # -> cells x genes
    if mtx.shape != (len(barcodes), len(genes)):
        raise ValueError(f"mtx {mtx.shape} vs barcodes={len(barcodes)} genes={len(genes)}")

    obs = meta.reindex(barcodes)
    # `cell_barcode` (no batch suffix) collides with — and disagrees with — the
    # batch-suffixed index h5ad wants as obs.index.name; drop the redundant column.
    obs = obs.drop(columns=["cell_barcode"], errors="ignore")
    obs.index = pd.Index(barcodes, name="cell_barcode")
    obs["condition"] = condition
    # Mixscape's `target` already gives us the perturb_key with 'CTRL' controls.
    obs["perturbed_gene"] = obs["target"].astype(str)

    # `genes.txt` holds Seurat rownames (gene symbols) — humanize_adata maps
    # var_names (source symbols) -> human ENSG and rebuilds var itself, so we
    # only need to dedupe the symbol index here.
    gene_index = pd.Index(genes, name="gene_symbol_raw")
    keep = ~gene_index.duplicated(keep="first")
    mtx = mtx[:, np.asarray(keep)]

    counts = mtx.tocsr()
    if counts.data.size and np.allclose(counts.data, np.round(counts.data)):
        counts.data = np.round(counts.data).astype(np.int64)

    a = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=gene_index[keep]))
    return a


def prepare_one(rdata_gz: Path, export_dir: Path, output_path: Path,
                condition: str, cache_dir: Path, force: bool = False) -> None:
    if output_path.exists() and not force:
        print(f"Skipping existing {output_path}", flush=True)
        return

    _export(rdata_gz, export_dir)
    print(f"Assembling {condition} ...", flush=True)
    counts = _assemble(export_dir, condition)
    print(f"   raw: n_obs={counts.n_obs}  n_vars={counts.n_vars}")
    print(f"   perturbed_gene value_counts (top 10):\n"
          f"{counts.obs['perturbed_gene'].value_counts().head(10).to_string()}")

    print("Humanizing gene namespace (symbol -> human ENSG) ...", flush=True)
    human = humanize_adata(counts, "human_symbol", cache_dir=cache_dir)
    human.uns["counts_state"] = {"X": "raw_counts"}
    human.uns["source"] = {
        "dataset": f"GSE303158 ({condition})",
        "input_path": str(rdata_gz),
        "reference": (
            "A single-cell CRISPR screen defines a gene regulatory network "
            "governing human pluripotency in primed and naive cells. "
            "GEO GSE303158."
        ),
        "notes": (
            "Perturbation calls (target/guide/mixscape_class*) are the "
            "authors' Mixscape output, kept as-is. perturb_key='perturbed_gene', "
            "control_label='CTRL'."
        ),
    }
    human.var_names_make_unique()

    # h5py can't write object columns with mixed types (e.g. bool+NaN from
    # the Mixscape meta.data); coerce object/bool obs columns to string.
    for c in human.obs.columns:
        col = human.obs[c]
        if col.dtype == object or col.dtype == bool:
            human.obs[c] = col.astype(str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {output_path} ({human.n_obs} cells x {human.n_vars} genes)", flush=True)
    human.write_h5ad(output_path, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GSE303158 h5ads for FM embedding extraction.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing derived h5ads.")
    args = parser.parse_args()

    prepare_one(
        args.raw_dir / "GSE303158_Seurat_after_mixscape.RData.gz",
        args.raw_dir / "export_primed",
        args.raw_dir.parent / "GSE303158_primed.h5ad",
        condition="primed",
        cache_dir=args.cache_dir,
        force=args.force,
    )
    prepare_one(
        args.raw_dir / "GSE303158_Seurat_after_mixscape_naive.RData.gz",
        args.raw_dir / "export_naive",
        args.raw_dir.parent / "GSE303158_naive.h5ad",
        condition="naive",
        cache_dir=args.cache_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
