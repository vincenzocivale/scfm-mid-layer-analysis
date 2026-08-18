"""Prepare Bessier et al. pseudotime h5ads for FM embedding extraction.

The source files under data/raw/pseudotime/bessier_et_al use human gene
symbols as var_names.  This script writes derived h5ads with:
  - var_names = human ENSG IDs, for Tahoe/GeneCompass
  - var['feature_name'] = human symbols, for scFoundation/scGPT/UCE
  - X = normalized/log1p counts, for scFoundation
  - raw.X = raw counts, restored by the extraction pipeline for raw-count models
  - a numeric pseudotime column in obs
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

from scfm_eval.preprocessing import humanize_adata


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO / "data" / "raw" / "pseudotime" / "bessier_et_al"
DEFAULT_CACHE_DIR = REPO / "data" / "checkpoints" / "gene_maps"


def _integerize_counts(X):
    """Return a sparse integer count matrix when the source is integer-like."""
    X = X.copy()
    if sparse.issparse(X):
        X = X.tocsr()
        if X.data.size and np.allclose(X.data, np.round(X.data)):
            X.data = np.round(X.data).astype(np.int64)
        return X
    if np.allclose(X, np.round(X)):
        return np.round(X).astype(np.int64)
    return X


def _base_counts_adata(src: ad.AnnData, counts_layer: str) -> ad.AnnData:
    if counts_layer not in src.layers:
        raise KeyError(f"Missing counts layer {counts_layer!r}; available: {list(src.layers.keys())}")

    obs = src.obs.copy()
    var = src.var.copy()
    var["feature_name"] = src.var_names.astype(str)
    var["gene_symbol"] = src.var_names.astype(str)

    return ad.AnnData(
        X=_integerize_counts(src.layers[counts_layer]),
        obs=obs,
        var=var,
    )


def _add_bro_time(obs):
    day = obs["Day"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
    obs["day"] = day
    obs["time_column"] = day
    obs["dataset"] = "bessier_bro_timecourse"
    return obs


def _add_dmgo_time(obs):
    def parse_time(value: str) -> int:
        match = re.search(r"_t([0-9]+)_", str(value))
        if not match:
            raise ValueError(f"Cannot parse DMGO timepoint from sample={value!r}")
        return int(match.group(1))

    timepoint = obs["sample"].map(parse_time).astype(int)
    obs["timepoint"] = timepoint
    obs["time_column"] = timepoint
    obs["dataset"] = "bessier_dmgo"
    return obs


def _normalize_for_x(counts_human: ad.AnnData) -> ad.AnnData:
    out = counts_human.copy()
    out.raw = counts_human.copy()
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    out.uns["counts_state"] = {
        "X": "log1p_normalized",
        "raw.X": "raw_counts",
    }
    return out


def prepare_one(
    input_path: Path,
    output_path: Path,
    counts_layer: str,
    time_kind: str,
    cache_dir: Path,
    force: bool = False,
) -> None:
    if output_path.exists() and not force:
        print(f"Skipping existing {output_path}", flush=True)
        return

    print(f"Reading {input_path}", flush=True)
    src = sc.read_h5ad(input_path)
    counts = _base_counts_adata(src, counts_layer)

    if time_kind == "bro":
        counts.obs = _add_bro_time(counts.obs)
    elif time_kind == "dmgo":
        counts.obs = _add_dmgo_time(counts.obs)
    else:
        raise ValueError(f"Unsupported time_kind={time_kind!r}")

    print(f"Humanizing gene namespace for {input_path.name}", flush=True)
    human = humanize_adata(counts, "human_symbol", cache_dir=cache_dir)
    out = _normalize_for_x(human)
    out.uns["source"] = {
        "dataset": "Bessier et al.",
        "input_path": str(input_path),
        "counts_layer": counts_layer,
        "reference": (
            "Bessler, N. et al. De novo H3.3K27M-altered diffuse midline glioma "
            "in human brainstem organoids to dissect GD2 CAR T cell function. "
            "Nat Cancer 7, 316-333 (2026)."
        ),
        "doi": "https://doi.org/10.1038/s43018-025-01084-0",
    }
    out.var_names_make_unique()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {output_path} ({out.n_obs} cells x {out.n_vars} genes)", flush=True)
    out.write_h5ad(output_path, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Bessier h5ads for FM embedding extraction.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing derived h5ads.")
    args = parser.parse_args()

    prepare_one(
        args.input_dir / "bro_timecourse_cleaned.h5ad",
        args.input_dir / "bro_timecourse_fm_ready.h5ad",
        counts_layer="counts",
        time_kind="bro",
        cache_dir=args.cache_dir,
        force=args.force,
    )
    prepare_one(
        args.input_dir / "dmgo_cleaned.h5ad",
        args.input_dir / "dmgo_fm_ready.h5ad",
        counts_layer="counts",
        time_kind="dmgo",
        cache_dir=args.cache_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
