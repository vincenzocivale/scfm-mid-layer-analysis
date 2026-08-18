"""Gene-ID / cross-species mapping for cross-species datasets.

The foundation-model embedders only match a single gene namespace each:

  * symbol models (scFoundation, scGPT, UCE) read ``var['feature_name']`` — human symbols
  * ENSG  models (Tahoe, GeneCompass) read ``var_names`` — human Ensembl gene IDs

This module produces a single "humanized" AnnData per non-human dataset with
``var_names = human ENSG`` **and** ``var['feature_name'] = human symbol`` so that all
five models can embed it without any model-side change.

Mappings are pulled once from Ensembl BioMart over HTTP and cached as TSV under
``data/checkpoints/gene_maps/``. Only ``requests`` is required (already a dependency);
no pybiomart/mygene.

Cross-species policy: mouse -> human uses **one2one, high-confidence** orthologs only.
This is lossy (roughly half of mouse genes have no 1:1 human ortholog) but it is the
only defensible way to feed mouse data to human-only foundation models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

# BioMart REST endpoint + a mirror fallback.
_BIOMART_HOSTS = [
    "https://www.ensembl.org/biomart/martservice",
    "https://useast.ensembl.org/biomart/martservice",
]

# Standard human chromosomes — excludes scaffolds/patches that introduce duplicate symbols.
_STD_CHROMS = ",".join([str(i) for i in range(1, 23)] + ["X", "Y", "MT"])

DEFAULT_CACHE_DIR = Path("data/checkpoints/gene_maps")


# ---------------------------------------------------------------------------
# BioMart access
# ---------------------------------------------------------------------------
def _biomart_query(dataset: str, attributes: list, filters_xml: str = "") -> pd.DataFrame:
    """Run a BioMart TSV query and return a DataFrame with one column per attribute."""
    import requests

    attrs = "".join(f'<Attribute name="{a}" />' for a in attributes)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<!DOCTYPE Query>"
        '<Query virtualSchemaName="default" formatter="TSV" header="0" '
        'uniqueRows="1" count="" datasetConfigVersion="0.6">'
        f'<Dataset name="{dataset}" interface="default">{filters_xml}{attrs}</Dataset>'
        "</Query>"
    )
    last_err = None
    for host in _BIOMART_HOSTS:
        try:
            resp = requests.get(host, params={"query": xml}, timeout=180)
            resp.raise_for_status()
            text = resp.text
            if text.startswith("Query ERROR"):
                raise RuntimeError(f"BioMart returned: {text[:200]}")
            from io import StringIO

            df = pd.read_csv(
                StringIO(text), sep="\t", header=None, names=attributes, dtype=str
            ).fillna("")
            if len(df) == 0:
                raise RuntimeError("BioMart returned 0 rows")
            return df
        except Exception as exc:  # noqa: BLE001 — try the mirror, then re-raise
            last_err = exc
            print(f"[gene_mapping] BioMart host {host} failed: {exc}", file=sys.stderr)
    raise RuntimeError(f"All BioMart hosts failed; last error: {last_err}")


# ---------------------------------------------------------------------------
# Mapping tables (cached)
# ---------------------------------------------------------------------------
def load_human_symbol_to_ensg(
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return ``(symbol_upper -> ENSG, ENSG -> symbol)`` for human genes.

    Restricted to standard chromosomes and deduped to one ENSG per symbol.
    """
    cache_dir = Path(cache_dir)
    cache = cache_dir / "human_symbol_ensg.tsv"
    if cache.exists():
        df = pd.read_csv(cache, sep="\t", dtype=str).fillna("")
    else:
        print("[gene_mapping] Fetching human symbol<->ENSG from BioMart ...")
        df = _biomart_query(
            "hsapiens_gene_ensembl",
            ["ensembl_gene_id", "external_gene_name"],
            filters_xml=f'<Filter name="chromosome_name" value="{_STD_CHROMS}"/>',
        )
        df.columns = ["ensembl_gene_id", "symbol"]
        df = df[(df["ensembl_gene_id"] != "") & (df["symbol"] != "")]
        df = df.drop_duplicates(subset="symbol", keep="first").reset_index(drop=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, sep="\t", index=False)
        print(f"[gene_mapping] Cached {len(df)} human symbol->ENSG rows to {cache}")

    sym2ensg = {s.upper(): e for s, e in zip(df["symbol"], df["ensembl_gene_id"])}
    ensg2sym = {e: s for s, e in zip(df["symbol"], df["ensembl_gene_id"])}
    return sym2ensg, ensg2sym


def load_mouse_to_human_orthologs(
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Dict[str, Tuple[str, str]]:
    """Return ``ENSMUSG -> (human ENSG, human symbol)`` for one2one, high-confidence orthologs."""
    cache_dir = Path(cache_dir)
    cache = cache_dir / "mouse_human_orthologs.tsv"
    if cache.exists():
        df = pd.read_csv(cache, sep="\t", dtype=str).fillna("")
    else:
        print("[gene_mapping] Fetching mouse->human orthologs from BioMart ...")
        raw = _biomart_query(
            "mmusculus_gene_ensembl",
            [
                "ensembl_gene_id",
                "hsapiens_homolog_ensembl_gene",
                "hsapiens_homolog_associated_gene_name",
                "hsapiens_homolog_orthology_type",
                "hsapiens_homolog_orthology_confidence",
            ],
        )
        raw.columns = ["ensmusg", "hsa_ensg", "hsa_symbol", "otype", "confidence"]
        df = raw[
            (raw["otype"] == "ortholog_one2one")
            & (raw["confidence"] == "1")
            & (raw["hsa_ensg"] != "")
        ].copy()
        df = df[["ensmusg", "hsa_ensg", "hsa_symbol"]].drop_duplicates(
            subset="ensmusg", keep="first"
        ).reset_index(drop=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, sep="\t", index=False)
        print(f"[gene_mapping] Cached {len(df)} mouse->human one2one orthologs to {cache}")

    return {
        m: (e, s)
        for m, e, s in zip(df["ensmusg"], df["hsa_ensg"], df["hsa_symbol"])
    }


# ---------------------------------------------------------------------------
# Humanization
# ---------------------------------------------------------------------------
def _strip_version(ids) -> list:
    """Drop Ensembl version suffixes (ENSG0000....5 -> ENSG0000....)."""
    return [str(g).split(".")[0] for g in ids]


def humanize_adata(adata, source_namespace: str, cache_dir: Path = DEFAULT_CACHE_DIR):
    """Map ``adata`` genes into human space (var_names=ENSG, feature_name=symbol).

    Parameters
    ----------
    adata : AnnData
        Raw-count AnnData. ``adata.var_names`` hold the source identifiers.
    source_namespace : {"human_symbol", "mouse_ensembl"}
        ``human_symbol``  — var_names are human-ortholog symbols (e.g. sheep GSE320427).
        ``mouse_ensembl`` — var_names are ENSMUSG IDs (e.g. mouse GSE322764).

    Returns
    -------
    AnnData
        New object: ``var_names`` = human ENSG, ``var['feature_name']`` =
        ``var['gene_symbol']`` = human symbol, X = raw counts with genes that collapse to
        the same human ENSG **summed**, ``obs`` preserved verbatim. ``uns['humanization']``
        records the provenance.
    """
    import anndata as ad

    src_ids = _strip_version(adata.var_names)
    n_in = len(src_ids)

    if source_namespace == "human_symbol":
        sym2ensg, _ = load_human_symbol_to_ensg(cache_dir)
        # target_ensg, target_symbol per source gene (None if unmapped)
        targets = []
        for g in src_ids:
            ensg = sym2ensg.get(str(g).upper())
            targets.append((ensg, str(g).upper()) if ensg else None)
    elif source_namespace == "mouse_ensembl":
        orth = load_mouse_to_human_orthologs(cache_dir)
        # Some one2one orthologs carry a human ENSG but an empty homolog symbol in
        # BioMart; backfill the symbol from the human ENSG->symbol map so the
        # symbol-matching models (scFoundation/scGPT/UCE) keep those genes too.
        _, ensg2sym = load_human_symbol_to_ensg(cache_dir)
        targets = []
        for g in src_ids:
            t = orth.get(g)
            if t is None:
                targets.append(None)
                continue
            ensg, sym = t
            if not sym:
                sym = ensg2sym.get(ensg, "")
            targets.append((ensg, sym))
    else:
        raise ValueError(
            f"source_namespace must be 'human_symbol' or 'mouse_ensembl', got {source_namespace!r}"
        )

    kept_cols = [i for i, t in enumerate(targets) if t is not None]
    if not kept_cols:
        raise RuntimeError(
            f"No genes mapped to human for source_namespace={source_namespace!r}. "
            f"Example source ids: {src_ids[:5]}"
        )

    # Order unique target ENSG by first appearance; build old-col -> new-col incidence.
    ensg_order: list = []
    ensg_to_newidx: Dict[str, int] = {}
    ensg_to_symbol: Dict[str, str] = {}
    rows, cols = [], []
    for i in kept_cols:
        ensg, sym = targets[i]
        if ensg not in ensg_to_newidx:
            ensg_to_newidx[ensg] = len(ensg_order)
            ensg_order.append(ensg)
            ensg_to_symbol[ensg] = sym
        rows.append(i)
        cols.append(ensg_to_newidx[ensg])

    n_old, n_new = adata.n_vars, len(ensg_order)
    incidence = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n_old, n_new)
    )

    X = adata.X
    X = sparse.csr_matrix(X) if not sparse.issparse(X) else X.tocsr()
    X_new = X @ incidence  # (n_obs x n_new); duplicate sources summed

    # Preserve integer dtype for raw counts.
    if np.issubdtype(adata.X.dtype, np.integer) or np.allclose(
        X.data[: min(len(X.data), 10000)], np.round(X.data[: min(len(X.data), 10000)])
    ):
        X_new = X_new.astype(np.int64) if sparse.issparse(X_new) else X_new.round().astype(np.int64)

    symbols = [ensg_to_symbol[e] for e in ensg_order]
    var = pd.DataFrame(
        {"feature_name": symbols, "gene_symbol": symbols}, index=pd.Index(ensg_order, name=None)
    )
    out = ad.AnnData(X=X_new, obs=adata.obs.copy(), var=var)
    out.uns["humanization"] = {
        "source_namespace": source_namespace,
        "genes_in": int(n_in),
        "genes_mapped": int(len(kept_cols)),
        "genes_out": int(n_new),
        "collapsed": int(len(kept_cols) - n_new),
        "ortholog_policy": "one2one_high_confidence" if source_namespace == "mouse_ensembl" else "symbol_to_ensg",
        "mapping_source": "ensembl_biomart",
    }

    print(
        f"[gene_mapping] humanize({source_namespace}): {n_in} genes in -> "
        f"{len(kept_cols)} mapped -> {n_new} unique human ENSG "
        f"({len(kept_cols) - n_new} collapsed)"
    )
    return out
