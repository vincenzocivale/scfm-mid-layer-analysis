"""Convert downloaded GEO supplementary files into the h5ad targets
declared in manifest.yaml.

Each dataset has its own `assemble_<name>` function — the supplementary
files vary too much to share a single template.  All assemblers:
  - read inputs from data/raw/<task>/_geo/<accession>/
  - write a single h5ad to the manifest target
  - leave .X as raw counts (sparse CSR) and the cell-type / time / perturb
    column under the manifest-declared `obs_required[0]` name
  - print n_obs, n_vars, and a value_counts() of the key column

Usage:
    python scripts/dataset_downloads/assemble.py <name> [<name> ...]
    python scripts/dataset_downloads/assemble.py --all
"""
from __future__ import annotations

import argparse
import gzip
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import yaml

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "scripts/dataset_downloads/manifest.yaml"

# Scripts may inject src/ so scfm_eval helpers (e.g. gene humanization) import.
_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _man(name: str) -> dict:
    m = yaml.safe_load(MANIFEST.read_text())["datasets"]
    if name not in m:
        raise SystemExit(f"{name} not in manifest")
    return m[name]


def _geo_dir(entry: dict) -> Path:
    return REPO / "data" / "raw" / entry["task"] / "_geo" / entry["source"]


def _save(adata, entry: dict, key: str):
    target = REPO / entry["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    # Make .X sparse if it isn't (saves disk, matches scanpy convention)
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    # h5py can't write object columns with mixed types (e.g. bool+NaN from
    # a left-join).  Coerce all object dtypes to string and bool→str.
    for c in adata.obs.columns:
        col = adata.obs[c]
        if col.dtype == object or col.dtype == bool:
            adata.obs[c] = col.astype(str)
    adata.write_h5ad(target, compression="gzip")
    print(f"   saved {target}")
    print(f"   n_obs={adata.n_obs}  n_vars={adata.n_vars}")
    if key in adata.obs.columns:
        vc = adata.obs[key].value_counts().head(10)
        print(f"   {key} value_counts (top 10):\n{vc.to_string()}")


# ============================================================
# Classification
# ============================================================

def assemble_liver_ma():
    e = _man("liver_ma")
    d = _geo_dir(e)
    mtx = sio.mmread(d / "GSE151530_matrix.mtx.gz").tocsr().T   # -> cells × genes
    barcodes = pd.read_csv(d / "GSE151530_barcodes.tsv.gz", header=None,
                           sep="\t")[0].values
    genes = pd.read_csv(d / "GSE151530_genes.tsv.gz", header=None,
                        sep="\t")
    # Genes file may have 1 or 2 cols (id, symbol); use last col as symbol
    gene_symbols = genes.iloc[:, -1].astype(str).values
    info = pd.read_csv(d / "GSE151530_Info.txt.gz", sep="\t")
    # Cell column may be 'Cell', 'Barcode', 'Cell.Barcode', 'Cells' — find it
    cell_col = next((c for c in info.columns if c.lower() in
                     {"cell", "barcode", "cells", "cell.barcode"}), info.columns[0])
    type_col = next((c for c in info.columns if c.lower() in
                     {"type", "celltype", "cell_type"}), None)
    info = info.set_index(cell_col)
    obs = pd.DataFrame(index=barcodes)
    obs.index.name = "cell_barcode"
    obs = obs.join(info, how="left")
    if type_col:
        obs["cell_type"] = obs[type_col]
    var = pd.DataFrame({"gene_symbol": gene_symbols},
                       index=pd.Index(gene_symbols, name="gene"))
    var = var[~var.index.duplicated(keep="first")]
    # Align mtx columns to deduplicated var
    keep_cols = ~pd.Index(gene_symbols).duplicated(keep="first")
    mtx = mtx[:, keep_cols]
    a = ad.AnnData(X=mtx, obs=obs, var=var)
    _save(a, e, "cell_type")


def assemble_lung_kim():
    """Streaming sparse parser.  Matrix is 29,635 genes × 208,506 cells
    (dense ~25 GB in int32); we stream gene rows and build a COO matrix
    oriented as cells × genes."""
    e = _man("lung_kim")
    d = _geo_dir(e)
    mtx_f = d / "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz"
    ann = pd.read_csv(d / "GSE131907_Lung_Cancer_cell_annotation.txt.gz",
                      sep="\t").set_index("Index")
    print(f"   streaming {mtx_f.name} ...")
    rows, cols, vals = [], [], []           # cells × genes COO
    gene_names = []
    with gzip.open(mtx_f, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        cell_barcodes = header[1:]          # first token "Index"
        n_cells = len(cell_barcodes)
        print(f"   n_cells from header = {n_cells}")
        gi = 0
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            gene_names.append(parts[0])
            arr = np.asarray(parts[1:], dtype=np.int32)
            nz = np.nonzero(arr)[0]
            if nz.size:
                rows.extend(nz.tolist())     # cell idx
                cols.extend([gi] * nz.size)  # gene idx
                vals.extend(arr[nz].tolist())
            gi += 1
            if gi % 1000 == 0:
                print(f"   parsed {gi} genes, nnz ~ {len(vals):,}")
    n_genes = gi
    print(f"   total cells={n_cells} genes={n_genes} nnz={len(vals):,}")
    X = sp.coo_matrix((vals, (rows, cols)),
                      shape=(n_cells, n_genes), dtype=np.int32).tocsr()

    obs = pd.DataFrame(index=cell_barcodes).join(ann, how="left")
    if "Cell_type" in obs.columns:
        obs["cell_type"] = obs["Cell_type"]
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene"))
    a = ad.AnnData(X=X, obs=obs, var=var)
    _save(a, e, "cell_type")


def assemble_gse296117():
    """RA synovial-fluid scRNA-seq (single Seurat .rds) -> classification h5ad.

    The authors' object already carries 7 cell-type labels in `celltype`
    (Macrophage, T cell, DC, NK cell, Fibroblast, ...), so this is the one
    download that ships real cell-type annotations. We export the RNA assay's
    raw counts + metadata via R, then humanize the gene symbols to ENSG
    (var_names=ENSG, feature_name=symbol) so both ENSG-based (Tahoe) and
    symbol-based (scFoundation) models match. The object also has an
    `mnn.reconstructed` assay we deliberately ignore (not raw counts).
    """
    from scfm_eval.preprocessing.gene_mapping import humanize_adata

    e = _man("gse296117")
    d = REPO / "data" / "raw" / "classification" / "GSE296117"
    rds = d / "GSE296117_RA_geo.rds"
    exp = d / "export"

    if not (exp / "counts.mtx").exists():
        rscript = REPO / "scripts" / "dataset_downloads" / "export_seurat_rds.R"
        print("   running R export (Seurat RNA counts -> mtx/meta) ...")
        subprocess.run(["Rscript", str(rscript), str(rds), str(exp), "RNA"], check=True)

    genes = (exp / "genes.txt").read_text().splitlines()
    barcodes = (exp / "barcodes.txt").read_text().splitlines()
    meta = pd.read_csv(exp / "meta.csv", index_col=0)
    mtx = sio.mmread(exp / "counts.mtx").tocsr().T.tocsr()   # -> cells x genes
    if mtx.shape != (len(barcodes), len(genes)):
        raise ValueError(f"mtx {mtx.shape} vs barcodes={len(barcodes)} genes={len(genes)}")

    obs = meta.reindex(barcodes)
    obs.index = pd.Index(barcodes, name="cell_barcode")
    if "celltype" not in obs.columns:
        raise KeyError(f"`celltype` not in meta columns: {list(obs.columns)}")
    obs["cell_type"] = obs["celltype"]

    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    a = ad.AnnData(X=mtx, obs=obs, var=var)
    # symbols -> human ENSG (collapsed-gene counts summed); sets feature_name=symbol
    a = humanize_adata(a, "human_symbol")
    _save(a, e, "cell_type")


# ============================================================
# Pseudotime
# ============================================================

def assemble_emt_cook():
    e = _man("emt_cook")
    d = _geo_dir(e)
    X = pd.read_csv(d / "GSE147405_A549_TGFB1_TimeCourse_UMI_matrix.csv.gz",
                    index_col=0)         # genes × cells
    meta = pd.read_csv(d / "GSE147405_A549_TGFB1_TimeCourse_metadata.csv.gz",
                       index_col=0)
    obs = pd.DataFrame(index=X.columns).join(meta, how="left")
    a = ad.AnnData(X=sp.csr_matrix(X.values.T), obs=obs,
                   var=pd.DataFrame(index=X.index))
    _save(a, e, "Time")


def assemble_veres():
    """Streaming sparse parser: the raw_indrops tsv is ~111k genes ×
    41k cells.  Dense pandas read OOMs (~95 GB); we stream line-by-line
    and build a COO matrix directly (cells × genes, since we transpose
    on the fly: each tsv row becomes one column of X)."""
    e = _man("veres")
    d = _geo_dir(e)
    counts_f = d / "GSE114412_Stage_5.raw_indrops_counts.tsv.gz"
    meta = pd.read_csv(d / "GSE114412_Stage_5.all.cell_metadata.tsv.gz",
                       sep="\t", index_col=0)

    print(f"   streaming {counts_f.name} ...")
    # Layout: cells in rows (data rows), genes in columns (header).
    rows, cols, vals = [], [], []   # cells × genes COO
    cell_barcodes = []
    with gzip.open(counts_f, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        gene_names = header[1:]
        n_genes = len(gene_names)
        print(f"   n_genes from header = {n_genes}")
        ci = 0
        first_gene = gene_names[0]
        skipped_headers = 0
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            # The file concatenates multiple batches and re-injects header
            # rows in the middle (parts[1] becomes the gene name again).
            if parts[1] == first_gene:
                skipped_headers += 1
                continue
            cell_barcodes.append(parts[0])
            arr = np.asarray(parts[1:], dtype=np.int32)
            nz = np.nonzero(arr)[0]
            if nz.size:
                rows.extend([ci] * nz.size)              # cell idx
                cols.extend(nz.tolist())                 # gene idx
                vals.extend(arr[nz].tolist())
            ci += 1
            if ci % 10000 == 0:
                print(f"   parsed {ci} cells, nnz so far ~ {len(vals):,}")
    n_cells = ci
    print(f"   total cells={n_cells} genes={n_genes} nnz={len(vals):,} "
          f"(skipped {skipped_headers} re-header lines)")
    X = sp.coo_matrix((vals, (rows, cols)),
                      shape=(n_cells, n_genes), dtype=np.int32).tocsr()

    obs = pd.DataFrame(index=cell_barcodes).join(meta, how="left")
    for cand in ("day", "Day", "time_point", "timepoint", "CellWeek"):
        if cand in obs.columns:
            obs["day"] = obs[cand]
            break
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene"))
    a = ad.AnnData(X=X, obs=obs, var=var)
    _save(a, e, "day")


def assemble_hspc_bouman():
    e = _man("hspc_bouman")
    d = _geo_dir(e)
    src = d / "GSE226824_HSPC-all_filtered.h5ad.gz"
    tgt = REPO / e["target"]
    tgt.parent.mkdir(parents=True, exist_ok=True)
    print(f"   gunzip {src.name} -> {tgt}")
    with gzip.open(src, "rb") as fi, open(tgt, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    a = ad.read_h5ad(tgt, backed="r")
    print(f"   n_obs={a.n_obs}  n_vars={a.n_vars}")
    print(f"   obs columns: {list(a.obs.columns)[:20]}")


def _mouse_age_to_dpc(age: str) -> float:
    """Mouse age string -> days-post-coitum on one monotonic axis.

    Embryonic stages keep their day (E12.5 -> 12.5).  Postnatal ages are
    placed after birth (~E19.5): P56 -> 19.5 + 56 = 75.5.
    """
    a = str(age).strip()
    if a.startswith("E"):
        return float(a[1:])
    if a.startswith("P"):
        return 19.5 + float(a[1:])
    return float("nan")


def _parse_series_matrix(path: Path) -> dict:
    """Parse a GEO series_matrix.txt.gz into {GSM: {characteristic: value}}.

    Reads !Sample_geo_accession, !Sample_title and every
    !Sample_characteristics_ch1 line (each carries one `key: value` per
    sample), keyed by GSM accession.
    """
    rows: dict = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!Sample_"):
                parts = line.rstrip("\n").split("\t")
                vals = [v.strip().strip('"') for v in parts[1:]]
                rows.setdefault(parts[0], []).append(vals)
    gsms = rows["!Sample_geo_accession"][0]
    titles = rows["!Sample_title"][0]
    meta = {g: {"title": t} for g, t in zip(gsms, titles)}
    for cl in rows.get("!Sample_characteristics_ch1", []):
        for g, cell in zip(gsms, cl):
            if ":" in cell:
                k, v = cell.split(":", 1)
                meta[g][k.strip()] = v.strip()
    return meta


def assemble_gse322764():
    """Mouse OINS development (19 GSM 10x triplets) -> one pseudotime h5ad.

    var_names = ENSMUSG Ensembl IDs (+ `feature_name`/`gene_symbol` columns
    with mouse symbols).  Per-cell metadata (age, tissue, cell-type,
    genotype) is joined from the series_matrix; `age_numeric` is mouse dpc
    and `icn_series` flags the clean E12.5->P56 intrinsic-cardiac-neuron
    developmental panel (GSM9558094-100).
    """
    e = _man("gse322764")
    # Tar was placed manually under the canonical pseudotime dir (not _geo).
    d = REPO / "data" / "raw" / "pseudotime" / "GSE322764"
    tar = d / "GSE322764_RAW.tar"
    smx = d / "GSE322764_series_matrix.txt.gz"

    smeta = _parse_series_matrix(smx)
    extract_dir = d / "extracted"
    if not extract_dir.exists():
        extract_dir.mkdir()
        print(f"   extracting {tar.name}")
        with tarfile.open(tar) as tf:
            tf.extractall(extract_dir)

    gsm_ids = sorted({p.name.split("_")[0] for p in extract_dir.iterdir()})
    print(f"   {len(gsm_ids)} samples: {gsm_ids}")
    icn_series = {f"GSM{n}" for n in range(9558094, 9558101)}  # 094..100 incl P56

    parts = []
    for gsm in gsm_ids:
        files = list(extract_dir.glob(f"{gsm}_*"))
        prefix = files[0].name.split("_barcodes")[0]
        mtx_f = next(f for f in files if "matrix.mtx" in f.name)
        bc_f = next(f for f in files if "barcodes.tsv" in f.name)
        ft_f = next(f for f in files if "features.tsv" in f.name)

        bc = pd.read_csv(bc_f, header=None, sep="\t")[0].values
        ft = pd.read_csv(ft_f, header=None, sep="\t")     # ensembl, symbol, type
        mtx = sio.mmread(mtx_f).tocsr()                   # genes x cells
        if mtx.shape[0] == len(ft) and mtx.shape[1] == len(bc):
            mtx = mtx.T.tocsr()                           # -> cells x genes
        elif not (mtx.shape[0] == len(bc) and mtx.shape[1] == len(ft)):
            raise ValueError(f"{prefix}: mtx {mtx.shape} vs bc={len(bc)} ft={len(ft)}")

        sm = smeta.get(gsm, {})
        age = sm.get("age", "")
        obs = pd.DataFrame(index=bc)
        obs.index.name = "cell_barcode"
        obs["gsm"] = gsm
        obs["sample"] = sm.get("title", prefix)
        obs["age"] = age
        obs["age_numeric"] = _mouse_age_to_dpc(age)
        obs["tissue"] = sm.get("tissue", "")
        obs["cell_type_geo"] = sm.get("cell type", "")
        obs["genotype"] = sm.get("genotype", "")
        obs["icn_series"] = gsm in icn_series

        ens = ft.iloc[:, 0].astype(str).values
        sym = ft.iloc[:, 1].astype(str).values
        var = pd.DataFrame({"feature_name": sym, "gene_symbol": sym},
                           index=pd.Index(ens, name="ensembl_id"))
        keep = ~var.index.duplicated(keep="first")
        var = var[keep]
        mtx = mtx[:, np.asarray(keep)]
        parts.append(ad.AnnData(X=mtx, obs=obs, var=var))
        print(f"   {prefix} [{gsm}] age={age}: n_obs={parts[-1].n_obs} "
              f"n_vars={parts[-1].n_vars}")

    # Inner join on the shared Cell Ranger reference (all 19 share it).
    a = ad.concat(parts, join="inner", merge="same",
                  label="batch", index_unique="-")
    # `merge="same"` keeps var only if identical across parts; re-attach to be safe.
    a.var = parts[0].var.loc[a.var_names]
    _save(a, e, "age")


_SHEEP_STAGE_DPC = {                       # ordered sheep stages -> days-post-coitum
    "E11": 11.0, "E11.5": 11.5,
    "E12.5E": 12.5, "E12.5L": 12.75,       # E/L = early/late within the day
    "E13.5E": 13.5, "E13.5L": 13.75,
}


def assemble_gse320427():
    """Sheep gastrulation atlas (single Seurat .rds) -> one pseudotime h5ad.

    The .rds rownames are human-ortholog UPPERCASE symbols (ENSOARG… only as
    fallback), so the human-symbol FMs match most genes via feature_name.
    A helper R script exports counts+meta; here we assemble them and add a
    numeric `stage_numeric` from the per-cell `stage` (E11..E13.5L).
    """
    e = _man("gse320427")
    d = REPO / "data" / "raw" / "pseudotime" / "GSE320427"
    rds = d / "GSM9543171_in_vivo.rds"
    exp = d / "export"

    if not (exp / "counts.mtx").exists():
        if not rds.exists():
            tar = d / "GSE320427_RAW.tar"
            print(f"   extracting {rds.name} from {tar.name}")
            with tarfile.open(tar) as tf:
                tf.extract("GSM9543171_in_vivo.rds", d)
        rscript = REPO / "scripts" / "dataset_downloads" / "export_seurat_rds.R"
        print("   running R export (Seurat -> mtx/meta) ...")
        subprocess.run(["Rscript", str(rscript), str(rds), str(exp)], check=True)

    genes = (exp / "genes.txt").read_text().splitlines()
    barcodes = (exp / "barcodes.txt").read_text().splitlines()
    meta = pd.read_csv(exp / "meta.csv", index_col=0)
    mtx = sio.mmread(exp / "counts.mtx").tocsr().T.tocsr()   # -> cells x genes
    if mtx.shape != (len(barcodes), len(genes)):
        raise ValueError(f"mtx {mtx.shape} vs barcodes={len(barcodes)} genes={len(genes)}")

    obs = meta.reindex(barcodes)
    obs.index = pd.Index(barcodes, name="cell_barcode")
    obs["stage_numeric"] = obs["stage"].map(_SHEEP_STAGE_DPC)

    var = pd.DataFrame({"feature_name": genes, "gene_symbol": genes},
                       index=pd.Index(genes, name="gene"))
    keep = ~var.index.duplicated(keep="first")
    var = var[keep]
    mtx = mtx[:, np.asarray(keep)]

    a = ad.AnnData(X=mtx, obs=obs, var=var)
    print(f"   stage_numeric NaN: {int(a.obs['stage_numeric'].isna().sum())}")
    _save(a, e, "stage")


def _parse_series_matrix_all_sample_fields(path: Path) -> dict:
    """Parse GEO sample-level fields into {GSM: metadata}.

    Keeps common !Sample_* fields plus parsed `key: value` characteristics.
    """
    rows: dict[str, list[list[str]]] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!Sample_"):
                parts = line.rstrip("\n").split("\t")
                vals = [v.strip().strip('"') for v in parts[1:]]
                rows.setdefault(parts[0], []).append(vals)
    gsms = rows["!Sample_geo_accession"][0]
    meta = {g: {"gsm": g} for g in gsms}
    for field, out_key in [
        ("!Sample_title", "title"),
        ("!Sample_source_name_ch1", "source_name"),
        ("!Sample_description", "description"),
        ("!Sample_library_strategy", "library_strategy"),
    ]:
        for vals in rows.get(field, []):
            for g, v in zip(gsms, vals):
                meta[g][out_key] = v
    for vals in rows.get("!Sample_characteristics_ch1", []):
        for g, cell in zip(gsms, vals):
            if ":" in cell:
                k, v = cell.split(":", 1)
                meta[g][k.strip().replace(" ", "_")] = v.strip()
    return meta


def _read_tar_gzip_table(tf: tarfile.TarFile, member_name: str, **kwargs):
    with tf.extractfile(member_name) as raw:
        if raw is None:
            raise FileNotFoundError(member_name)
        with gzip.GzipFile(fileobj=raw) as gz:
            return pd.read_csv(gz, **kwargs)


def _read_tar_gzip_mtx(tf: tarfile.TarFile, member_name: str):
    with tf.extractfile(member_name) as raw:
        if raw is None:
            raise FileNotFoundError(member_name)
        with gzip.GzipFile(fileobj=raw) as gz:
            return sio.mmread(gz).tocsr()


def assemble_gse277032_organoids():
    """Human midbrain organoid scRNA-seq time course -> pseudotime h5ad.

    Uses only GSM8513025-GSM8513033, the nine single-cell organoid RNA-seq
    libraries with sample-level differentiation days 40, 70, and 120.  Spatial
    Visium and pooled fetal libraries are intentionally excluded.
    """
    e = _man("gse277032_organoids")
    d = REPO / "data" / "raw" / "pseudotime" / "GSE277032"
    tar = d / "GSE277032_RAW.tar"
    if not tar.exists():
        tar = d / "index.html?acc=GSE277032&format=file"
    smeta = _parse_series_matrix_all_sample_fields(
        d / "GSE277032-GPL20301_series_matrix.txt.gz"
    )

    parts = []
    with tarfile.open(tar) as tf:
        names = tf.getnames()
        organoid_gsms = [f"GSM85130{i}" for i in range(25, 34)]
        for gsm in organoid_gsms:
            sample_files = [n for n in names if n.startswith(f"{gsm}_")]
            mtx_f = next(n for n in sample_files if n.endswith("_matrix.mtx.gz"))
            bc_f = next(n for n in sample_files if n.endswith("_barcodes.tsv.gz"))
            ft_f = next(n for n in sample_files if n.endswith("_features.tsv.gz"))

            bc = _read_tar_gzip_table(tf, bc_f, header=None, sep="\t")[0].astype(str).values
            ft = _read_tar_gzip_table(tf, ft_f, header=None, sep="\t")
            mtx = _read_tar_gzip_mtx(tf, mtx_f)
            if mtx.shape[0] == len(ft) and mtx.shape[1] == len(bc):
                mtx = mtx.T.tocsr()
            elif not (mtx.shape[0] == len(bc) and mtx.shape[1] == len(ft)):
                raise ValueError(f"{gsm}: mtx {mtx.shape} vs bc={len(bc)} ft={len(ft)}")

            sm = smeta[gsm]
            source = sm.get("source_name", "")
            day_match = re.search(r"day\s+(\d+)", source, flags=re.IGNORECASE)
            if day_match is None:
                raise ValueError(f"{gsm}: cannot parse differentiation day from {source!r}")
            day = int(day_match.group(1))

            obs = pd.DataFrame(index=pd.Index([f"{gsm}_{x}" for x in bc], name="cell_barcode"))
            obs["barcode"] = bc
            obs["gsm"] = gsm
            obs["sample"] = sm.get("title", gsm)
            obs["source_name"] = source
            obs["day"] = day
            obs["tissue"] = sm.get("tissue", "")
            obs["cell_line"] = sm.get("cell_line", "")
            obs["cell_type_geo"] = sm.get("cell_type", "")

            ens = ft.iloc[:, 0].astype(str).values
            sym = ft.iloc[:, 1].astype(str).values if ft.shape[1] > 1 else ens
            var = pd.DataFrame(
                {"feature_name": sym, "gene_symbol": sym},
                index=pd.Index(ens, name="ensembl_id"),
            )
            keep = ~var.index.duplicated(keep="first")
            var = var[keep]
            mtx = mtx[:, np.asarray(keep)]
            parts.append(ad.AnnData(X=mtx, obs=obs, var=var))
            print(f"   {gsm} day={day}: n_obs={parts[-1].n_obs} n_vars={parts[-1].n_vars}")

    a = ad.concat(parts, join="inner", merge="same",
                  label="sample_id", index_unique=None)
    a.var = parts[0].var.loc[a.var_names]
    _save(a, e, "day")


def assemble_gse307094_counts():
    """Assemble GSE307094 counts, without claiming benchmark-ready time labels."""
    e = _man("gse307094_counts")
    d = REPO / "data" / "raw" / "pseudotime" / "GSE307094"
    mtx = sio.mmread(d / "GSE307094_expression_matrix_counts.mtx.gz").tocsr()
    barcodes = pd.read_csv(
        d / "GSE307094_expression_matrix_counts_barcodes.csv.gz"
    )["x"].astype(str).values
    genes = pd.read_csv(
        d / "GSE307094_expression_matrix_counts_genes.csv.gz"
    )["x"].astype(str).values
    if mtx.shape[0] == len(genes) and mtx.shape[1] == len(barcodes):
        mtx = mtx.T.tocsr()
    elif not (mtx.shape[0] == len(barcodes) and mtx.shape[1] == len(genes)):
        raise ValueError(f"mtx {mtx.shape} vs barcodes={len(barcodes)} genes={len(genes)}")

    barcode_batch = pd.Series(barcodes).str.rsplit("_", n=1).str[-1].values
    obs = pd.DataFrame(index=pd.Index(barcodes, name="cell_barcode"))
    obs["barcode"] = barcodes
    obs["barcode_batch"] = barcode_batch
    obs["time_label_status"] = "unmapped: seven barcode suffixes do not match eleven RNA GEO samples"

    var = pd.DataFrame(
        {"feature_name": genes, "gene_symbol": genes},
        index=pd.Index(genes, name="gene"),
    )
    keep = ~var.index.duplicated(keep="first")
    var = var[keep]
    mtx = mtx[:, np.asarray(keep)]

    a = ad.AnnData(X=mtx, obs=obs, var=var)
    _save(a, e, "barcode_batch")


# ============================================================
# Perturbation
# ============================================================

def assemble_norman():
    e = _man("norman")
    d = _geo_dir(e)
    barcodes = pd.read_csv(d / "GSE133344_filtered_barcodes.tsv.gz",
                           header=None, sep="\t")[0].values
    genes = pd.read_csv(d / "GSE133344_filtered_genes.tsv.gz", header=None,
                        sep="\t")
    gene_symbols = genes.iloc[:, -1].astype(str).values
    mtx = sio.mmread(d / "GSE133344_filtered_matrix.mtx.gz").tocsr()
    # Orient so rows = cells, cols = genes (anchor to file lengths, not
    # which dim is larger — Norman has cells > genes).
    if mtx.shape[0] == len(gene_symbols) and mtx.shape[1] == len(barcodes):
        mtx = mtx.T.tocsr()
    elif not (mtx.shape[0] == len(barcodes)
              and mtx.shape[1] == len(gene_symbols)):
        raise ValueError(f"mtx shape {mtx.shape} vs barcodes={len(barcodes)} "
                         f"genes={len(gene_symbols)}")
    cell_id = pd.read_csv(d / "GSE133344_filtered_cell_identities.csv.gz")
    bc_col = next((c for c in cell_id.columns if c.lower() in
                   {"cell_barcode", "barcode", "cell"}), cell_id.columns[0])
    cell_id = cell_id.set_index(bc_col)
    obs = pd.DataFrame(index=barcodes).join(cell_id, how="left")
    # `guide_identity` is the conventional Norman column name
    var = pd.DataFrame({"gene_symbol": gene_symbols},
                       index=pd.Index(gene_symbols, name="gene"))
    keep = ~pd.Index(gene_symbols).duplicated(keep="first")
    var = var[keep]
    mtx = mtx[:, np.asarray(keep)]
    a = ad.AnnData(X=mtx, obs=obs, var=var)
    _save(a, e, "guide_identity")


def assemble_adamson():
    e = _man("adamson")
    d = _geo_dir(e)
    tar = d / "GSE90546_RAW.tar"
    extract_dir = d / "extracted"
    if not extract_dir.exists():
        extract_dir.mkdir()
        print(f"   extracting {tar.name}")
        with tarfile.open(tar) as tf:
            tf.extractall(extract_dir)
    # Group files by GSM prefix
    gsm_ids = sorted({p.name.split("_")[0] for p in extract_dir.iterdir()})
    print(f"   {len(gsm_ids)} samples: {gsm_ids}")
    parts = []
    for gsm in gsm_ids:
        files = list(extract_dir.glob(f"{gsm}_*"))
        prefix = files[0].name.split("_barcodes")[0]  # e.g. GSM2406675_10X001
        mtx_f = next(f for f in files if "matrix.mtx" in f.name)
        bc_f = next(f for f in files if "barcodes.tsv" in f.name)
        gn_f = next(f for f in files if "genes.tsv" in f.name)
        cid_f = next(f for f in files if "cell_identities" in f.name)
        bc = pd.read_csv(bc_f, header=None, sep="\t")[0].values
        gn = pd.read_csv(gn_f, header=None, sep="\t")
        mtx = sio.mmread(mtx_f).tocsr()
        if mtx.shape[0] == len(gn) and mtx.shape[1] == len(bc):
            mtx = mtx.T.tocsr()
        elif not (mtx.shape[0] == len(bc) and mtx.shape[1] == len(gn)):
            raise ValueError(f"{prefix}: mtx {mtx.shape} vs bc={len(bc)} "
                             f"gn={len(gn)}")
        cid = pd.read_csv(cid_f)
        bc_col = next((c for c in cid.columns
                       if c.lower() in {"cell_barcode", "barcode", "cell"}),
                      cid.columns[0])
        cid = cid.set_index(bc_col)
        obs = pd.DataFrame(index=bc).join(cid, how="left")
        obs["sample"] = prefix
        # Resolve a `perturbation` column (Adamson uses `guide_identity` /
        # `gene` depending on replicate); coalesce.
        for cand in ("perturbation", "guide_identity", "gene", "target_gene"):
            if cand in obs.columns:
                obs["perturbation"] = obs[cand]
                break
        var = pd.DataFrame({"gene_symbol": gn.iloc[:, -1].astype(str).values},
                           index=pd.Index(gn.iloc[:, -1].astype(str).values,
                                          name="gene"))
        keep = ~var.index.duplicated(keep="first")
        var = var[keep]
        mtx = mtx[:, np.asarray(keep)]
        parts.append(ad.AnnData(X=mtx, obs=obs, var=var))
        print(f"   {prefix}: n_obs={parts[-1].n_obs} n_vars={parts[-1].n_vars}")
    # Concatenate on common genes (inner join)
    a = ad.concat(parts, join="inner", merge="same",
                  label="sample_id", index_unique="-")
    _save(a, e, "perturbation")


ASSEMBLERS = {
    "liver_ma": assemble_liver_ma,
    "lung_kim": assemble_lung_kim,
    "gse296117": assemble_gse296117,
    "emt_cook": assemble_emt_cook,
    "veres": assemble_veres,
    "hspc_bouman": assemble_hspc_bouman,
    "gse322764": assemble_gse322764,
    "gse320427": assemble_gse320427,
    "gse277032_organoids": assemble_gse277032_organoids,
    "gse307094_counts": assemble_gse307094_counts,
    "norman": assemble_norman,
    "adamson": assemble_adamson,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    names = list(ASSEMBLERS.keys()) if args.all else args.names
    if not names:
        p.error("pass dataset names or --all")

    for n in names:
        if n not in ASSEMBLERS:
            print(f"!! no assembler for {n}", file=sys.stderr)
            continue
        print(f"\n=== assemble {n} ===")
        try:
            ASSEMBLERS[n]()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"!! {n} FAILED: {exc}")


if __name__ == "__main__":
    main()
