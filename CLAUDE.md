# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A benchmarking framework that compares how the **per-layer hidden states** of single-cell foundation models (scFoundation, Tahoe-X1, optionally scGPT) represent biology. The same set of embeddings (one per transformer layer of one model on one dataset) is fed through three independent evaluation pipelines: cell-type classification, pseudo-time inference, and perturbation response. Code comments and CLI help strings are mostly in Italian.

## Environment

- Python 3.9 in `.venv/` (uv-managed; no `pyproject.toml` is checked in).
- Always invoke `./.venv/bin/python` (or activate the venv) — system Python will not have `tahoe_x1`, `scanpy`, `cuml`, etc.
- GPU is optional. `inference.py` is hardcoded to CPU; the embedder scripts auto-detect CUDA and use `fp16` on GPU by default (`--no-fp16` to disable). PCA preprocessing optionally uses `cupy`/`cuml` when available.

## Workflow: extract embeddings, then benchmark

The two stages are **decoupled by h5ad files**: extraction writes per-layer embeddings into `adata.obsm['X_layer_{i}']`, and every benchmark pipeline discovers them by that prefix. Treat that prefix as a contract — changing it breaks all three benchmarks.

### Stage 1 — extract embeddings

The orchestrator shell script auto-detects the number of layers from `models/get_model_info.py`, then loops layers through the chunked extractor. Inputs are positional; flags pass through; config is via env vars.

```bash
MODELS=scfoundation CHUNK_SIZE=20000 BATCH_SIZE=1 \
  ./run_embedding_extraction.sh data/raw/brain_dataset.h5ad

# Tahoe variant — TAHOE_MODEL_SIZE picks 70m / 1b / 3b
MODELS=tahoe TAHOE_MODEL_SIZE=1b CHUNK_SIZE=50000 BATCH_SIZE=16 \
  ./run_embedding_extraction.sh data/raw/liver_dataset.h5ad --no-fp16
```

Outputs land in `data/embeddings/<input>_<model>[_<size>]/`.

### Stage 2 — run a benchmark on an embedding h5ad

Each pipeline lives in its own package (`cell_type_classification_benchmark/`, `pseudo_time_benchmark/`, `perturbation_analysis/`) and follows the same three-file layout: `preprocessing.py` → `evaluator.py` → `pipeline.py` exposes a single `run_*` function. The argparse launchers under `scripts/` are thin wrappers over those functions.

```bash
./.venv/bin/python scripts/run_cell_type_classification.py \
  --input data/embeddings/brain_dataset_scfoundation_embeddings.h5ad \
  --cell_type_column cell_type

./.venv/bin/python scripts/run_pseudotime_analysis.py \
  --input data/embeddings/GSE276896_..._timeordered.h5ad \
  --time_column week

./.venv/bin/python scripts/run_perturbation_analysis.py \
  --input data/embeddings/D1_Stim8hr...scfoundation_embeddings.h5ad \
  --output_dir data/perturbation_metrics --perturb_key gene_name --control_label control
```

All three default to discovering layer embeddings by `--embedding_prefix X_layer` and writing CSVs into `data/{classification,pseudotime,perturbation_metrics}_results/`. The classification pipeline additionally builds an `X_pca` baseline from `adata.raw` (HVG → PCA) and renames it `Baseline_PCA` in the output.

## Architecture: the embedder layer

`models/base_embedder.py` defines `BaseEmbedder` (abstract). Each concrete embedder (`scfoundation_embedder.py`, `tahoe_embedder.py`, optional `scgpt_embedder.py`) must implement `load_model`, `prepare_data`, `get_all_layer_indices`, and **must** override `extract_embeddings_for_layers(adata, layer_indices, batch_size) → {layer_idx: np.ndarray}`.

The pattern for multi-layer extraction is **forward hooks on transformer blocks** (see `TahoeEmbedder.extract_embeddings_for_layers`) so all requested layers come out of a single forward pass. `extract_embeddings_chunked_all_layers.py` then drives this chunk-by-chunk over a `backed='r'` AnnData to keep RAM bounded.

Model-specific gotchas:

- **scFoundation**: weight/gene-index paths default to `models/models.ckpt` and `models/OS_scRNA_gene_index.19264.tsv` (relative to `scfoundation_embedder.py`). Override via constructor args or env vars `SCFOUNDATION_CKPT` / `SCFOUNDATION_GENE_INDEX`. `prepare_data` remaps `adata.var_names` (Ensembl) to the 19,264 scFoundation gene-symbol vocabulary and zero-pads missing genes. It also requires/computes `obs['log_total_count']`.
- **Tahoe**: loads weights from HF (`tahoebio/tahoe-x1`) at init; `prepare_data` filters genes to the vocabulary by name.
- `models/load.py` is vendored from BioMap and provides `load_model_frommmf`, `gatherData`, etc. Don't edit unless touching scFoundation loading specifically.

## Conventions to preserve

- **Embedding keys**: `adata.obsm['X_layer_{i}']` for layer `i`. Metadata summary lives in `adata.uns['layer_embeddings']` (keys: `model`, `n_layers`, `layer_keys`).
- **Output paths**: pipelines compute `<input_stem>_results.csv` (or `_classification_results.csv`) inside their default `--output_dir`. The input file's stem is the canonical run identifier — don't rename embedding files mid-experiment.
- **Adding a new model**: subclass `BaseEmbedder`, register it in the two `if model == ...` branches (`get_model_info.py`, `extract_embeddings_chunked_all_layers.py`). Layer indices returned by `get_all_layer_indices()` must match the indices the hook-based extractor will register on.
- **Adding a new benchmark**: mirror the existing package layout (`preprocessing.py`, `evaluator.py`, `pipeline.py` with a `run_*` entry point), discover layers via `[k for k in adata.obsm if k.startswith(embedding_prefix)]`, and ship a `scripts/run_<name>.py` launcher that prepends the repo root to `sys.path`.

## Data layout

- `data/raw/` — input h5ad files (datasets sourced from CellxGene, see `README.md`).
- `data/embeddings/` — output of stage 1 (one h5ad per chunk under `<input>_<model>[_<size>]/`).
- `data/{classification,pseudotime}_results/`, `data/perturbation_metrics/` — output CSVs of stage 2.
- `models/models.ckpt` and `models/OS_scRNA_gene_index.19264.tsv` are gitignored (`*.ckpt`, `*.tsv`) — they must be present locally for scFoundation to load.
