# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A benchmarking framework that compares how the **per-layer hidden states** of single-cell foundation models (scFoundation, Tahoe-X1, scGPT, planned: Geneformer) represent biology. The same set of embeddings (one per transformer layer of one model on one dataset) is fed through three independent evaluation pipelines: cell-type classification, pseudo-time inference, and perturbation response. Code comments and CLI help strings are mostly in Italian.

For paper-grade documentation see `docs/` (start with `docs/README.md`).

## Environment

- Python 3.9 in `.venv/` (uv-managed; `pyproject.toml` declares the source layout).
- Either `./.venv/bin/pip install -e .` once, or rely on the `PYTHONPATH=src` injection in the shell/scripts. Both work.
- GPU optional. `fp16` is the default on CUDA; pass `--no-fp16` to disable.

## Repository layout

```
src/scfm_eval/                   # the only place library code lives
  embedders/                     # per-FM wrappers + base ABC + lazy registry
    base.py                      # BaseEmbedder ABC
    registry.py                  # name → class (lazy import; no torch at package import)
    scfoundation.py
    tahoe.py
    vendor/scfoundation/         # vendored upstream scFoundation code, isolated
  benchmarks/                    # downstream evaluation packages
    classification/  pseudotime/  perturbation/    # each: preprocessing → evaluator → pipeline
  extraction/
    chunked.py                   # `python -m scfm_eval.extraction.chunked` does stage 1
    model_info.py                # n_layers helper
  io/
    embeddings.py                # write/read embedded h5ads with structured metadata
    results.py                   # canonical CSV paths/writers for the 3 tasks

scripts/                         # CLI entry points (argparse → calls into scfm_eval)
  run_cell_type_classification.py
  run_perturbation_analysis.py
  run_pseudotime_analysis.py
  merge_chunks.py                # consolidate per-chunk h5ads into one
  aggregate_results.py           # produce data/results_all.csv long-format
  run_experiment_grid.py         # driver: reads config/experiments.yaml

config/                          # YAML registries (versioned)
  models.yaml  datasets.yaml  experiments.yaml

docs/                            # paper-grade docs (architecture, models, datasets, benchmarks, ...)

data/                            # gitignored
  raw/                           # input h5ads
  checkpoints/scfoundation/      # models.ckpt + gene-index TSV (gitignored)
  embeddings/                    # output of stage 1 (per-chunk h5ads + optional merged)
  classification_results/  pseudotime_results/  perturbation_metrics/
  results_all.csv                # output of aggregate_results.py
  run_manifest.csv               # output of run_experiment_grid.py

run_embedding_extraction.sh      # stage-1 orchestrator (loops inputs × models)
notebooks/
```

## Workflow

```
                   STAGE 1                                STAGE 2
                   ───────                                ───────
   raw h5ad   ─►  embedder.extract  ─►   embedded h5ad   ─►  pipeline.run_*()  ─►  CSV
              ─►  (chunked, all layers)      X_layer_{i}        (per-task evaluator)    metrics

                       │                          │                 │
                       │                          │                 ▼
                       │                          │            aggregate_results.py
                       │                          │                 │
                       ▼                          ▼                 ▼
              data/embeddings/         data/<task>_results/    data/results_all.csv
                                                                    │
                                                                    ▼
                                                           notebooks/paper_figures.ipynb
```

### Running the full grid

```bash
./.venv/bin/python scripts/run_experiment_grid.py
./.venv/bin/python scripts/aggregate_results.py
```

### Single cell of the grid

```bash
# Stage 1
MODELS=scfoundation ./run_embedding_extraction.sh data/raw/classification/tabula/tabula-blood.h5ad
./.venv/bin/python scripts/merge_chunks.py --input-dir data/embeddings/tabula-blood/chunks/scfoundation/

# Stage 2 (any of)
./.venv/bin/python scripts/run_cell_type_classification.py --input <merged.h5ad> --cell_type_column cell_type
./.venv/bin/python scripts/run_pseudotime_analysis.py     --input <merged.h5ad> --time_column week
./.venv/bin/python scripts/run_perturbation_analysis.py   --input <merged.h5ad> --output_dir data/perturbation_metrics --perturb_key gene_name --control_label non-targeting
```

## The `X_layer_*` contract

The synchronization point between stage 1 and stage 2 is `adata.obsm['X_layer_{i}']`. Every benchmark auto-discovers layers via `[k for k in adata.obsm if k.startswith('X_layer')]`. Don't rename this.

Stage 1 also writes structured metadata in `adata.uns['layer_embeddings']` (see `src/scfm_eval/io/embeddings.py` for the canonical schema). Critical fields: `n_layers_total`, `hidden_dim`, `pooling`, `expected_input`, `genes_matched`. These flow into `results_all.csv` so the paper plots can normalize by `relative_depth = layer / (n_layers_total - 1)` and compare models with different depth.

## Layer numbering semantics

`X_layer_i` = output of the i-th transformer block, after pooling. Pooling is model-specific:

| Model | Hook | Pooling |
|---|---|---|
| scFoundation | inline iteration of `encoder.transformer_encoder[i]` | `x[:, -1, :]` (S-token) |
| Tahoe-X1     | `register_forward_hook` on `transformer_encoder.layers[i]` | `output[:, 0, :]` (CLS) |

Layer 0 is post-first-block (not the embedding layer). Add a layer-naming sanity check before publishing.

## Adding things

- **New model**: see `docs/adding_a_model.md`. The lazy registry in `src/scfm_eval/embedders/registry.py` is the single place to wire it.
- **New benchmark**: see `docs/adding_a_benchmark.md`. Mirror an existing package under `src/scfm_eval/benchmarks/<task>/`, register the task in `src/scfm_eval/io/results.py` (`TASK_OUTPUT_DIRS`, `TASK_SUFFIX`) and in `scripts/run_experiment_grid.py` (`TASK_LAUNCHERS`).
- **New dataset**: declare in `config/datasets.yaml`. The driver will pick it up automatically when a run references it in `config/experiments.yaml`.

## Conventions to preserve

- **Embedding output naming**: stage 1 writes `data/embeddings/<input_stem>/chunks/<model>[_<size>]/<input_stem>_chunk_NNNN.h5ad`. `scripts/merge_chunks.py` produces `data/embeddings/<input_stem>/<input_stem>_<model>[_<size>].h5ad` — one level above the chunks folder, grouped by dataset.
- **Result naming**: `<merged_h5ad_stem><task_suffix>.csv` per `src/scfm_eval/io/results.py`. The aggregator parses these via `config/models.yaml` + `config/datasets.yaml`, so don't hand-rename them.
- **Subprocess per (model, dataset)** in the driver — gives clean GPU memory between runs and natural isolation.
- **Italian comments OK** in legacy code; new code prefers English (more searchable). Don't translate existing comments unless touching the surrounding code.

## Don't

- Don't add `sys.path` hacks to library code under `src/scfm_eval/`. Library code uses package-relative imports. Only `scripts/*.py` may inject `src/` into sys.path.
- Don't import torch / tahoe_x1 at package-import time. Keep them inside class methods or behind the lazy registry, so `from scfm_eval import ...` in a CPU-only analysis context stays fast.
- Don't hardcode absolute paths to weights. Use `data/checkpoints/<model>/` defaults overridable via env vars (`SCFOUNDATION_CKPT`, etc.).
- Don't leave debug/exploratory scripts in `scripts/` or `notebooks/` after the task is done — remove them.
