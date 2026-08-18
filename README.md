# scfm-layer-analysis

Benchmark per confrontare come i **singoli layer** dei modelli foundation per single-cell (scFoundation, Tahoe-X1, opzionalmente scGPT) rappresentano la biologia. Gli embedding di ogni layer vengono valutati con tre pipeline indipendenti: classificazione dei tipi cellulari, inferenza di pseudo-time e risposta a perturbazioni.

## Setup

Python 3.9 in un venv gestito con uv:

```bash
.venv/bin/python --version  # Python 3.9.7
```

Per scFoundation servono due file (gitignorati) in `data/checkpoints/scfoundation/`:
- `models.ckpt` (~1.4 GB)
- `OS_scRNA_gene_index.19264.tsv`

Path overridabili via env var:
```bash
export SCFOUNDATION_CKPT=/path/to/models.ckpt
export SCFOUNDATION_GENE_INDEX=/path/to/OS_scRNA_gene_index.19264.tsv
```

Tahoe-X1 scarica i pesi da Hugging Face (`tahoebio/tahoe-x1`) al primo avvio.

## Workflow

### 1. Estrazione embedding

Lo script shell rileva automaticamente il numero di layer e processa l'h5ad a chunk:

```bash
# scFoundation (12 layer)
MODELS=scfoundation CHUNK_SIZE=20000 BATCH_SIZE=1 \
  ./run_embedding_extraction.sh data/raw/classification/brain_dataset.h5ad

# Tahoe-X1 (model size 70m / 1b / 3b)
MODELS=tahoe TAHOE_MODEL_SIZE=1b CHUNK_SIZE=50000 BATCH_SIZE=16 \
  ./run_embedding_extraction.sh data/raw/classification/liver_dataset.h5ad --no-fp16
```

Output in `data/embeddings/<input>_<model>[_<size>]/`.

### 2. Pipeline di benchmark

Ogni pipeline scopre gli embedding per layer cercando `X_layer_*` in `adata.obsm`.

```bash
# Classificazione tipi cellulari
./.venv/bin/python scripts/run_cell_type_classification.py \
  --input data/embeddings/brain_dataset_scfoundation_embeddings.h5ad \
  --cell_type_column cell_type

# Pseudo-time
./.venv/bin/python scripts/run_pseudotime_analysis.py \
  --input data/embeddings/inner_ear_development_tahoe_1b.h5ad \
  --time_column week

# Perturbation
./.venv/bin/python scripts/run_perturbation_analysis.py \
  --input data/embeddings/D1_Stim8hr_scfoundation_embeddings.h5ad \
  --output_dir data/perturbation_metrics \
  --perturb_key gene_name --control_label control
```

Output CSV in `data/{classification,pseudotime}_results/` e `data/perturbation_metrics/`.

## Dataset di riferimento

- **Brain & spinal cord**: [CellxGene collection 0986e4cd-7a58-405d-9b91-4b199bb4124e](https://cellxgene.cziscience.com/collections/0986e4cd-7a58-405d-9b91-4b199bb4124e)
- **Liver**: [CellxGene collection be679cb1-35f0-46c9-9a2d-30691862a54a](https://cellxgene.cziscience.com/collections/be679cb1-35f0-46c9-9a2d-30691862a54a)

## Layout

```
src/scfm_eval/                   # library: embedders, benchmarks, extraction, io
scripts/                         # CLI entry points (argparse)
config/                          # YAML registries: models, datasets, experiments
docs/                            # documentazione architetturale per il paper
run_embedding_extraction.sh      # orchestratore stage 1
notebooks/                       # EDA + paper_figures.ipynb
data/                            # input, checkpoints, embeddings, results (gitignored)
```

Per dettagli architetturali vedi `docs/` (start `docs/README.md`) e `CLAUDE.md`.
