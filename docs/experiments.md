# Experiments

Come eseguire la griglia di valutazione del paper e dove finiscono i risultati.

## La griglia

Tutto è dichiarato in [`config/experiments.yaml`](../config/experiments.yaml). Schema:

```yaml
runs:
  - dataset: brain
    model: scfoundation
    tasks: [classification]
  - dataset: brain
    model: tahoe
    model_size: 1b
    tasks: [classification]
  - dataset: GSE276896
    model: scfoundation
    tasks: [pseudotime]
  ...
```

Il driver risolve ogni `run` come:
1. **Stage 1** — se `data/embeddings/<dataset>_<model>[_<size>]/` non esiste o è incompleto, lancia `run_embedding_extraction.sh`.
2. **Merge chunks** — se serve, consolida i chunk in un singolo h5ad (`scripts/merge_chunks.py`).
3. **Stage 2** — per ogni `task` in `tasks`, lancia il rispettivo `scripts/run_*.py` con i parametri risolti da `config/datasets.yaml`.

## Eseguire tutto

```bash
./.venv/bin/python scripts/run_experiment_grid.py --config config/experiments.yaml
```

### Opzioni utili

- `--filter dataset=brain,model=scfoundation` — esegue solo le run che matchano i predicati.
- `--filter tasks=classification` — solo classification, indipendentemente da (dataset, model).
- `--resume` — skippa le run il cui output CSV esiste già (default: true).
- `--dry-run` — stampa il piano senza eseguire.
- `--manifest data/run_manifest.csv` — append manifest di tutte le run con `(run_id, dataset, model, task, status, started, finished, error)`.

## Eseguire una singola cella della griglia

Le shell/script sottostanti restano direttamente invocabili — il driver è solo un orchestratore.

```bash
# Solo extraction
MODELS=scfoundation ./run_embedding_extraction.sh data/raw/brain_dataset.h5ad

# Solo classification su un embedding già estratto
./.venv/bin/python scripts/run_cell_type_classification.py \
  --input data/embeddings/brain_dataset_scfoundation/brain_dataset_scfoundation.h5ad \
  --cell_type_column cell_type
```

## Aggregazione

Quando la griglia è completa (o anche parziale), aggrega i CSV in un unico long-format:

```bash
./.venv/bin/python scripts/aggregate_results.py
```

Produce `data/results_all.csv` con colonne:

| Colonna | Esempio | Note |
|---|---|---|
| `dataset` | `brain` | da `config/datasets.yaml` |
| `model` | `scfoundation` | |
| `model_size` | `` (per scFoundation) o `1b` (per Tahoe) | |
| `task` | `classification` | uno di `classification | pseudotime | perturbation` |
| `layer` | `5` | int. La riga `Baseline_PCA` ha `layer=-1` per convenzione |
| `n_layers_total` | `12` | dal `adata.uns['layer_embeddings']` se disponibile, altrimenti dal config models.yaml |
| `relative_depth` | `0.4545` | `layer / (n_layers_total - 1)`, NaN per la baseline |
| `metric` | `Accuracy` | nome della colonna metrica del CSV originale |
| `value` | `0.873` | valore numerico |

Questo è il file che il notebook `notebooks/paper_figures.ipynb` consuma.

## Layout dei risultati su disco

```
data/
├── embeddings/
│   ├── brain_dataset_scfoundation/         # output stage 1 (per-chunk h5ads)
│   │   ├── chunk_0000.h5ad
│   │   ├── chunk_0001.h5ad
│   │   └── brain_dataset_scfoundation.h5ad   # opzionale: chunks consolidati (output merge_chunks.py)
│   ├── brain_dataset_tahoe_1b/
│   └── ...
├── classification_results/
│   ├── brain_dataset_scfoundation_classification_results.csv
│   ├── brain_dataset_tahoe_1b_classification_results.csv
│   └── ...
├── pseudotime_results/
│   └── ...
├── perturbation_metrics/
│   └── ...
├── results_all.csv         # output di aggregate_results.py — il file canonico per il paper
└── run_manifest.csv        # storico delle run del driver
```

## Riproducibilità

Per garantire che la griglia sia riproducibile:

1. **Versionare `config/`** (committato nel repo).
2. **Salvare il manifest** delle run (`data/run_manifest.csv`), che cattura `git_sha`, `started_at`, `model_size`, `chunk_size`, `batch_size` per ogni run.
3. **Versionare i risultati aggregati** (`data/results_all.csv` può essere committato — è piccolo).
4. **NON versionare**: `data/raw/`, `data/embeddings/`, i CSV grossi per-dataset. Già gitignored.

## Hardware

Lo stage 1 è il collo di bottiglia. Indicazioni:

- scFoundation: VRAM ~12-16 GB con `batch_size=1`, `chunk_size=20000`.
- Tahoe-1b: VRAM ~24 GB con `batch_size=16`, `chunk_size=50000`. Per Tahoe-3b servono GPU >=40 GB.
- Time budget tipico per un dataset da 100k celle e un modello: 1-3 ore.

Lo stage 2 gira su CPU; il bottleneck è il computo del k-NN per pseudo-time e Leiden per ARI. Tipicamente <30 min per dataset.
