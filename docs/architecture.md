# Architecture

Il sistema è organizzato in **due stage disaccoppiati** che comunicano via h5ad su disco. Questa scelta permette di iterare sulle metriche di valutazione senza ri-eseguire l'inferenza dei foundation models (che è la parte computazionalmente costosa).

```
                   STAGE 1                              STAGE 2
                   ───────                              ───────
   raw h5ad   ─►  Embedder.extract     ─►   embedded h5ad   ─►  pipeline.run_*()  ─►  CSV
              ─►  (chunked, all layers)      X_layer_{i}        (per-task evaluator)    metrics
                                             in adata.obsm
```

## Contratto `X_layer_*`

Il punto di sincronizzazione tra stage 1 e stage 2 è la chiave `X_layer_{i}` in `adata.obsm`. **Non rinominare questa convenzione**: tutte le pipeline downstream auto-scoprono i layer da valutare con:

```python
layers_to_test = [k for k in adata.obsm.keys() if k.startswith('X_layer')]
```

Inoltre lo stage 1 scrive `adata.uns['layer_embeddings']`:

```python
{
    'model': 'tahoe',                  # nome canonico del modello
    'model_size': '1b',                # None se non applicabile
    'n_layers_total': 24,              # numero totale di layer del modello
    'layers_extracted': [0, 1, ..., 23],
    'hidden_dim': 1024,
    'pooling': 'cls_token',            # 'cls_token' | 's_token' | 'mean'
    'fp16': True,
    'genes_matched': 18432,            # geni del vocab matchati nel dataset
    'cell_range': [0, 100000],         # (chunked output) range delle celle
    'chunk_index': 0,
}
```

Questi metadati sono essenziali per:
- **Normalizzare la profondità tra modelli**: `relative_depth = i / (n_layers_total - 1)` permette di confrontare modelli con numero di layer diverso (scFoundation 12, Tahoe-1b 24, scGPT 12).
- **Annotare le tabelle del paper** con la configurazione esatta.
- **L'aggregator** (`scripts/aggregate_results.py`) le legge per costruire il long-format `results_all.csv`.

## Stage 1: embedding extraction

`src/scfm_eval/extraction/chunked.py` è il driver. Workflow:

1. Apre il dataset in **backed mode** (no full load in RAM).
2. Istanzia l'embedder una volta sola (model load in GPU costoso).
3. Itera a chunk: `[0:CHUNK_SIZE]`, `[CHUNK_SIZE:2*CHUNK_SIZE]`, …
4. Per ogni chunk: `embedder.prepare_data(chunk)` → `embedder.extract_embeddings_for_layers(chunk, all_layers, batch_size)` (un singolo forward pass per chunk, tutti i layer estratti via hook).
5. Scrive un h5ad per chunk in `data/embeddings/<input>_<model>[_<size>]/chunk_{idx:04d}.h5ad`.
6. Idempotente: chunk esistenti vengono skippati (resume gratuito).

Per ottenere un singolo h5ad consolidato: `python scripts/merge_chunks.py --input-dir data/embeddings/brain_dataset_scfoundation/`.

### Hook semantics e pooling

Tutti gli embedder restituiscono **un vettore per cellula per layer**: `dict[int, np.ndarray]` con shape `(n_cells, hidden_dim)`. Il modo in cui la sequenza di token viene poolata in un singolo vettore dipende dal modello:

| Model | Hook target | Pooling | Note |
|---|---|---|---|
| scFoundation | post-blocco `encoder.transformer_encoder[i]` | last token (S-token) | l'ultimo token è una somma totale di espressione, usata dal modello come embedding cellulare |
| Tahoe-X1 | post-blocco `transformer_encoder.layers[i]` | CLS token (pos 0) | il modello inietta un CLS token come prima posizione |

**Convenzione di indicizzazione**: `layer_i = output dell'i-esimo blocco transformer`. NON include l'output del solo embedding layer (per il quale servirebbe `layer_-1` o simile, non implementato).

## Stage 2: per-task evaluation

Ogni package benchmark (`src/scfm_eval/benchmarks/classification/`, `src/scfm_eval/benchmarks/pseudotime/`, `src/scfm_eval/benchmarks/perturbation/`) segue lo stesso layout di tre file:

| File | Ruolo |
|---|---|
| `preprocessing.py` | Filtra/prepara `adata` per il task (e.g. rimuove celle con label NaN, costruisce baseline PCA su HVG) |
| `evaluator.py` | Implementa `evaluate_layer(layer_key)` che restituisce un `dict` di metriche per quel layer |
| `pipeline.py` | Espone `run_*_pipeline(h5ad_path, …)`: load → preprocess → evaluator loop su tutti i `X_layer_*` → CSV |

Vedi [benchmarks.md](benchmarks.md) per metodologia e metriche di ogni task.

## Layout su disco

```
config/
  models.yaml              ← spec dei FM (n_layers, pooling, env vars)
  datasets.yaml            ← per-dataset: path, task supportati, colonne obs
  experiments.yaml         ← griglia (dataset × FM × task) da eseguire

src/scfm_eval/             ← TUTTA la library code
  embedders/
    base.py                ← BaseEmbedder ABC
    registry.py            ← name → class (lazy import)
    scfoundation.py
    tahoe.py
    vendor/scfoundation/   ← codice vendored upstream, isolato
  benchmarks/
    classification/  pseudotime/  perturbation/   ← preprocessing + evaluator + pipeline
  extraction/
    chunked.py             ← stage 1: estrazione chunk-by-chunk all-layer
    model_info.py          ← helper per ispezionare n_layers
  io/
    embeddings.py          ← write/read embedded h5ads (schema X_layer_* + uns metadata)
    results.py             ← path/writer canonici per i CSV di output

scripts/                   ← CLI entry points (argparse → chiamano scfm_eval)
  run_cell_type_classification.py
  run_perturbation_analysis.py
  run_pseudotime_analysis.py
  merge_chunks.py          ← consolida i chunk in un singolo h5ad
  aggregate_results.py     ← produce data/results_all.csv long-format
  run_experiment_grid.py   ← driver: legge experiments.yaml, dispatcha tutto

run_embedding_extraction.sh   ← orchestratore stage 1 (loopa input × modelli)

notebooks/
  paper_figures.ipynb      ← plot finali per il paper
  eda*.ipynb               ← prototipi

data/                      ← gitignored
  raw/                     ← input h5ad
  checkpoints/scfoundation/← models.ckpt + gene-index TSV
  embeddings/              ← output stage 1
  classification_results/  pseudotime_results/  perturbation_metrics/  ← CSV per (dataset, model, task)
  results_all.csv          ← output dell'aggregator
  run_manifest.csv         ← output del driver
```

## Design rationale

- **Perché un h5ad invece di un tensore puro?** L'h5ad mantiene allineate le metadati delle celle (`.obs`) con gli embedding. Le pipeline downstream hanno bisogno di queste annotazioni (cell_type, perturbation, time) per costruire ground truth.
- **Perché un chunk produce un h5ad e non un singolo file finale?** I dataset reali sono troppo grossi per stare in RAM completi quando moltiplicati per N layer × hidden_dim. Il chunking permette di processare 1M cell × 24 layer × 1024 dim senza OOM. L'unione è opzionale e separata.
- **Perché subprocess nel driver invece di import diretto?** Garantisce GPU memory cleanup pulito tra (model, dataset) diversi. L'overhead di startup è trascurabile rispetto al tempo di inferenza.
