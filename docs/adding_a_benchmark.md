# Adding a new downstream benchmark

Checklist per aggiungere un nuovo task di valutazione. Esempio: gene perturbation prediction (predire l'espressione genica post-perturbazione), batch correction, drug response.

## 1. Crea il package

```
<task_name>_benchmark/
├── __init__.py
├── preprocessing.py
├── evaluator.py
└── pipeline.py
```

Convenzione di naming: nome breve descrittivo + suffisso `_benchmark` (o omettilo, vedi `src/scfm_eval/benchmarks/perturbation/` che è un'eccezione storica).

## 2. Implementa i tre file

### `preprocessing.py`

Una funzione `prepare_<task>_dataset(adata, **task_specific_kwargs) → adata_proc, *extras`. Deve:
- Filtrare le celle inutilizzabili (label NaN, gruppo con N<soglia, etc.).
- Calcolare ground truth/baseline se applicabile.
- **Preservare tutti gli `X_layer_*` in obsm** del'adata input.

### `evaluator.py`

Una classe `<Task>Evaluator(adata, **config)` con un metodo:
```python
def evaluate_layer(self, layer_key: str) -> dict | None:
    # Return {metric_name: value} or None if the layer is unusable (e.g. all NaN).
```

Esegue una singola valutazione: dato un embedding di un layer, computa le metriche e restituisce il dict. La pipeline si occuperà del loop su tutti i layer.

### `pipeline.py`

```python
def run_<task>_pipeline(h5ad_path, embedding_prefix='X_layer', output_path=None, **kwargs):
    adata = sc.read_h5ad(h5ad_path)
    adata_proc, *extras = prepare_<task>_dataset(adata, **kwargs)
    evaluator = <Task>Evaluator(adata_proc, **kwargs)

    layers = [k for k in adata_proc.obsm.keys() if k.startswith(embedding_prefix)]

    results = []
    for k in layers:
        r = evaluator.evaluate_layer(k)
        if r is not None:
            r['Layer'] = k
            results.append(r)

    df = pd.DataFrame(results)
    if output_path:
        df.to_csv(output_path, index=False)
    return df
```

## 3. Aggiungi il launcher CLI

`scripts/run_<task>_analysis.py`: argparse + chiamata a `run_<task>_pipeline`. Modello: copia uno degli esistenti (`scripts/run_cell_type_classification.py` è il più semplice).

Aggancia `sys.path` al root del progetto:
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

## 4. Registra il task in `config/`

In `config/datasets.yaml`, per ogni dataset compatibile, aggiungi `<task>` alla lista `tasks` e dichiara la config necessaria:

```yaml
brain:
  path: data/raw/brain_dataset.h5ad
  tasks:
    classification:
      cell_type_column: cell_type
    <new_task>:
      <param_name>: <value>
```

In `config/experiments.yaml`, aggiungi run che includono il nuovo task.

## 5. Estendi il driver

`scripts/run_experiment_grid.py` deve sapere come dispatcha il nuovo task. Aggiungi un branch alla mappa `TASK_DISPATCH` (o equivalente) che lega il task name al launcher script + la lista degli argomenti CLI attesi.

## 6. Estendi l'aggregator

`scripts/aggregate_results.py` deve parsare il/i CSV prodotti dal nuovo task. Tipicamente questo significa:
- Aggiungere il task al pattern di scoperta dei file (`data/<task>_results/*.csv` o equivalente).
- Verificare che le colonne metrica siano riconosciute (oppure scegliere un naming di output coerente con gli altri task).

## 7. Documenta in `docs/benchmarks.md`

Aggiungi una sezione che descrive:
- Ipotesi testata.
- Preprocessing.
- Tabella metriche (nome, implementazione, interpretazione).
- Baseline disponibile (se non c'è, dichiaralo come limitazione).

## 8. Sanity check

- [ ] Il pipeline gira end-to-end su un dataset piccolo.
- [ ] Le metriche hanno il segno atteso (più alto = migliore, in caso contrario documentalo).
- [ ] Almeno una riga `Baseline_*` è inclusa nei risultati (o l'assenza è dichiarata).
- [ ] L'aggregator riconosce il task nel `results_all.csv`.
