# Documentation

Questo repository implementa il framework di valutazione per il paper sulla **rappresentazione biologica nei layer intermedi dei foundation models per single-cell**. La domanda di ricerca: *come variano le proprietà biologicamente rilevanti degli embedding lungo la profondità della rete*, confrontata across foundation models e across task downstream zero-shot.

## Indice della documentazione

| File | Contenuto |
|---|---|
| [architecture.md](architecture.md) | Architettura del sistema, contratto `X_layer_*`, workflow due-stage |
| [models.md](models.md) | Specifiche per ogni foundation model: numero di layer, semantica dell'indice di layer, pooling, preprocessing richiesto |
| [datasets.md](datasets.md) | Dataset usati, sorgenti CellxGene/GEO, task supportati e colonne `.obs` richieste |
| [benchmarks.md](benchmarks.md) | Metodologia di ogni task downstream (classification, pseudo-time, perturbation), metriche, baseline |
| [experiments.md](experiments.md) | Come eseguire la griglia (dataset × FM × task), dove finiscono i risultati, come aggregare |
| [adding_a_model.md](adding_a_model.md) | Checklist per integrare un nuovo foundation model |
| [adding_a_benchmark.md](adding_a_benchmark.md) | Checklist per aggiungere un nuovo task downstream |

## Quick reference

- Tutti gli embedding finiscono in `adata.obsm[X_layer_{i}]` con metadati in `adata.uns[layer_embeddings]`. Vedi [architecture.md](architecture.md#contratto-x_layer_).
- La griglia di esperimenti è dichiarata in [config/experiments.yaml](../config/experiments.yaml). Il driver è [scripts/run_experiment_grid.py](../scripts/run_experiment_grid.py).
- I risultati finiscono in `data/<task>_results/` come CSV; lo script `scripts/aggregate_results.py` produce un singolo `data/results_all.csv` long-format per le analisi finali.
- Il notebook `notebooks/paper_figures.ipynb` legge `results_all.csv` e genera le figure del paper.
