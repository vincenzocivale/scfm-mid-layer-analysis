# Adding a new foundation model

Checklist per integrare un nuovo FM nel framework. Sostituisci `<myfm>` con il nome canonico del modello (es. `geneformer`, `scgpt`).

## 1. Implementa l'embedder

Crea `src/scfm_eval/embedders/<myfm>.py` che estende `BaseEmbedder`. Interfaccia minima:

```python
from scfm_eval.embedders.base import BaseEmbedder

class MyFMEmbedder(BaseEmbedder):
    # Class-level metadata (read by the extraction script for adata.uns)
    pooling = 'cls_token'   # one of 'cls_token' | 's_token' | 'mean' | ...

    def __init__(self, device='auto', fp16=True, **kwargs):
        super().__init__('myfm', device)
        self.fp16 = fp16
        self.load_model()

    def load_model(self):
        # Load weights into self.model, move to self.device with appropriate dtype
        ...

    def prepare_data(self, adata):
        # Return an adata that the model can consume. Document what's required:
        # - normalized vs raw counts
        # - which gene vocabulary
        # - any required obs columns
        ...

    def extract_embeddings_for_layers(self, adata, layer_indices, batch_size=4):
        # Return {layer_idx: np.ndarray of shape (n_cells, hidden_dim)}
        # Use hooks where possible to get ALL layers in ONE forward pass.
        ...

    def get_all_layer_indices(self):
        return list(range(len(self.model.<transformer_block_list>)))

    @property
    def hidden_dim(self):
        return self.model.<config>.hidden_size
```

### Cosa rispettare

- **Convenzione di indicizzazione**: `layer_i` = output dell'i-esimo blocco transformer (vedi [models.md](models.md)). Se il modello ha un layer di embedding separato che vuoi includere, dichiaralo esplicitamente (es. `layer_-1`) e documentalo qui.
- **Una forward pass per chunk**: non ri-fare inferenza una volta per layer. Usa hook o early-return loop sugli moduli.
- **Output sempre `(n_cells, hidden_dim)`**: applica il pooling che il modello usa nativamente (CLS, mean, S-token). Documentalo nel campo `pooling`.
- **fp16/fp32**: rispetta `self.fp16`. Cast del modello e dei tensori coerentemente.

## 2. Registra l'embedder nel dispatcher

In **due file**:

- `src/scfm_eval/extraction/model_info.py`: aggiungi un branch nell'`if/elif` di `get_model_num_layers`.
- `src/scfm_eval/extraction/chunked.py`: aggiungi un branch nell'`if/elif` che istanzia l'embedder (intorno alla riga 60-75).

## 3. Aggiungi il modello al config

In `config/models.yaml`:

```yaml
myfm:
  class: models.myfm_embedder.MyFMEmbedder
  sizes: [base, large]          # se applicabile, altrimenti [default]
  expected_input: raw_counts    # 'raw_counts' | 'log1p' | 'rank_tokens'
  pooling: cls_token
  env_vars:                     # se servono path di weight overridabili
    MYFM_CKPT: /path/to/weights
```

Il numero di layer è scoperto a runtime da `get_model_info.py`, non va hardcoded.

## 4. Documenta in `docs/models.md`

Aggiungi una sezione che descrive:
- Da dove vengono i pesi (HF hub / file locale / download script).
- Cosa richiede `prepare_data` (normalized? raw counts? rank-ordered tokens?).
- Quale token viene usato come pooling e perché.
- Numero di layer per ogni size variante.

## 5. Aggiungi una riga di test alla griglia

In `config/experiments.yaml`, aggiungi una run su un dataset piccolo per validare end-to-end:

```yaml
- dataset: brain
  model: myfm
  tasks: [classification]
```

Poi:

```bash
./.venv/bin/python scripts/run_experiment_grid.py \
    --config config/experiments.yaml \
    --filter model=myfm
```

## 6. Sanity check

Prima di considerare il modello "integrato":

- [ ] Gli embedding non contengono NaN (controllo già presente nei due embedder esistenti).
- [ ] La shape è coerente: `n_cells × hidden_dim` per ogni layer richiesto.
- [ ] `adata.uns['layer_embeddings']` viene popolato con `n_layers_total`, `hidden_dim`, `pooling`.
- [ ] L'aggregator (`scripts/aggregate_results.py`) riconosce il nome del modello dal CSV (potresti dover aggiornare la regex di parsing — vedi commenti in quel file).
- [ ] La classification baseline gira: confronta `X_layer_{n-1}` con `Baseline_PCA` per il sanity check "ultimo layer batte la baseline?".
