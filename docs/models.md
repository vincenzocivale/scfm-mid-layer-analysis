# Foundation models

Per ogni FM documentiamo: come carica i pesi, come prepara i dati, qual è la semantica di "layer i", come viene poolato il rappresentazione cellulare. Questo è il dato necessario per validare scientificamente i confronti cross-modello.

## Riepilogo

| Model | Sizes | `n_layers` | Pooling | Input richiesto | Stato |
|---|---|---|---|---|---|
| scFoundation | unica | 12 | S-token (`x[:, -1, :]`) | normalized+log1p, gene-symbol vocab 19264 | Implementato |
| Tahoe-X1 | 70m / 1b / 3b | 12 / 24 / 32* | CLS token (`x[:, 0, :]`) | raw counts, vocab Tahoe | Implementato |
| scGPT | unica | 12* | CLS token (`x[:, 0, :]`) | raw counts, vocab scGPT | Implementato |
| Geneformer | varia | TBD | mean pooling (tipico) | rank-ordered tokens | **Da implementare** |

*Il numero di layer di Tahoe va verificato a runtime con `python src/scfm_eval/extraction/model_info.py --model tahoe --tahoe_size <size>`.
**Il numero di layer di scGPT dipende dalla variante scaricata (whole_human = 12). Verificare a runtime.

## scFoundation

- **Paper**: Hao et al., *Large Scale Foundation Model on Single-cell Transcriptomics*, BioMap.
- **Weights**: `data/checkpoints/scfoundation/models.ckpt` (~1.4 GB, gitignored). Path overridabile via `SCFOUNDATION_CKPT` env var.
- **Gene index**: `data/checkpoints/scfoundation/OS_scRNA_gene_index.19264.tsv` (vocab di 19264 gene symbol). Override via `SCFOUNDATION_GENE_INDEX`.
- **Preprocessing** (`prepare_data`):
  1. Mappa `adata.var_names` (Ensembl) → gene symbol via `adata.var['feature_name']`.
  2. Zero-padding per geni mancanti dal vocab.
  3. Calcola/copia `obs['log_total_count']` (richiesto come token aggiuntivo dal modello).
- **Forward pass**: aggiunge due token speciali alla sequenza: `[4.0, log_total_count]`. Il modello processa una sequenza di geni non-zero (gather) + questi due. L'ultima posizione (S-token) è la rappresentazione pooled della cellula.
- **Layer hook**: implementato inline (non via PyTorch hook), iterando `self.model.encoder.transformer_encoder` e raccogliendo `x[:, -1, :]` dopo ogni blocco se `idx ∈ layer_indices`.
- **n_layers_total**: `len(model.encoder.transformer_encoder)` = 12.

## Tahoe-X1

- **Weights**: scaricati da Hugging Face (`tahoebio/tahoe-x1`) al primo init. Cache locale via `transformers`.
- **Preprocessing**: filtra `adata.var_names` ai geni presenti nel vocab Tahoe (`self.vocab`). Le celle restano invariate.
- **Forward pass**: usa `loader_from_adata` di `tahoe_x1.utils.util`. Inietta un token `<cls>` all'inizio della sequenza.
- **Layer hook**: `register_forward_hook` su `model.model.transformer_encoder.layers[i]`. Estrae `output[:, 0, :]` (CLS).
- **n_layers_total**: `len(model.model.transformer_encoder.layers)`.

## scGPT

- **Paper**: Cui et al., *scGPT: toward building a foundation model for single-cell multi-omics using generative AI*, Nature Methods 2024.
- **Repo**: `scGPT/` (vendored nella repo; installabile con `pip install --no-deps -e scGPT/`).
- **Weights**: `data/checkpoints/scgpt/whole_human/` — richiede download manuale (o altra variante pretrained). Il path è overridabile via `SCGPT_MODEL_DIR` env var. La directory deve contenere: `best_model.pt`, `args.json`, `vocab.json`.
- **Environment dedicato**: usare `conda activate scgpt-env` (creato da `envs/scgpt.yml`). scGPT richiede dipendenze incompatibili con il `.venv` principale (scanpy<2, datasets, networkx). Non usare flash-attn (`use_fast_transformer=False` è hardcoded nell'embedder).
- **Preprocessing** (`prepare_data`):
  1. Usa `adata.var['feature_name']` se presente, altrimenti `adata.var_names`.
  2. Filtra ai geni presenti nel vocab scGPT (`vocab.json` nella model dir).
  3. Aggiunge colonna `id_in_vocab` a `adata.var`.
- **Forward pass**: tokenizza (gene_id + espressione binned), inietta un token `<cls>` in posizione 0, processa con `TransformerModel._encode()`.
- **Layer hook**: `register_forward_hook` su `model.transformer_encoder.layers[i]`. Estrae `output[:, 0, :]` (CLS token). Nota: PyTorch 2.1+ usa NestedTensor internamente nella `TransformerEncoder.forward()` — l'hook chiama `.to_padded_tensor(0.0)` prima di indicizzare.
- **n_layers_total**: `len(model.transformer_encoder.layers)` — dipende dalla variante (whole_human = 12).

## Geneformer (placeholder, non implementato)

Geneformer ha un input semantico differente: tokens rank-ordered per espressione (non count-based). Questo richiede una `prepare_data` che produce sequenze diverse, non un X di counts/log-counts.
- Pooling tipico: mean pooling dell'ultima sequence (escludendo padding).
- n_layers: 6 (Geneformer-12L) o 12 (Geneformer-12L-95M). Verificare la variante usata.

## Convenzione di indicizzazione layer (CRITICA per il paper)

**`X_layer_i` = output dell'i-esimo blocco transformer, pre-norma finale, dopo pooling.**

Questo significa:
- `X_layer_0` NON è l'output dell'embedding layer (token embedding + pos embedding). È l'output del **primo blocco transformer** (post self-attention + FFN).
- L'output finale del modello (l'ultimo blocco) è `X_layer_{n-1}`.

Per il paper, riportare la profondità come **`relative_depth = i / (n_layers_total - 1)`** in modo che 0.0 = primo blocco e 1.0 = ultimo blocco. Questo permette il confronto onesto tra scFoundation (12 layer) e Tahoe-1b (24 layer).

### Limitazioni della convenzione

- **Non cattura l'embedding layer**: se vuoi includere "layer pre-transformer" come baseline, va aggiunto a parte (e.g. `X_embedding`). Oggi non implementato.
- **Pooling diverso tra FM**: il vettore confrontato per `X_layer_i` tra Tahoe (CLS) e scFoundation (S-token) è semanticamente diverso, ma è la rappresentazione cellulare ufficiale di ciascun modello. È il confronto giusto da fare; va dichiarato esplicitamente nel paper.
- **Non normalizziamo la norma del vettore tra layer/modelli**: layer profondi hanno tipicamente magnitudo diversa. Le metriche downstream (logistic regression, cosine similarity, silhouette) sono parzialmente scale-invarianti, ma worth a sanity check in EDA.
