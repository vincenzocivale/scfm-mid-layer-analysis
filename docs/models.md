# Foundation models

Per ogni FM documentiamo: come carica i pesi, come prepara i dati, qual è la semantica di "layer i", come viene poolato il rappresentazione cellulare. Questo è il dato necessario per validare scientificamente i confronti cross-modello.

## Riepilogo

| Model | Sizes | `n_layers` | Pooling | Input richiesto | Stato |
|---|---|---|---|---|---|
| scFoundation | unica | 12 | S-token (`x[:, -1, :]`) | normalized+log1p, gene-symbol vocab 19264 | Implementato |
| Tahoe-X1 | 70m / 1b / 3b | 12 / 24 / 32* | CLS token (`x[:, 0, :]`) | raw counts, vocab Tahoe | Implementato |
| scGPT | unica | 12** | CLS token (`x[:, 0, :]`) | raw counts, vocab scGPT | Implementato |
| CellFM | unica | 40 | mean non-zero genes | raw counts, CellFM gene vocab | Implementato |
| UCE | 4layer / 33layer | 4 / 33 | CLS token (`x[0, :, :]`, seq-first) | raw counts, ESM2 protein vocab | Implementato |
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
- **Repo**: https://github.com/bowang-lab/scGPT (installabile con `pip install --no-deps git+https://github.com/bowang-lab/scGPT.git`).
- **Weights**: `data/checkpoints/scgpt/whole_human/` — richiede download manuale (o altra variante pretrained). Il path è overridabile via `SCGPT_MODEL_DIR` env var. La directory deve contenere: `best_model.pt`, `args.json`, `vocab.json`.
- **Environment dedicato**: usare `conda activate scgpt-env` (creato da `envs/scgpt.yml`). scGPT richiede dipendenze incompatibili con il `.venv` principale (scanpy<2, datasets, networkx). Non usare flash-attn (`use_fast_transformer=False` è hardcoded nell'embedder).
- **Preprocessing** (`prepare_data`):
  1. Usa `adata.var['feature_name']` se presente, altrimenti `adata.var_names`.
  2. Filtra ai geni presenti nel vocab scGPT (`vocab.json` nella model dir).
  3. Aggiunge colonna `id_in_vocab` a `adata.var`.
- **Forward pass**: tokenizza (gene_id + espressione binned), inietta un token `<cls>` in posizione 0, processa con `TransformerModel._encode()`.
- **Layer hook**: `register_forward_hook` su `model.transformer_encoder.layers[i]`. Estrae `output[:, 0, :]` (CLS token). Nota: PyTorch 2.1+ usa NestedTensor internamente nella `TransformerEncoder.forward()` — l'hook chiama `.to_padded_tensor(0.0)` prima di indicizzare.
- **n_layers_total**: `len(model.transformer_encoder.layers)` — dipende dalla variante (whole_human = 12).

## CellFM

- **Paper**: Zeng et al., *CellFM: a large-scale foundation model pre-trained on transcriptomics of 100 million human cells*, 2024.
- **Framework**: **MindSpore** (not PyTorch). Requires `conda activate cellfm-env` (see `envs/cellfm-env.yml`).
- **Weights**: `data/checkpoints/cellfm/base_weight.ckpt` — download from [HuggingFace ShangguanNingyuan/CellFM](https://huggingface.co/ShangguanNingyuan/CellFM/tree/main). Override via `CELLFM_CKPT` env var.
- **Gene vocabulary**: `CellFM/csv/gene_info.csv` (~19 k genes, gene symbol). Override via `CELLFM_GENE_INFO` not exposed yet; path is resolved relative to the `CellFM/` subdir.
- **Architecture**: `Encoder` class — 40 `RetentionLayer` blocks (Multi-Head Retention, not standard attention), hidden_dim=1536, num_heads=48. No CLS token (unlike scGPT/Tahoe).
- **Preprocessing** (`prepare_data`):
  1. Maps `adata.var_names` to CellFM vocab indices via `gene_info.csv`.
  2. Unmatched genes receive index 0 (zeroed embedding, effectively masked).
  3. Updates `encoder.used_gene` in-place — no weight reload needed.
- **Forward pass**: PYNATIVE_MODE (eager execution). The initial embedding sums value-encoder output and gene embedding. Then 40 RetentionLayer blocks iterate sequentially, each applying Multi-Head Retention + GatedLinearUnit FFN with SRMSNorm.
- **Layer hook**: implemented inline — manual iteration of `enc.encoder[i]()` in PYNATIVE_MODE. Output collected at requested layer indices.
- **Pooling**: mean over non-zero expression positions → `(b, 1536)` cell vector. No CLS token exists in `Encoder`.
- **n_layers_total**: 40 (hardcoded; verified from `config.py` `enc_nlayers=40`).
- **Nota**: un PyTorch version (CellFM-torch) è disponibile su GitHub ma non è stata ancora integrata. Se si vuole evitare MindSpore, quella è la strada.

## UCE (Universal Cell Embeddings)

- **Paper**: Rosen et al., *Universal Cell Embeddings: A Foundation Model for Cell Biology*, bioRxiv 2023.
- **Source repo**: https://github.com/snap-stanford/UCE (MIT License; code vendored in `src/scfm_eval/embedders/vendor/uce/`).
- **Weights**: auto-downloaded from figshare on first use to `data/checkpoints/uce/`.  Override via `UCE_MODEL_CKPT` and `UCE_MODEL_FILES_DIR` env vars.
- **Environment dedicato**: usare `conda activate uce-env` (creato da `envs/uce.yml`).
- **Architecture**: 4-layer Transformer (pretrained default; 33-layer variant also available).
  - Input token dim: 5120 (ESM2 protein embeddings).
  - d_model: 1280, nhead: 20, d_hid: 5120, output_dim: 1280.
  - Total learnable parameters: ~90M (4-layer).
- **Input semantics** (`prepare_data`):
  1. Resolves gene symbols from `adata.var['feature_name']` if present, otherwise `adata.var_names`. Matching is case-insensitive.
  2. Filters to genes that have ESM2 protein embeddings for the target species (default: `human`).
  3. Further filters to genes present in the chromosome position file.
  4. Computes per-gene: token index in UCE vocabulary (`pe_row_idxs`), chromosome code, genomic start position.
  5. `adata.X` must contain **raw (un-normalised) counts** — gene sampling is weighted by `log1p(counts)`.
- **Cell sentence construction** (`_make_cell_sentences`):
  - Samples `sample_size=1024` genes per cell, weighted by `log1p` expression.
  - Orders sampled genes by chromosome (shuffled), then by genomic start within each chromosome.
  - Sequence: `[CLS, chrom_open, gene, gene, ..., chrom_close, chrom_open, ...] + PAD` up to length 1536.
  - Gene tokens are ESM2 protein embeddings (5120-dim), L2-normalised before input to the model.
  - Sampling is stochastic; a fixed numpy seed (42) is set during `extract_embeddings_for_layers` for reproducibility.
- **Forward pass**: `TransformerModel.forward(src, mask)` where `src` is sequence-first `[seq_len, batch, token_dim]`.
- **Layer hook**: `register_forward_hook` on `model.transformer_encoder.layers[i]`.  Extracts `output[0, :, :]` (CLS token, position 0, seq-first) → shape `(batch, 1280)`.
- **Pooling**: CLS token at position 0.  Pre-decoder output (unlike the published UCE embedding which applies an additional MLP decoder); consistent with the `X_layer_i` convention in this framework.
- **n_layers_total**: 4 (default pretrained model) or 33 (larger variant).  Verified at runtime via `len(model.transformer_encoder.layers)`.

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
