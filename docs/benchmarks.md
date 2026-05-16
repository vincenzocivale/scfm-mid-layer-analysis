# Benchmarks

I tre task downstream, le loro metriche, e cosa cercare nelle curve per-layer.

## 1. Cell-type classification

**Pacchetto**: `src/scfm_eval/benchmarks/classification/`
**Pipeline entry**: `scripts/run_cell_type_classification.py`
**Ipotesi testata**: gli embedding di un layer separano linearmente i tipi cellulari? E quanto bene clusterizzano in modo non supervisionato?

### Preprocessing (`preprocessing.py`)

- Filtra celle con `cell_type` NaN.
- Calcola **baseline PCA** su HVG da `adata.raw` (2500 HVG, 50 PC). Appare come `Baseline_PCA` nei risultati.
- Trasferisce tutti gli `X_layer_*` dall'adata input all'adata processato.

### Metriche (`evaluator.py`)

Tutte le probe lineari e kNN usano `StandardScaler` upstream, in modo che la regolarizzazione non venga sbilanciata dalle scale diverse tra layer profondi.

| Metrica | Implementazione | Interpretazione |
|---|---|---|
| `Accuracy`, `F1_macro`, `Precision_macro`, `Recall_macro` | `StandardScaler` + `LogisticRegression(solver='lbfgs', max_iter=1000)`, 5-fold StratifiedCV, `random_state=42` | Separabilità lineare |
| `kNN_Accuracy`, `kNN_F1_macro` | `StandardScaler` + `KNeighborsClassifier(n_neighbors=15)`, 5-fold StratifiedCV | Struttura locale dello spazio embedding |
| `Silhouette_Score` | `sklearn.metrics.silhouette_score`, `metric='euclidean'`, subsample seed-controlled a 20k cells | Coesione vs separazione cluster |
| `ARI`, `NMI`, `leiden_resolution`, `n_clusters` | **Leiden sweep** su `{0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0}` con `k=15`; riportato il best ARI e la risoluzione che l'ha raggiunto | Stabilità della partizione non supervisionata, robusta alla scelta della risoluzione |

### Output

`data/classification_results/<input_stem>_classification_results.csv` con una riga per layer (più la riga `Baseline_PCA`).

## 2. Pseudo-time

**Pacchetto**: `src/scfm_eval/benchmarks/pseudotime/`
**Pipeline entry**: `scripts/run_pseudotime_analysis.py`
**Ipotesi testata**: la geometria degli embedding di un layer cattura la progressione temporale?

### Preprocessing (`preprocessing.py`)

- Costruisce un **reference dataset** filtrando per i timepoint disponibili.
- Trova la **root cell**: cellula del timepoint più precoce (root index salvato in `adata.uns['iroot']`).
- Calcola il **baseline DPT** con scanpy.tl.dpt su HVG-PCA con `k=30` vicini. Nota: è una *baseline*, non un ground truth biologico; il ground truth reale è la colonna temporale (`time_col`).

### Metriche (`evaluator.py`)

Il grafo k-NN per layer viene calcolato, usato per tutte le metriche di quel layer, e poi rimosso (`obsp/uns/obsm` puliti) prima di passare al layer successivo, per evitare leak tra iterazioni.

| Metrica | Implementazione | Interpretazione |
|---|---|---|
| `Pseudotime_Corr_vs_Time` | Spearman(real_time_col, layer_DPT) — usa direttamente l'annotazione temporale biologica (week/day/stage) | Ordering reale catturato dall'embedding |
| `Pseudotime_Corr_vs_RefDPT` | Spearman(baseline_DPT_PCA-HVG, layer_DPT) | Quanto l'ordering replica la baseline non-FM |
| `Neighborhood_Overlap` | Media di `|N_ref(i) ∩ N_layer(i)| / k` con i k vicini **ordinati per distanza** (k=30) | Continuità locale dello spazio embedding |
| `Global_Geom_Corr` | Spearman(\|t_i - t_j\|, \|\|emb_i - emb_j\|\|) su 5000 coppie campionate con seed fisso | Coerenza tra distanza temporale e distanza in embedding |

> **Bug fix (2026-05-16)**: la precedente implementazione di `Neighborhood_Overlap` estraeva i primi k indici della riga sparsa di `obsp['distances']`, che sono in ordine di colonna anziché di distanza. I valori di `Neighborhood_Overlap` calcolati prima di questa data sono invalidi.

### Output

`data/pseudotime_results/<input_stem>_results.csv`. Include una riga per ogni `X_layer_*` più la riga `Baseline_PCA` (PCA-HVG riferimento).

## 3. Perturbation response

**Pacchetto**: `src/scfm_eval/benchmarks/perturbation/`
**Pipeline entry**: `scripts/run_perturbation_analysis.py`
**Ipotesi testata**: gli embedding di un layer catturano la similarità biologica tra perturbazioni geniche?

> **Caveat di leakage**: la reference DE è calcolata sugli stessi conteggi che il FM ha visto al momento dell'inferenza. Quindi `semantic_similarity` misura quanto l'embedding *preserva* la struttura DE derivabile dagli stessi conteggi, non capacità predittiva indipendente. Per uno zero-shot più stringente, fornire un reference esterno (CMap, MSigDB, ecc.) tramite la stessa interfaccia.

### Reference building (`reference.py`)

Per ogni perturbazione con ≥10 cellule (vs control), il modulo:
1. Esegue **`sc.tl.rank_genes_groups(method='wilcoxon')`** vs il gruppo control.
2. Costruisce un vettore log-fold-change su *tutti* i geni (no top-N + zero-fill — questo evitava bias delle versioni precedenti).
3. Calcola la *bio-reference similarity matrix*: cosine similarity sui rank dei profili (= Spearman), oppure Pearson sui profili centrati.

L'AnnData viene auto-normalizzato (`normalize_total` + `log1p`) se `uns['log1p']` non è già presente.

### Metriche (`evaluator.py`)

| Metrica | Implementazione | Interpretazione |
|---|---|---|
| `correlation` (per layer) | Spearman tra reference_sim_matrix e centroid_cosine_sim per layer | Allineamento biologico globale |
| `null_mean`, `null_std`, `z_score` | Null distribution via 100 label-shuffles della centroid matrix | Significatività della correlation vs random baseline |
| `p_value`, `n_common_perturbations` | p-value analitico di Spearman; numero di perturbazioni in comune con la reference | Trasparenza sulla power della stima |
| `mean_dose_correlation`, `std_dose_correlation`, `n_perturbations_with_dose` | Spearman(dose, cosine_distance(emb, control_centroid)), mediato sulle perturbazioni con dose numerica | Sensibilità monotona alla dose (solo se `--dose_key`). **Cosine distance**, non L2: invariante alla scala dei layer profondi |
| `pathway_gap`, `n_within`, `n_between` | mean(sim_within_pathway) − mean(sim_between_pathway) usando un dict pathway → genes | Quanto pathways correlati sono vicini in embedding (solo se `--pathway_json`) |

### Output

`data/perturbation_metrics/<input_stem>_*.csv` — un CSV per metrica (`semantic_similarity`, `dose_response`, `pathway_clustering`), più la matrice di riferimento `<input_stem>_reference_de_similarity.csv`.

> **Nota retro-compatibilità**: il parametro CLI `--de_top_n` è deprecato (la reference ora usa tutti i geni). Viene accettato senza effetto per non rompere script esistenti.

## Baseline

- **Classification**: PCA su HVG → riga `Baseline_PCA`.
- **Pseudo-time**: PCA-HVG-DPT è esposta come `Baseline_PCA` nei risultati a partire dalla v2.
- **Perturbation**: la reference DE è il ground truth biologico; per una baseline "no-FM" comparativa si può eseguire `run_perturbation_pipeline` su un AnnData con PCA copiata in `obsm['X_layer_0']`.

## Cosa cercare nei plot per-layer

Per la sezione *Results* del paper, ognuna di queste curve `metric(layer)` ha un'interpretazione tipica:

- **Monotonamente crescente fino all'ultimo layer**: il modello migliora la rappresentazione fino in fondo per quel task → ultimo layer è "best".
- **Picco intermedio**: i layer profondi *perdono* informazione specifica al task → fenomeno tipico nei language model, evidenza che l'ultimo layer è ottimizzato per il pretraining loss, non per il task downstream.
- **Curva piatta**: il task non discrimina tra layer, oppure il modello già conserva quella informazione nei primi blocchi.
- **Curva con maggior varianza in layer profondi**: instabilità di scala, vedi note in [models.md](models.md#convenzione-di-indicizzazione-layer-critica-per-il-paper). Con `StandardScaler` upstream nelle probe questo effetto è mitigato.

## Riproducibilità

Tutte le metriche stocastiche usano seed fissi (`random_state=42` / `np.random.default_rng(42)`):
- Classification: CV split, silhouette subsample, Leiden.
- Pseudotime: campionamento delle coppie in `Global_Geom_Corr`.
- Perturbation: permutations della null distribution.
