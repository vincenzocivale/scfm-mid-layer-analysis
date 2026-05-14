# Benchmarks

I tre task downstream, le loro metriche, e cosa cercare nelle curve per-layer.

## 1. Cell-type classification

**Pacchetto**: `src/scfm_eval/benchmarks/classification/`
**Pipeline entry**: `scripts/run_cell_type_classification.py`
**Ipotesi testata**: gli embedding di un layer separano linearmente i tipi cellulari? E quanto?

### Preprocessing (`preprocessing.py`)

- Filtra celle con `cell_type` NaN.
- Calcola **baseline PCA** su HVG da `adata.raw` (2500 HVG, 50 PC). Appare come `Baseline_PCA` nei risultati.
- Trasferisce tutti gli `X_layer_*` dall'adata input all'adata processato.

### Metriche (`evaluator.py`)

| Metrica | Implementazione | Interpretazione |
|---|---|---|
| `Accuracy` | LogisticRegression `liblinear` + 5-fold StratifiedCV | Quanto è linearmente separabile |
| `F1_macro` | come sopra | Performance bilanciata tra classi |
| `Precision_macro`, `Recall_macro` | come sopra | |
| `Silhouette_Score` | `sklearn.metrics.silhouette_score`, metric=`euclidean`, sample a 20k cells | Coesione vs separazione cluster nello spazio embedding |
| `ARI` | Leiden clustering (`k=15`, `resolution=1.0`) vs ground truth | Stabilità della partizione non supervisionata |

### Output

`data/classification_results/<input_stem>_classification_results.csv` con una riga per layer (più la riga `Baseline_PCA`).

## 2. Pseudo-time

**Pacchetto**: `src/scfm_eval/benchmarks/pseudotime/`
**Pipeline entry**: `scripts/run_pseudotime_analysis.py`
**Ipotesi testata**: la geometria degli embedding di un layer cattura la progressione temporale?

### Preprocessing (`preprocessing.py`)

- Costruisce un **reference dataset** filtrando per i timepoint disponibili.
- Trova la **root cell**: cellula del timepoint più precoce (root index salvato in `adata.uns['iroot']`).
- Calcola il **ground truth pseudotime** con DPT (Diffusion Pseudotime) di scanpy su HVG-PCA con `k=30` vicini.

### Metriche (`evaluator.py`)

| Metrica | Implementazione | Interpretazione |
|---|---|---|
| `Pseudotime_Corr` | Spearman(ref_DPT, layer_DPT) — entrambi calcolati con scanpy.tl.dpt sul rispettivo grafo k-NN | Quanto l'ordering temporale è preservato |
| `Neighborhood_Overlap` | Media di `|N_ref(i) ∩ N_layer(i)| / k` su tutte le cellule (k=15) | Continuità locale dello spazio embedding |
| `Global_Geom_Corr` | Spearman(\|t_i - t_j\|, \|\|emb_i - emb_j\|\|) su 5000 coppie random | Coerenza tra distanza temporale e distanza in embedding |

### Output

`data/pseudotime_results/<input_stem>_results.csv`.

## 3. Perturbation response

**Pacchetto**: `src/scfm_eval/benchmarks/perturbation/`
**Pipeline entry**: `scripts/run_perturbation_analysis.py`
**Ipotesi testata**: gli embedding di un layer catturano la similarità biologica tra perturbazioni geniche?

### Reference building (`reference.py`)

Per ogni perturbazione (vs control), il modulo:
1. Identifica i top-N DE genes (default N=500) via Wilcoxon o t-test su `adata.raw`.
2. Costruisce una *bio-reference similarity matrix*: ogni cella `(p1, p2)` è la correlazione Spearman (o Pearson) tra i ranghi DE delle due perturbazioni.

Questa matrice è il "ground truth biologico" contro cui valutare la similarità degli embedding.

### Metriche (`evaluator.py`)

| Metrica | Implementazione | Interpretazione |
|---|---|---|
| `semantic_similarity` | Spearman tra reference_sim_matrix e centroid_cosine_sim (centroid per perturbazione nello spazio embedding del layer) | Allineamento biologico globale |
| `dose_response` | Spearman(dose, \|\|emb - control_centroid\|\|), mediato sulle perturbazioni con dose numerica | Sensibilità monotona alla dose (solo se `--dose_key`) |
| `pathway_gap` | mean(sim_within_pathway) − mean(sim_between_pathway) usando un dict pathway → genes | Quanto pathways correlati sono vicini in embedding (solo se `--pathway_json`) |

### Output

`data/perturbation_metrics/<input_stem>_*.csv` — più CSV separati per metrica (`semantic_similarity`, `dose_response`, `pathway_clustering`), più la matrice di riferimento DE `<input_stem>_reference_de_similarity.csv`.

## Baseline

Solo il task di classification ha una baseline automatica (PCA su HVG). Per gli altri due:

- **Pseudo-time**: la ground-truth DPT è già un riferimento "non-FM". Si potrebbe aggiungere come baseline esplicita la PCA-DPT come riga `Baseline_PCA` per essere consistenti.
- **Perturbation**: la reference DE similarity matrix è già il ground truth biologico. Si potrebbe aggiungere come baseline la similarità basata su X grezza (no FM) o su PCA come comparativa.

Quando si aggiungeranno queste baseline esplicite, riportarle anche nella tabella riepilogativa del paper.

## Cosa cercare nei plot per-layer

Per la sezione *Results* del paper, ognuna di queste curve `metric(layer)` ha un'interpretazione tipica:

- **Monotonamente crescente fino all'ultimo layer**: il modello migliora la rappresentazione fino in fondo per quel task → ultimo layer è "best".
- **Picco intermedio**: i layer profondi *perdono* informazione specifica al task → fenomeno tipico nei language model (anche scBERT?), evidenza che l'ultimo layer è ottimizzato per il pretraining loss, non per il task downstream.
- **Curva piatta**: il task non discrimina tra layer, oppure il modello già conserva quella informazione nei primi blocchi.
- **Curva con maggior varianza in layer profondi**: instabilità di scala, vedi note in [models.md](models.md#convenzione-di-indicizzazione-layer-critica-per-il-paper).
