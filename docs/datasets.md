# Datasets

Tabella canonica dei dataset usati nel benchmark. Ogni dataset è dichiarato anche in [`config/datasets.yaml`](../config/datasets.yaml) per il driver di esperimenti.

| Dataset key | File | Sorgente | Task supportati | `cell_type` col | `time` col | `perturb` col |
|---|---|---|---|---|---|---|
| `brain` | `brain_dataset.h5ad` | [CellxGene 0986e4cd](https://cellxgene.cziscience.com/collections/0986e4cd-7a58-405d-9b91-4b199bb4124e) | classification | `cell_type` | — | — |
| `spinal` | `spinal_dataset.h5ad` | [CellxGene 0986e4cd](https://cellxgene.cziscience.com/collections/0986e4cd-7a58-405d-9b91-4b199bb4124e) | classification | `cell_type` | — | — |
| `liver` | `liver_dataset.h5ad` | [CellxGene be679cb1](https://cellxgene.cziscience.com/collections/be679cb1-35f0-46c9-9a2d-30691862a54a) | classification | `cell_type` | — | — |
| `inner_ear` | `inner_ear_development.h5ad` | TBD (GEO?) | pseudotime | — | `week` | — |
| `GSE276896` | `GSE276896_adata_meta.h5ad` | GEO GSE276896 | pseudotime | — | `week` | — |
| `D1_Rest` | `D1_Rest.assigned_guide_undersampled.h5ad` | Perturb-seq dataset | perturbation | — | — | `gene_name` |
| `D1_Stim8hr` | `D1_Stim8hr.assigned_guide_undersampled.h5ad` | Perturb-seq dataset | perturbation | — | — | `gene_name` |
| `D1_Stim48hr` | `D1_Stim48hr.assigned_guide_undersampled.h5ad` | Perturb-seq dataset | perturbation | — | — | `gene_name` |

> Aggiorna questa tabella ogni volta che modifichi `config/datasets.yaml`. Le due fonti devono restare sincronizzate.

## Convenzioni `.obs` per task

Ogni pipeline downstream assume una colonna specifica in `adata.obs`:

- **Classification** (`scripts/run_cell_type_classification.py`): richiede `cell_type_col` (default `cell_type`). Celle con label `NaN` vengono filtrate. Label encoder applicato automaticamente.
- **Pseudo-time** (`scripts/run_pseudotime_analysis.py`): richiede `time_column` (default `week`). Usata per identificare la **root cell** (la cellula al tempo più precoce diventa il punto di partenza di DPT).
- **Perturbation** (`scripts/run_perturbation_analysis.py`): richiede `perturb_key` (es. `gene_name`) e `control_label` (es. `control`, `NTC`). Le perturbazioni con `<5` cellule vengono escluse.

## Baseline (PCA su HVG)

Per il task di classification, il preprocessor calcola automaticamente una baseline PCA su HVG dal raw count (`adata.raw`). Questa appare come `Baseline_PCA` nei CSV di output. È il punto di riferimento per dire "il foundation model batte la baseline non-FM o no?".

Per gli altri due task, non c'è ancora una baseline automatica. Aggiungerla è un follow-up (vedi [benchmarks.md](benchmarks.md#baseline)).

## Stato preprocessing degli h5ad

Lo stato dei dati in `data/raw/` non è uniforme:

- I dataset CellxGene (brain, spinal, liver) sono già **normalized + log1p** in `adata.X`, con i raw counts in `adata.raw`.
- I dataset Perturb-seq sono in **raw counts** in `adata.X`.

Questo è critico perché:
- **scFoundation** richiede `adata.X` già normalized+log1p (`prepare_data` lo assume).
- **Tahoe** vuole raw counts.

Il driver di esperimenti dovrebbe rifiutare combinazioni incompatibili. Oggi NON c'è check automatico. Va aggiunto.

## Note sul subsetting

Diversi dataset hanno il suffisso `undersampled` perché sono stati subsettati per ridurre i tempi di inferenza durante il prototyping. Per il paper:
- O ri-runnare su dataset completi e dichiararlo
- O documentare la strategia di subsetting (random? stratified per perturbazione?) in modo riproducibile

Il file `subset_creation.py` (rimosso) implementava questo; va recuperato/riscritto se necessario per la sezione *Methods* del paper.
