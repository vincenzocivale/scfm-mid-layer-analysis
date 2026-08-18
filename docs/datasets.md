# Datasets

Tabella canonica dei dataset usati nel benchmark. Ogni dataset è dichiarato anche in [`config/datasets.yaml`](../config/datasets.yaml) per il driver di esperimenti.

I raw sono organizzati in sottocartelle per task: `data/raw/{classification,pseudotime,perturbation}/`.

| Dataset key | File | Sorgente | Reference (DOI) | Task | Key col |
|---|---|---|---|---|---|
| `inner_ear` | `pseudotime/inner_ear_development.h5ad` | TBD (GEO?) | TBD | pseudotime | `week` |
| `D1_Rest` | `perturbation/D1_Rest.assigned_guide_undersampled.h5ad` | Perturb-seq dataset | TBD | perturbation | `gene_name` |
| `D1_Stim8hr` | `perturbation/D1_Stim8hr.assigned_guide_undersampled.h5ad` | Perturb-seq dataset | TBD | perturbation | `gene_name` |
| `D1_Stim48hr` | `perturbation/D1_Stim48hr.assigned_guide_undersampled.h5ad` | Perturb-seq dataset | TBD | perturbation | `gene_name` |
| `liver_ma` | `classification/liver_ma.h5ad` | [GEO GSE151530](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE151530) | Ma et al., *Cancer Cell* 36(4):418–430 (2019). [doi:10.1016/j.ccell.2019.08.007](https://doi.org/10.1016/j.ccell.2019.08.007) | classification | `cell_type` |
| `lung_kim` | `classification/lung_kim.h5ad` | [GEO GSE131907](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907) | Kim et al., *Nat. Commun.* 11:2285 (2020). [doi:10.1038/s41467-020-16164-1](https://doi.org/10.1038/s41467-020-16164-1) | classification | `cell_type` (alias of `Cell_type`) |
| `emt_cook` | `pseudotime/emt_cook.h5ad` | [GEO GSE147405](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE147405) (A549 TGFB1 TimeCourse) | Cook & Vanderhyden, *Nat. Commun.* 11:2142 (2020). [doi:10.1038/s41467-020-16066-2](https://doi.org/10.1038/s41467-020-16066-2) | pseudotime | `Time` |
| `veres` | `pseudotime/veres.h5ad` | [GEO GSE114412](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114412) (Stage 5 aggregate) | Veres et al., *Nature* 569:368–373 (2019). [doi:10.1038/s41586-019-1168-5](https://doi.org/10.1038/s41586-019-1168-5) | pseudotime | `day` (alias of `CellWeek`; ~46% cells annotated) |
| `hspc_bouman` | `pseudotime/hspc_bouman.h5ad` | [GEO GSE226824](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226824) | Bouman et al., *Nat. Commun.* 15:7079 (2024). [doi:10.1038/s41467-024-51442-2](https://doi.org/10.1038/s41467-024-51442-2) | pseudotime | `time` |
| `norman_2019` | `perturbation/norman_2019.h5ad` | [GEO GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344) | Norman et al., *Science* 365(6455):786–793 (2019). [doi:10.1126/science.aax4438](https://doi.org/10.1126/science.aax4438) | perturbation | `guide_identity` (control = `NegCtrl1_NegCtrl0__NegCtrl1_NegCtrl0`) |
| `adamson_2016` | `perturbation/adamson_2016.h5ad` | [GEO GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546) (pilot 3 GSMs) | Adamson et al., *Cell* 167(7):1867–1882 (2016). [doi:10.1016/j.cell.2016.11.048](https://doi.org/10.1016/j.cell.2016.11.048) | perturbation | `perturbation` (control = `3x_neg_ctrl_pMJ144-1`) |

> Aggiorna questa tabella ogni volta che modifichi `config/datasets.yaml`. Le due fonti devono restare sincronizzate. Per la procedura di download/assembly di nuovi dataset (solo **GEO** e **HuggingFace** sono accessibili da questo server) vedi [`scripts/dataset_downloads/`](../scripts/dataset_downloads/) (`manifest.yaml` → `download.py` → `assemble.py` → `verify.py`). Le citazioni complete (autori + journal + anno + DOI) per ogni dataset, **inclusi quelli non scaricabili da qui**, sono in `scripts/dataset_downloads/manifest.yaml`.

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
