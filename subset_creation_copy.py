import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datasets import load_dataset
import anndata
from scipy.sparse import lil_matrix
import random

random.seed(42)
np.random.seed(42)

# --- PARAMETRI ---
N_LOW = 500
N_MID = 1000
N_HIGH = 500

MAX_CELLS_LOW = None   # tutte
MAX_CELLS_MID = 15
MAX_CELLS_HIGH = 20

# --- PASSO 1: Conta cellule per perturbazione (gene_target) ---

print("Counting cells per gene_target...")

hct = load_dataset("Xaira-Therapeutics/X-Atlas-Orion", split="HEK293T", streaming=True)

gene_counts = Counter()
for i, cell in enumerate(hct):
    if cell["pass_guide_filter"]:
        gene_counts[cell["gene_target"]] += 1
    if i > 2_000_000:  # early stop per velocità
        break

counts = np.array(list(gene_counts.values()))
p25, p50, p75 = np.percentile(counts, [25, 50, 75])
print(f"Percentili (p25, p50, p75): {p25}, {p50}, {p75}")

# --- PASSO 2: Definisci strati ---

low_genes = [g for g, c in gene_counts.items() if c <= p25]
mid_genes = [g for g, c in gene_counts.items() if p25 < c <= p75]
high_genes = [g for g, c in gene_counts.items() if c > p75]

print(f"Low abundance genes: {len(low_genes)}")
print(f"Mid abundance genes: {len(mid_genes)}")
print(f"High abundance genes: {len(high_genes)}")

# --- PASSO 3: Seleziona random stratificato perturbazioni ---

selected_low = random.sample(low_genes, min(N_LOW, len(low_genes)))
selected_mid = random.sample(mid_genes, min(N_MID, len(mid_genes)))
selected_high = random.sample(high_genes, min(N_HIGH, len(high_genes)))

selected_genes = set(selected_low + selected_mid + selected_high)
print(f"Selected genes total: {len(selected_genes)}")

# --- PASSO 4: Setup max cellule per gene ---

max_cells_per_gene = {}
for g in selected_low:
    max_cells_per_gene[g] = MAX_CELLS_LOW
for g in selected_mid:
    max_cells_per_gene[g] = MAX_CELLS_MID
for g in selected_high:
    max_cells_per_gene[g] = MAX_CELLS_HIGH

# --- PASSO 5: Estrazione dati cellula per cellula ---

print("Extracting cells for selected perturbations...")

ds = load_dataset("Xaira-Therapeutics/X-Atlas-Orion", split="HCT116", streaming=True)

cell_counts = Counter()
cell_data = []
gene_token_set = set()

for cell in ds:
    if not cell["pass_guide_filter"]:
        continue
    gene = cell["gene_target"]
    if gene not in selected_genes:
        continue

    if max_cells_per_gene[gene] is not None and cell_counts[gene] >= max_cells_per_gene[gene]:
        continue

    cell_data.append(cell)
    cell_counts[gene] += 1

print(f"Extracted {len(cell_data)} cells in total.")

# --- PASSO 6: Costruzione matrice espressione sparsa ---

for cell in cell_data:
    gene_token_set.update(cell["gene_token_id"])

gene_token_list = sorted(gene_token_set)
gene_token_to_idx = {g: i for i, g in enumerate(gene_token_list)}

n_cells = len(cell_data)
n_genes = len(gene_token_list)

print(f"Building sparse matrix: {n_cells} cells × {n_genes} genes...")

X = lil_matrix((n_cells, n_genes), dtype=np.float32)

for i, cell in enumerate(cell_data):
    gene_ids = cell["gene_token_id"]
    exprs = cell["gene_expression"]
    for g_id, expr in zip(gene_ids, exprs):
        j = gene_token_to_idx[g_id]
        X[i, j] = expr

# --- PASSO 7: Prepara metadata cellule ---

obs = pd.DataFrame({
    "cell_barcode": [c["cell_barcode"] for c in cell_data],
    "sample": [c["sample"] for c in cell_data],
    "guide_target": [c["guide_target"] for c in cell_data],
    "gene_target": [c["gene_target"] for c in cell_data],
    "n_genes_by_counts": [c["n_genes_by_counts"] for c in cell_data],
    "total_counts": [c["total_counts"] for c in cell_data],
    "total_counts_mt": [c["total_counts_mt"] for c in cell_data],
    "pct_counts_mt": [c["pct_counts_mt"] for c in cell_data],
    "pass_guide_filter": [c["pass_guide_filter"] for c in cell_data],
})

# --- PASSO 8: Carica metadata geni (FIX CORRETTO) ---

gene_metadata = load_dataset(
    "Xaira-Therapeutics/X-Atlas-Orion",
    "gene_metadata"
)

# Prendi esplicitamente lo split "train"
gene_meta_df = gene_metadata["train"].to_pandas()

# Subset sui gene_token_id effettivamente presenti
gene_meta_sub = gene_meta_df[
    gene_meta_df["gene_token_id"].isin(gene_token_list)
].copy()

# Usa gene_token_id come indice e riordina
gene_meta_sub = gene_meta_sub.set_index("gene_token_id").loc[gene_token_list]


# --- PASSO 9: Costruisci AnnData e salva ---

adata = anndata.AnnData(X=X.tocsr(), obs=obs, var=gene_meta_sub)

output_path = "XAtlasOrion_HEK293T_subset.h5ad"
print(f"Saving AnnData to {output_path} ...")
adata.write(output_path)
print("Done.")
