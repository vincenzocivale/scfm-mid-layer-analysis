import numpy as np
import pandas as pd
from collections import Counter
from datasets import load_dataset
import anndata
from scipy.sparse import coo_matrix
import random
from tqdm import tqdm
import gc

random.seed(42)
np.random.seed(42)

# --- PARAMETRI ---
N_LOW = 500
N_MID = 1000
N_HIGH = 500

MAX_CELLS_LOW = None
MAX_CELLS_MID = 15
MAX_CELLS_HIGH = 20

BATCH_SIZE = 10000  # Processa in batch per efficienza

# --- PASSO 1: Conta cellule (OTTIMIZZATO) ---
print("Counting cells per gene_target...")

hct = load_dataset("Xaira-Therapeutics/X-Atlas-Orion", split="HCT116", streaming=True)

gene_counts = Counter()
batch = []

for i, cell in enumerate(hct):
    if cell["pass_guide_filter"]:
        batch.append(cell["gene_target"])
    
    # Processa batch
    if len(batch) >= BATCH_SIZE:
        gene_counts.update(batch)
        batch = []
    
    if i > 2_000_000:
        break

# Ultimo batch
if batch:
    gene_counts.update(batch)

counts = np.array(list(gene_counts.values()))
p25, p50, p75 = np.percentile(counts, [25, 50, 75])
print(f"Percentili (p25, p50, p75): {p25}, {p50}, {p75}")

# --- PASSO 2-4: Selezione stratificata ---
low_genes = [g for g, c in gene_counts.items() if c <= p25]
mid_genes = [g for g, c in gene_counts.items() if p25 < c <= p75]
high_genes = [g for g, c in gene_counts.items() if c > p75]

print(f"Low: {len(low_genes)}, Mid: {len(mid_genes)}, High: {len(high_genes)}")

selected_low = random.sample(low_genes, min(N_LOW, len(low_genes)))
selected_mid = random.sample(mid_genes, min(N_MID, len(mid_genes)))
selected_high = random.sample(high_genes, min(N_HIGH, len(high_genes)))

selected_genes = set(selected_low + selected_mid + selected_high)
print(f"Selected genes total: {len(selected_genes)}")

# Pre-computa max cells
max_cells_per_gene = {g: MAX_CELLS_LOW for g in selected_low}
max_cells_per_gene.update({g: MAX_CELLS_MID for g in selected_mid})
max_cells_per_gene.update({g: MAX_CELLS_HIGH for g in selected_high})

# --- STRATEGIA SINGLE-PASS ---
print("Single pass: collecting data and building matrix...")

ds = load_dataset("Xaira-Therapeutics/X-Atlas-Orion", split="HCT116", streaming=True)

# Accumulatori pre-allocati
cell_counts = Counter()
gene_token_set = set()

# Pre-alloca con capacità stimata
ESTIMATED_CELLS = 20000
rows = []
cols = []
data = []

obs_lists = {
    "cell_barcode": [],
    "sample": [],
    "guide_target": [],
    "gene_target": [],
    "n_genes_by_counts": [],
    "total_counts": [],
    "total_counts_mt": [],
    "pct_counts_mt": [],
    "pass_guide_filter": []
}

# Mappatura temporanea gene_token -> index (si aggiornerà)
temp_gene_map = {}
next_gene_idx = 0

cell_idx = 0
processed = 0

# Usa itertools per velocizzare
from itertools import islice

for cell in tqdm(ds, desc="Processing cells"):
    processed += 1
    
    if not cell["pass_guide_filter"]:
        continue
    
    gene = cell["gene_target"]
    if gene not in selected_genes:
        continue
    
    # Controllo limite celle
    max_cells = max_cells_per_gene[gene]
    if max_cells is not None and cell_counts[gene] >= max_cells:
        continue
    
    # Processa gene expression
    gene_ids = cell["gene_token_id"]
    exprs = cell["gene_expression"]
    
    # Aggiungi gene tokens e costruisci matrice in un solo loop
    for g_id, expr in zip(gene_ids, exprs):
        gene_token_set.add(g_id)
        
        # Usa mappatura temporanea (riordineremo dopo)
        if g_id not in temp_gene_map:
            temp_gene_map[g_id] = next_gene_idx
            next_gene_idx += 1
        
        rows.append(cell_idx)
        cols.append(temp_gene_map[g_id])
        data.append(expr)
    
    # Aggiungi metadata (più veloce con append diretto)
    obs_lists["cell_barcode"].append(cell["cell_barcode"])
    obs_lists["sample"].append(cell["sample"])
    obs_lists["guide_target"].append(cell["guide_target"])
    obs_lists["gene_target"].append(cell["gene_target"])
    obs_lists["n_genes_by_counts"].append(cell["n_genes_by_counts"])
    obs_lists["total_counts"].append(cell["total_counts"])
    obs_lists["total_counts_mt"].append(cell["total_counts_mt"])
    obs_lists["pct_counts_mt"].append(cell["pct_counts_mt"])
    obs_lists["pass_guide_filter"].append(cell["pass_guide_filter"])
    
    cell_counts[gene] += 1
    cell_idx += 1
    
    # Early stop se abbiamo tutte le celle necessarie
    if all(
        (max_cells_per_gene[g] is None or cell_counts[g] >= max_cells_per_gene[g])
        for g in selected_genes
    ):
        print(f"All genes filled at {processed} processed cells")
        break

n_cells_total = cell_idx
print(f"\nExtracted {n_cells_total} cells, {len(gene_token_set)} unique genes")

# --- Riordina geni e ricostruisci indici ---
print("Reordering genes...")
gene_token_list = sorted(gene_token_set)
gene_token_to_final_idx = {g: i for i, g in enumerate(gene_token_list)}

# Rimappa cols
cols_final = [gene_token_to_final_idx[gene_token_list[temp_gene_map[list(temp_gene_map.keys())[list(temp_gene_map.values()).index(old_idx)]]]] 
              for old_idx in cols]

# Modo più semplice: ricostruisci usando la mappatura corretta
print("Remapping column indices...")
# Crea reverse map: temp_idx -> gene_id
temp_idx_to_gene = {v: k for k, v in temp_gene_map.items()}
# Rimappa tutte le colonne
cols_final = [gene_token_to_final_idx[temp_idx_to_gene[old_idx]] for old_idx in cols]

# --- Costruisci matrice ---
print("Building sparse matrix...")
n_genes = len(gene_token_list)

# Converti a numpy arrays (più veloce)
rows_arr = np.array(rows, dtype=np.int32)
cols_arr = np.array(cols_final, dtype=np.int32)
data_arr = np.array(data, dtype=np.float32)

X = coo_matrix((data_arr, (rows_arr, cols_arr)), 
               shape=(n_cells_total, n_genes), 
               dtype=np.float32)
X = X.tocsr()

print(f"Matrix: {X.shape}, sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.4f}")

# Cleanup
del rows, cols, data, rows_arr, cols_arr, data_arr, cols_final
gc.collect()

# --- Metadata ---
print("Creating DataFrames...")
obs = pd.DataFrame(obs_lists)

# --- Gene metadata ---
print("Loading gene metadata...")
gene_metadata = load_dataset("Xaira-Therapeutics/X-Atlas-Orion", "gene_metadata")
gene_meta_df = gene_metadata["train"].to_pandas()

# Filtra e riordina in un colpo
gene_meta_sub = (gene_meta_df[gene_meta_df["gene_token_id"].isin(gene_token_list)]
                 .set_index("gene_token_id")
                 .loc[gene_token_list]
                 .reset_index(drop=False))

# --- AnnData ---
print("Creating AnnData...")
adata = anndata.AnnData(X=X, obs=obs, var=gene_meta_sub)

output_path = "XAtlasOrion_HCT116_subset.h5ad"
print(f"Saving to {output_path}...")
adata.write(output_path, compression="gzip")
print("Done!")