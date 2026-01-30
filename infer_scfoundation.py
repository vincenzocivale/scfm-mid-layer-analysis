import scanpy as sc
import torch
import numpy as np
import pandas as pd
import os
from models.scfoundation_embedder import scFoundationEmbedder

# Configurazione percorso e parametri
input_path = "/home/oem/scfm-layer-analysis-refactored/data/raw/brain_dataset.h5ad"  # Modifica se necessario
output_path = "/home/oem/scfm-layer-analysis-refactored/data/data/embeddings/brain_dataset_scfoundation_embeddings.h5ad"
gene_index_path = "/home/oem/scfm-layer-analysis-refactored/models/OS_scRNA_gene_index.19264.tsv"
model_ckpt = "/home/oem/scfm-layer-analysis-refactored/models/models.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica AnnData
adata = sc.read_h5ad(input_path)

# Carica modello

embedder = scFoundationEmbedder(fp16=False)
adata_prep = embedder.prepare_data(adata)

# Esegui inferenza su tutti i layer (0-11)
layer_indices = list(range(12))
embeddings = embedder.extract_embeddings_for_layers(adata_prep, layer_indices=layer_indices, batch_size=4)

# Crea nuovo AnnData con embeddings
adata_out = sc.AnnData(X=embeddings, obs=adata.obs.copy())

for layer, arr in embeddings.items():
    adata_out.obsm[f'X_layer_{layer}'] = arr

adata_out.uns['layer_embeddings'] = {
    'model': 'scfoundation',
    'n_layers': len(embeddings),
    'layer_keys': sorted([f"X_layer_{l}" for l in embeddings.keys()]),
}


# Copia .raw in modo compatibile
if adata.raw is not None:
    adata_out.raw = sc.AnnData(X=adata.raw.X.copy(), var=adata.raw.var.copy(), obs=adata_out.obs.copy())
else:
    adata_out.raw = sc.AnnData(X=adata.X.copy(), var=adata.var.copy(), obs=adata_out.obs.copy())



# Salva
adata_out.write(output_path)
print(f"Embeddings salvati in {output_path}")
