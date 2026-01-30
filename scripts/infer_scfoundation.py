import scanpy as sc
import torch
import numpy as np
import pandas as pd
import os
from models.scfoundation_embedder import scFoundationEmbedder

# Configurazione percorso e parametri
input_path = "data/raw/GSE276896_adata_meta.h5ad"  # Modifica se necessario
output_path = "data/embeddings/GSE276896_adata_meta_scfoundation_embeddings.h5ad"
gene_index_path = "models/OS_scRNA_gene_index.19264.tsv"
model_ckpt = "models/models.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica AnnData
adata = sc.read_h5ad(input_path)

# Carica modello

embedder = scFoundationEmbedder(
    model_ckpt=model_ckpt,
    gene_index_path=gene_index_path,
    device=device,
    batch_size=4,
    fp16=False
)


# Estrai embeddings su tutte le cellule
embeddings = []
bs = 128
for i in range(0, adata.n_obs, bs):
    batch = adata.X[i:i+bs]
    emb = embedder.get_embeddings(batch)
    embeddings.append(emb)
embeddings = np.concatenate(embeddings, axis=0)

# Crea nuovo AnnData con embeddings
adata_out = sc.AnnData(X=embeddings, obs=adata.obs.copy())

# Copia .raw in modo compatibile
if adata.raw is not None:
    adata_out.raw = sc.AnnData(X=adata.raw.X.copy(), var=adata.raw.var.copy(), obs=adata_out.obs.copy())
else:
    adata_out.raw = sc.AnnData(X=adata.X.copy(), var=adata.var.copy(), obs=adata_out.obs.copy())



# Salva
adata_out.write(output_path)
print(f"Embeddings salvati in {output_path}")
