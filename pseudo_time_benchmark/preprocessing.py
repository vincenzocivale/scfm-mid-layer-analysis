
import scanpy as sc
import numpy as np

try:
    import cupy as cp
    from cuml.decomposition import PCA as cuPCA
    from cuml.neighbors import NearestNeighbors as cuNN
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def prepare_reference_dataset(adata_full, time_col='week', n_top_genes=2500, n_pca_comps=50, use_gpu=False):
    """
    Crea un subset ottimizzato per il calcolo del Ground Truth (Pseudotime).
    
    Args:
        adata_full: AnnData originale (92k celle).
        time_col: Colonna in obs che indica il tempo (es. 'week').
        n_top_genes: Numero di High Variable Genes.
        n_pca_comps: Componenti PCA.
        
    Returns:
        adata_ref: AnnData subset (HVG + PCA + Embedding copiati).
        root_idx: Indice della cellula root nel nuovo adata.
    """

    print(f"--- Preprocessing Reference ({n_top_genes} HVGs) ---")
    # 1. Copia per lavorare sugli HVG senza perdere dati originali
    adata_ref = adata_full.copy()

    print("Normalizing and Log-transforming...")
    sc.pp.normalize_total(adata_ref, target_sum=1e4)
    sc.pp.log1p(adata_ref)

    # 3. High Variable Genes
    sc.pp.highly_variable_genes(adata_ref, n_top_genes=n_top_genes, subset=True)

    # 4. PCA (CPU o GPU)
    print("Computing PCA...")
    if use_gpu and GPU_AVAILABLE:
        X = adata_ref.X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        X_gpu = cp.asarray(X)
        pca = cuPCA(n_components=n_pca_comps)
        X_pca = pca.fit_transform(X_gpu)
        adata_ref.obsm['X_pca'] = cp.asnumpy(X_pca)
    else:
        sc.pp.pca(adata_ref, n_comps=n_pca_comps)

    # 5. Root Finding
    if time_col not in adata_ref.obs:
        raise ValueError(f"Colonna '{time_col}' non trovata.")
    
    min_time = adata_ref.obs[time_col].min()
    # Trova indici dove il tempo è minimo
    root_candidates = np.where(adata_ref.obs[time_col] == min_time)[0]
    
    if len(root_candidates) == 0:
        raise ValueError(f"Nessuna cellula trovata per tempo {min_time}")
        
    root_idx = root_candidates[0]
    print(f"Root cell index selected: {root_idx} (Time point: {min_time})")


    print("Transferring embeddings from original data...")
    for key in adata_full.obsm.keys():
        adata_ref.obsm[key] = adata_full.obsm[key]

    return adata_ref, root_idx