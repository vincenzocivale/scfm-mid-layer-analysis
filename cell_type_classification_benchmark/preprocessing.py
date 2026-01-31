import scanpy as sc
import numpy as np
import scipy.sparse

try:
    import cupy as cp
    from cuml.decomposition import PCA as cuPCA
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def prepare_classification_dataset(
    adata_full,
    cell_type_col='cell_type',
    n_top_genes=2500,
    n_pca_comps=50,
    use_gpu=False,
):
    """
    Prepara un AnnData per la valutazione della classificazione,
    calcolando anche una rappresentazione di baseline (PCA su HVG).

    - Filtra le cellule senza etichetta.
    - Se sono presenti dati RNA in .raw, calcola HVG+PCA.
    - Copia tutti gli embedding dall'AnnData originale a quello filtrato.

    Returns
    -------
    adata_proc : AnnData
        Dataset preprocessato (con PCA se possibile) e filtrato.
    """
    print(f"--- Preprocessing for Classification (using '{cell_type_col}') ---")

    # 1. Filtra cellule senza etichetta
    if cell_type_col not in adata_full.obs:
        raise ValueError(f"Colonna '{cell_type_col}' non trovata in adata.obs")

    valid_cells = adata_full.obs[cell_type_col].notna()
    if valid_cells.sum() == 0:
        raise ValueError(f"Nessuna cellula trovata con etichette valide nella colonna '{cell_type_col}'")

    adata_proc = adata_full[valid_cells].copy()
    print(f"Dataset ridotto da {adata_full.n_obs} a {adata_proc.n_obs} cellule.")

    # 2. Calcola baseline PCA se possibile
    if adata_full.raw is not None:
        print("Dati RNA trovati in .raw. Calcolo baseline PCA su HVG...")
        # Usa un subset di adata_proc con i dati raw per il calcolo
        adata_rna = adata_full.raw.to_adata()
        adata_rna = adata_rna[adata_proc.obs_names].copy() # allinea le cellule

        # Normalizzazione e log
        sc.pp.normalize_total(adata_rna, target_sum=1e4)
        sc.pp.log1p(adata_rna)
        
        # HVG
        sc.pp.highly_variable_genes(
            adata_rna,
            n_top_genes=min(n_top_genes, adata_rna.n_vars),
            subset=True
        )

        # PCA
        if use_gpu and GPU_AVAILABLE:
            X = adata_rna.X.A if scipy.sparse.issparse(adata_rna.X) else adata_rna.X
            X_gpu = cp.asarray(X)
            pca = cuPCA(n_components=min(n_pca_comps, X.shape[1]))
            adata_proc.obsm['X_pca'] = cp.asnumpy(pca.fit_transform(X_gpu))
        else:
            sc.pp.pca(adata_rna, n_comps=min(n_pca_comps, adata_rna.n_vars))
            adata_proc.obsm['X_pca'] = adata_rna.obsm['X_pca']
        
        print("Baseline 'X_pca' calcolata e aggiunta a .obsm")
    else:
        print("Warning: Dati RNA non trovati in .raw. La baseline PCA non sarà calcolata.")
        if 'X_pca' in adata_proc.obsm:
             print("Trovata una rappresentazione 'X_pca' preesistente, verrà usata come baseline.")
        else:
             print("Nessuna baseline 'X_pca' disponibile.")


    # 3. Assicurati che gli embedding originali siano ancora lì
    # (sono stati già copiati con adata_proc = adata_full[valid_cells].copy())
    if not adata_proc.obsm:
        print("Warning: l'oggetto AnnData non ha embedding in .obsm.")

    return adata_proc