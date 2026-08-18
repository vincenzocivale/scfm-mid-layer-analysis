import scanpy as sc
import numpy as np
import scipy.sparse

try:
    import cupy as cp
    from cuml.decomposition import PCA as cuPCA
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def _is_raw_counts(X, n_sample=500):
    """Return True if X appears to contain raw integer counts (not yet normalized)."""
    X_s = X[:min(n_sample, X.shape[0]), :min(n_sample, X.shape[1])]
    if scipy.sparse.issparse(X_s):
        X_s = X_s.toarray()
    if np.issubdtype(X_s.dtype, np.integer):
        return True
    return bool(np.all(X_s == np.floor(X_s)))


def prepare_reference_dataset(
    adata_full,
    time_col='week',
    n_top_genes=2500,
    n_pca_comps=50,
    use_gpu=False,
):
    """
    Prepara un AnnData di riferimento per il calcolo del pseudotime (ground truth).

    - HVG + PCA vengono calcolati su RNA counts (da adata.raw se presente,
      altrimenti da adata.X rilevando automaticamente se già normalizzato)
    - Gli embedding (es. scFM / Tahoe) vengono solo copiati in obsm
    - Robusto a NaN, celle zero, embedding-only AnnData

    Returns
    -------
    adata_ref : AnnData
        Dataset preprocessato (RNA → HVG+PCA, embedding copiati)
    root_idx : int
        Indice della cellula root
    """

    print(f"--- Preprocessing Reference ({n_top_genes} HVGs) ---")

    # ------------------------------------------------------------------
    # 1. Selezione matrice RNA
    # ------------------------------------------------------------------
    if adata_full.raw is not None:
        print("Using adata.raw (RNA counts) for HVG/PCA")
        adata_ref = adata_full.raw.to_adata()
        needs_normalization = True
    else:
        # Fall back to adata.X; auto-detect if it's raw counts or already normalized.
        # Integer-valued matrices are treated as raw counts and get normalize+log1p.
        # Non-integer (float) matrices are assumed already log-normalized — skipped.
        needs_normalization = _is_raw_counts(adata_full.X)
        if needs_normalization:
            print("Using adata.X (raw counts detected) for HVG/PCA")
        else:
            print("Using adata.X (pre-normalized detected) for HVG/PCA — skipping normalize+log1p")
        adata_ref = sc.AnnData(
            X=adata_full.X.copy() if not scipy.sparse.issparse(adata_full.X)
              else adata_full.X.copy(),
            obs=adata_full.obs.copy(),
            var=adata_full.var.copy(),
        )

    # Mantieni solo celle valide
    if time_col not in adata_full.obs:
        raise ValueError(f"Colonna '{time_col}' non trovata in obs")

    X_for_sum = adata_ref.X
    row_sums = (X_for_sum.sum(axis=1).A1 if scipy.sparse.issparse(X_for_sum)
                else np.asarray(X_for_sum).sum(axis=1))
    valid_cells = (
        ~adata_full.obs.loc[adata_ref.obs_names, time_col].isna().values
        & (row_sums > 0)
    )

    adata_ref = adata_ref[valid_cells].copy()
    adata_ref.obs[time_col] = adata_full.obs.loc[adata_ref.obs_names, time_col]

    adata_ref.var_names_make_unique()

    # ------------------------------------------------------------------
    # 2. Normalizzazione + log (solo se raw counts)
    # ------------------------------------------------------------------
    if needs_normalization:
        print("Normalizing and log-transforming RNA...")
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)

    # Rimuovi geni con NaN. Per matrici sparse, evita di densificare l'intera
    # matrice: i NaN sono sempre valori espliciti in .data (NaN != 0), quindi
    # basta scorrere .data per colonna.
    X = adata_ref.X
    if scipy.sparse.issparse(X):
        Xc = X.tocsc()
        nan_mask = np.isnan(Xc.data)
        valid_genes = np.ones(Xc.shape[1], dtype=bool)
        if nan_mask.any():
            col_ids = np.repeat(np.arange(Xc.shape[1]), np.diff(Xc.indptr))
            valid_genes[np.unique(col_ids[nan_mask])] = False
    else:
        valid_genes = ~np.isnan(X).any(axis=0)

    if valid_genes.sum() == 0:
        raise ValueError("All genes contain NaNs after normalization")

    adata_ref = adata_ref[:, valid_genes].copy()

    # ------------------------------------------------------------------
    # 3. Highly Variable Genes
    # ------------------------------------------------------------------
    print("Selecting highly variable genes...")
    sc.pp.highly_variable_genes(
        adata_ref,
        n_top_genes=min(n_top_genes, adata_ref.n_vars),
        subset=True,
    )

    if adata_ref.n_vars == 0:
        raise ValueError("No HVGs selected")

    # ------------------------------------------------------------------
    # 4. PCA
    # ------------------------------------------------------------------
    print("Computing PCA...")
    if use_gpu and GPU_AVAILABLE:
        X = adata_ref.X.A if scipy.sparse.issparse(adata_ref.X) else adata_ref.X
        X_gpu = cp.asarray(X)
        pca = cuPCA(n_components=min(n_pca_comps, X.shape[1]))
        adata_ref.obsm["X_pca"] = cp.asnumpy(pca.fit_transform(X_gpu))
    else:
        sc.pp.pca(adata_ref, n_comps=min(n_pca_comps, adata_ref.n_vars))

    # ------------------------------------------------------------------
    # 5. Root cell selection
    # ------------------------------------------------------------------
    min_time = adata_ref.obs[time_col].min()
    root_candidates = np.where(adata_ref.obs[time_col].values == min_time)[0]

    if len(root_candidates) == 0:
        raise ValueError(f"Nessuna cellula trovata per tempo {min_time}")

    root_idx = int(root_candidates[0])
    print(f"Root cell index selected: {root_idx} (time={min_time})")

    # ------------------------------------------------------------------
    # 6. Trasferimento embedding (subset corretto)
    # ------------------------------------------------------------------
    print("Transferring embeddings from original AnnData...")

    # Mappa cell name → indice numerico in adata_full
    cell_idx = adata_full.obs_names.get_indexer(adata_ref.obs_names)

    if np.any(cell_idx < 0):
        raise ValueError("Mismatch between adata_ref and adata_full cell names")

    for key, val in adata_full.obsm.items():
        adata_ref.obsm[key] = val[cell_idx]


    return adata_ref, root_idx
