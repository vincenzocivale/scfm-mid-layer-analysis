import numpy as np
import scanpy as sc
from joblib import Parallel, delayed
from scipy.sparse import issparse, csr_matrix
from scipy.stats import spearmanr

_SEED = 42


def _topk_neighbors(distances, k):
    """Return (n, k) indices of the k nearest neighbours per row, sorted by distance.

    Works on a CSR sparse distance matrix where stored entries are the kNN distances
    (scanpy's `obsp['*_distances']` format). Previous implementation took the first
    k column-indices in the row, which is arbitrary order, not nearest order.
    """
    if not issparse(distances):
        distances = csr_matrix(distances)
    n = distances.shape[0]

    # Fast path: every row has the same number of stored entries (always true
    # for scanpy/pynndescent kNN graphs) -> fully vectorized top-k via argsort.
    row_nnz = np.diff(distances.indptr)
    if n > 0 and np.all(row_nnz == row_nnz[0]):
        m = int(row_nnz[0])
        if m == 0:
            return np.full((n, k), -1, dtype=np.int64)
        cols = distances.indices.reshape(n, m)
        vals = distances.data.reshape(n, m)
        order = np.argsort(vals, axis=1, kind='quicksort')[:, :k]
        sel = np.take_along_axis(cols, order, axis=1)
        if m >= k:
            return sel.astype(np.int64)
        out = np.full((n, k), -1, dtype=np.int64)
        out[:, :m] = sel
        return out

    # Fallback for ragged rows (variable nnz per row).
    out = np.full((n, k), -1, dtype=np.int64)
    for i in range(n):
        start, end = distances.indptr[i], distances.indptr[i + 1]
        cols = distances.indices[start:end]
        vals = distances.data[start:end]
        if len(cols) == 0:
            continue
        order = np.argsort(vals)[:k]
        sel = cols[order]
        out[i, : len(sel)] = sel
    return out


def _metric_ordering(ref_values, layer_dpt):
    if ref_values is None:
        return np.nan
    valid = np.isfinite(ref_values) & np.isfinite(layer_dpt)
    if valid.sum() < 3:
        return np.nan
    corr, _ = spearmanr(ref_values[valid], layer_dpt[valid])
    return corr


def _metric_continuity(ref_neighbors_indices, layer_indices):
    valid_a = ref_neighbors_indices != -1
    match = (ref_neighbors_indices[:, :, None] == layer_indices[:, None, :])
    in_b = match.any(axis=2)
    intersect_count = (valid_a & in_b).sum(axis=1)
    denom = np.maximum(valid_a.sum(axis=1), 1)
    overlaps = intersect_count / denom
    return float(np.mean(overlaps))


def _metric_geometry(ref_pseudotime, emb, seed, n_pairs=5000):
    rng = np.random.default_rng(seed)
    n_cells = emb.shape[0]
    idx1 = rng.integers(0, n_cells, n_pairs)
    idx2 = rng.integers(0, n_cells, n_pairs)
    delta_time = np.abs(ref_pseudotime[idx1] - ref_pseudotime[idx2])
    emb_dists = np.linalg.norm(emb[idx1] - emb[idx2], axis=1)
    valid = np.isfinite(delta_time) & np.isfinite(emb_dists)
    if valid.sum() < 3:
        return np.nan
    corr, _ = spearmanr(delta_time[valid], emb_dists[valid])
    return corr


def _evaluate_layer_isolated(layer_key, emb, n_obs, iroot, k_neighbors,
                              ref_pseudotime, ref_neighbors_indices, real_time, seed):
    """Evaluate one embedding layer on a fresh, minimal AnnData.

    Building a small per-layer AnnData (instead of mutating a shared one via
    `obsp['temp_*']` / `obsm['X_diffmap']`) makes layers independent, so they
    can be evaluated in parallel without one layer's neighbors/diffmap/dpt
    state clobbering another's.
    """
    print(f"Testing: {layer_key}")
    if issparse(emb):
        emb = emb.toarray()
    n_nans = np.isnan(emb).sum()
    if n_nans > 0:
        print(f"WARNING: {layer_key} has {n_nans} NaNs. Skipping.")
        return None

    adata = sc.AnnData(X=np.zeros((n_obs, 1), dtype=np.float32))
    adata.obsm[layer_key] = emb
    adata.uns['iroot'] = iroot

    sc.pp.neighbors(adata, use_rep=layer_key, n_neighbors=k_neighbors, key_added='temp')
    sc.tl.diffmap(adata, neighbors_key='temp')
    sc.tl.dpt(adata, neighbors_key='temp')
    layer_dpt = adata.obs['dpt_pseudotime'].values.copy()

    layer_indices = _topk_neighbors(adata.obsp['temp_distances'], k_neighbors)

    return {
        'Layer': layer_key,
        'Pseudotime_Corr_vs_RefDPT': _metric_ordering(ref_pseudotime, layer_dpt),
        'Pseudotime_Corr_vs_Time': _metric_ordering(real_time, layer_dpt),
        'Neighborhood_Overlap': _metric_continuity(ref_neighbors_indices, layer_indices),
        'Global_Geom_Corr': _metric_geometry(ref_pseudotime, emb, seed),
    }


class EmbeddingEvaluator:
    def __init__(self, adata, root_cell_index, time_col=None, seed=_SEED):
        self.adata = adata
        self.adata.uns['iroot'] = root_cell_index
        self.time_col = time_col
        self.seed = seed
        self.ref_pseudotime = None
        self.ref_neighbors_indices = None
        self.k_neighbors = 15
        self.real_time = None
        if time_col is not None and time_col in adata.obs.columns:
            t = adata.obs[time_col].values
            try:
                self.real_time = np.asarray(t, dtype=float)
            except (TypeError, ValueError):
                # Categorical / ordered: map to codes preserving order
                self.real_time = adata.obs[time_col].astype('category').cat.codes.values.astype(float)

    def setup_reference(self, use_rep='X_pca', n_neighbors=30):
        """Compute DPT and kNN graph on the reference representation."""
        print(f"--- Building reference graph (k={n_neighbors}, rep={use_rep}) ---")
        self.k_neighbors = n_neighbors
        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, use_rep=use_rep, key_added='ref')
        sc.tl.diffmap(self.adata, neighbors_key='ref')
        sc.tl.dpt(self.adata, neighbors_key='ref')
        self.ref_pseudotime = self.adata.obs['dpt_pseudotime'].values.copy()
        self.ref_neighbors_indices = _topk_neighbors(self.adata.obsp['ref_distances'], n_neighbors)

    def evaluate_layer(self, layer_key):
        return _evaluate_layer_isolated(
            layer_key, self.adata.obsm[layer_key], self.adata.n_obs, self.adata.uns['iroot'],
            self.k_neighbors, self.ref_pseudotime, self.ref_neighbors_indices, self.real_time, self.seed,
        )

    def evaluate_layers(self, layer_keys, n_jobs=1):
        """Evaluate several layers, optionally in parallel (n_jobs > 1).

        Each layer is evaluated on its own isolated AnnData (see
        `_evaluate_layer_isolated`), so results are identical regardless of
        n_jobs / execution order.
        """
        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_layer_isolated)(
                layer_key, self.adata.obsm[layer_key], self.adata.n_obs, self.adata.uns['iroot'],
                self.k_neighbors, self.ref_pseudotime, self.ref_neighbors_indices, self.real_time, self.seed,
            )
            for layer_key in layer_keys
        )
        return [r for r in results if r is not None]
