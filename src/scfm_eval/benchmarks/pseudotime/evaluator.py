import numpy as np
import scanpy as sc
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

    def _compute_layer_graph(self, layer_key):
        sc.pp.neighbors(self.adata, use_rep=layer_key, n_neighbors=self.k_neighbors, key_added='temp')
        sc.tl.diffmap(self.adata, neighbors_key='temp')
        sc.tl.dpt(self.adata, neighbors_key='temp')

    def _cleanup_layer_graph(self):
        for k in ['temp', 'temp_distances', 'temp_connectivities']:
            self.adata.uns.pop(k, None)
            self.adata.obsp.pop(k, None)
        self.adata.obsm.pop('X_diffmap', None)
        self.adata.uns.pop('diffmap_evals', None)

    def _metric_ordering_vs_ref(self, layer_dpt):
        valid = np.isfinite(self.ref_pseudotime) & np.isfinite(layer_dpt)
        if valid.sum() < 3:
            return np.nan
        corr, _ = spearmanr(self.ref_pseudotime[valid], layer_dpt[valid])
        return corr

    def _metric_ordering_vs_time(self, layer_dpt):
        if self.real_time is None:
            return np.nan
        valid = np.isfinite(self.real_time) & np.isfinite(layer_dpt)
        if valid.sum() < 3:
            return np.nan
        corr, _ = spearmanr(self.real_time[valid], layer_dpt[valid])
        return corr

    def _metric_continuity(self):
        layer_indices = _topk_neighbors(self.adata.obsp['temp_distances'], self.k_neighbors)
        overlaps = np.empty(self.adata.shape[0])
        for i in range(self.adata.shape[0]):
            a = self.ref_neighbors_indices[i]
            b = layer_indices[i]
            a = a[a >= 0]
            b = b[b >= 0]
            denom = max(len(a), 1)
            overlaps[i] = np.intersect1d(a, b).size / denom
        return float(np.mean(overlaps))

    def _metric_geometry(self, layer_key, n_pairs=5000):
        rng = np.random.default_rng(self.seed)
        n_cells = self.adata.shape[0]
        idx1 = rng.integers(0, n_cells, n_pairs)
        idx2 = rng.integers(0, n_cells, n_pairs)
        delta_time = np.abs(self.ref_pseudotime[idx1] - self.ref_pseudotime[idx2])
        emb = self.adata.obsm[layer_key]
        if issparse(emb):
            emb = emb.toarray()
        emb_dists = np.linalg.norm(emb[idx1] - emb[idx2], axis=1)
        valid = np.isfinite(delta_time) & np.isfinite(emb_dists)
        if valid.sum() < 3:
            return np.nan
        corr, _ = spearmanr(delta_time[valid], emb_dists[valid])
        return corr

    def evaluate_layer(self, layer_key):
        print(f"Testing: {layer_key}")
        arr = self.adata.obsm[layer_key]
        if issparse(arr):
            arr = arr.toarray()
        n_nans = np.isnan(arr).sum()
        if n_nans > 0:
            print(f"WARNING: {layer_key} has {n_nans} NaNs. Skipping.")
            return None
        try:
            self._compute_layer_graph(layer_key)
            layer_dpt = self.adata.obs['dpt_pseudotime'].values.copy()
            result = {
                'Layer': layer_key,
                'Pseudotime_Corr_vs_RefDPT': self._metric_ordering_vs_ref(layer_dpt),
                'Pseudotime_Corr_vs_Time': self._metric_ordering_vs_time(layer_dpt),
                'Neighborhood_Overlap': self._metric_continuity(),
                'Global_Geom_Corr': self._metric_geometry(layer_key),
            }
        finally:
            self._cleanup_layer_graph()
        return result
