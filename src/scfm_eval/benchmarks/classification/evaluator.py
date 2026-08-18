import gc

import numpy as np
import scanpy as sc
from joblib import Parallel, delayed
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

_SEED = 42
_LEIDEN_RESOLUTIONS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)


class ClassificationEvaluator:
    def __init__(self, adata, cell_type_col, k_neighbors=15, seed=_SEED):
        self.adata = adata
        self.cell_type_col = cell_type_col
        self.k_neighbors = k_neighbors
        self.seed = seed
        self.labels = self._prepare_labels()
        self.n_true_classes = len(np.unique(self.labels))

    def _prepare_labels(self):
        if self.cell_type_col not in self.adata.obs.columns:
            raise ValueError(f"Column '{self.cell_type_col}' not in adata.obs")
        valid = self.adata.obs[self.cell_type_col].notna()
        self.adata = self.adata[valid].copy()
        return LabelEncoder().fit_transform(self.adata.obs[self.cell_type_col].values)

    def _get_X(self, layer_key):
        X = self.adata.obsm[layer_key]
        if issparse(X):
            X = X.toarray()
        return np.asarray(X)

    def _metric_logreg(self, X, y):
        clf = make_pipeline(
            StandardScaler(with_mean=True),
            LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto', random_state=self.seed),
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        # lbfgs solver has no internal threading, so running the 5 folds in
        # parallel is safe (no oversubscription) and doesn't change results.
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=5)
        return {
            'Accuracy':           float(np.mean(scores['test_accuracy'])),
            'Accuracy_std':       float(np.std(scores['test_accuracy'])),
            'F1_macro':           float(np.mean(scores['test_f1_macro'])),
            'F1_macro_std':       float(np.std(scores['test_f1_macro'])),
            'Precision_macro':    float(np.mean(scores['test_precision_macro'])),
            'Precision_macro_std': float(np.std(scores['test_precision_macro'])),
            'Recall_macro':       float(np.mean(scores['test_recall_macro'])),
            'Recall_macro_std':   float(np.std(scores['test_recall_macro'])),
        }

    def _metric_knn(self, X, y):
        # Parallelize across the 5 CV folds instead of inside KNeighborsClassifier:
        # 5-way fold parallelism plus single-threaded kNN search avoids the
        # oversubscription of 5 folds x 16 threads, with identical results.
        clf = make_pipeline(
            StandardScaler(with_mean=True),
            KNeighborsClassifier(n_neighbors=self.k_neighbors, n_jobs=1),
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        scores = cross_validate(clf, X, y, cv=cv, scoring=['accuracy', 'f1_macro'], n_jobs=5)
        return {
            'kNN_Accuracy':     float(np.mean(scores['test_accuracy'])),
            'kNN_Accuracy_std': float(np.std(scores['test_accuracy'])),
            'kNN_F1_macro':     float(np.mean(scores['test_f1_macro'])),
            'kNN_F1_macro_std': float(np.std(scores['test_f1_macro'])),
        }

    def _metric_silhouette(self, X, y, n_bootstrap=10, sample_size=20000):
        # Bootstrap over subsamples (same idx for X and y) to get mean ± std.
        rng = np.random.default_rng(self.seed)
        n = min(sample_size, X.shape[0])
        # Draw all bootstrap indices sequentially first (preserves the exact RNG
        # draw order/values of the original loop), then compute each O(n^2)
        # silhouette_score in parallel.
        idxs = [rng.choice(X.shape[0], n, replace=True) for _ in range(n_bootstrap)]
        scores = Parallel(n_jobs=n_bootstrap)(
            delayed(silhouette_score)(X[idx], y[idx], metric='euclidean') for idx in idxs
        )
        scores = [float(s) for s in scores]
        return {
            'Silhouette_Score':     float(np.mean(scores)),
            'Silhouette_Score_std': float(np.std(scores)),
        }

    def _metric_clustering(self, X, y):
        """Leiden resolution sweep; return best ARI/NMI and the resolution that achieved it."""
        temp = sc.AnnData(X)
        # Reduce to 50 PCs before kNN to avoid OOM: PyNNDescent tree memory scales
        # as O(n * n_trees * leaf_size * dim), exploding for dim > 50.
        n_comps = min(50, X.shape[1] - 1)
        if X.shape[1] > 50:
            sc.pp.pca(temp, n_comps=n_comps, random_state=self.seed, svd_solver='arpack')
            use_rep = 'X_pca'
        else:
            use_rep = 'X'
        sc.pp.neighbors(temp, n_neighbors=self.k_neighbors, use_rep=use_rep)

        # Build a minimal AnnData carrying only the neighbor graph (obsp +
        # uns['neighbors']), so each parallel Leiden run pickles a small copy
        # instead of the full embedding/PCA matrices.
        graph_adata = sc.AnnData(
            X=np.zeros((temp.n_obs, 1), dtype=np.float32),
            obsp={k: v for k, v in temp.obsp.items()},
            uns={'neighbors': temp.uns['neighbors']},
        )
        del temp
        gc.collect()

        def _run_leiden(resolution, random_state, key):
            g = graph_adata.copy()
            sc.tl.leiden(g, resolution=resolution, random_state=random_state, key_added=key)
            clusters = g.obs[key].values
            return clusters, adjusted_rand_score(y, clusters), normalized_mutual_info_score(y, clusters)

        # Resolution sweep, run in parallel. Tie-breaking (first max-ARI wins)
        # is applied afterwards, in resolution order, to match the original
        # sequential `ari > best['ARI']` logic.
        sweep_results = Parallel(n_jobs=len(_LEIDEN_RESOLUTIONS))(
            delayed(_run_leiden)(res, self.seed, f'leiden_{res}') for res in _LEIDEN_RESOLUTIONS
        )
        best = {'ARI': -np.inf, 'NMI': -np.inf, 'leiden_resolution': None, 'n_clusters': None}
        for res, (clusters, ari, nmi) in zip(_LEIDEN_RESOLUTIONS, sweep_results):
            if ari > best['ARI']:
                best = {
                    'ARI': float(ari),
                    'NMI': float(nmi),
                    'leiden_resolution': float(res),
                    'n_clusters': int(len(np.unique(clusters))),
                }

        # Multi-seed Leiden at best resolution to estimate stochastic variance.
        best_res = best['leiden_resolution']
        multiseed_results = Parallel(n_jobs=5)(
            delayed(_run_leiden)(best_res, run_i * 7 + 13, f'leiden_rep_{run_i}')
            for run_i in range(5)
        )
        best['ARI_std'] = float(np.std([ari for _, ari, _ in multiseed_results]))
        best['NMI_std'] = float(np.std([nmi for _, _, nmi in multiseed_results]))

        del graph_adata
        gc.collect()
        return best

    def evaluate_layer(self, layer_key):
        print(f"--- Evaluating layer: {layer_key} ---")
        X = self._get_X(layer_key)
        if np.isnan(X).any():
            print(f"WARNING: {layer_key} has NaNs. Skipped.")
            return None
        y = self.labels
        result = {'Layer': layer_key}
        result.update(self._metric_logreg(X, y))
        result.update(self._metric_knn(X, y))
        result.update(self._metric_silhouette(X, y))
        result.update(self._metric_clustering(X, y))
        return result
