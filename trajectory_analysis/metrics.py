"""
Funzioni per il calcolo delle metriche di trajectory evaluation layer-wise.
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, balanced_accuracy_score
from scipy.stats import spearmanr, kendalltau

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def evaluate_regression(emb, timepoint):
    try:
        from cuml.linear_model import Ridge
        cuml_available = True
    except ImportError:
        from sklearn.linear_model import Ridge
        cuml_available = False
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb)
    if np.issubdtype(timepoint.dtype, np.floating):
        bins = np.linspace(np.min(timepoint), np.max(timepoint), 6)
        stratify_labels = np.digitize(timepoint, bins)
    else:
        stratify_labels = timepoint
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros_like(timepoint, dtype=float)
    for train_idx, test_idx in cv.split(emb_scaled, stratify_labels):
        model = Ridge()
        model.fit(emb_scaled[train_idx], timepoint[train_idx])
        y_pred[test_idx] = model.predict(emb_scaled[test_idx])
    r2 = r2_score(timepoint, y_pred)
    mae = mean_absolute_error(timepoint, y_pred)
    spearman = spearmanr(timepoint, y_pred).correlation
    return {"R2": r2, "MAE": mae, "Spearman": spearman}

def evaluate_classification(emb, timepoint):
    try:
        from cuml.linear_model import LogisticRegression as cuMLLogisticRegression
        LogisticRegression = cuMLLogisticRegression
    except ImportError:
        from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from collections import Counter
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb)
    counts = Counter(timepoint)
    valid_classes = [c for c, n in counts.items() if n >= 5]
    mask = np.isin(timepoint, valid_classes)
    emb_scaled = emb_scaled[mask]
    timepoint_filt = timepoint[mask]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros_like(timepoint_filt)
    for train_idx, test_idx in cv.split(emb_scaled, timepoint_filt):
        model = LogisticRegression(max_iter=1000)
        model.fit(emb_scaled[train_idx], timepoint_filt[train_idx])
        y_pred[test_idx] = model.predict(emb_scaled[test_idx])
    macro_f1 = f1_score(timepoint_filt, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(timepoint_filt, y_pred)
    acc_pm1 = np.mean(np.abs(y_pred - timepoint_filt) <= 1)
    return {"MacroF1": macro_f1, "BalancedAcc": bal_acc, "Acc_pm1": acc_pm1}

def evaluate_pseudotime(adata, emb_key, time_col, k=15):
    sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=k, key_added="traj")
    sc.tl.diffmap(adata, neighbors_key="traj")
    if 'iroot' not in adata.uns:
        root_name = adata.obs[time_col].idxmin()
        root_idx = adata.obs.index.get_loc(root_name)
        adata.uns['iroot'] = root_idx
    sc.tl.dpt(adata, neighbors_key="traj")
    if "dpt_pseudotime" not in adata.obs.columns:
        raise KeyError("dpt_pseudotime non trovato in adata.obs dopo sc.tl.dpt")
    pt = adata.obs["dpt_pseudotime"].values
    tp = adata.obs[time_col].values
    spearman = spearmanr(tp, pt).correlation
    kendall = kendalltau(tp, pt).correlation
    return {"Pseudotime_Spearman": spearman, "Pseudotime_Kendall": kendall}

def evaluate_smoothness(adata, time_col, k=15):
    neighbors = adata.obsp["traj_connectivities"].toarray()
    tp = adata.obs[time_col].values
    diffs = []
    for i in range(neighbors.shape[0]):
        idx = np.argsort(-neighbors[i])[:k+1]
        local_diff = np.abs(tp[i] - tp[idx[1:]]).mean()
        diffs.append(local_diff)
    return {"Smoothness": np.mean(diffs)}
