
"""
Plotting per metriche di pseudotempo, stile chiaro per articoli.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def natural_sort_key(s):
    """Ordina stringhe come Layer_1, Layer_2, ..., Layer_10 in modo naturale."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def plot_pseudotime_correlation(df, model_name=None, ci=None, ax=None):
    """
    Primary figure: layer-wise pseudotime preservation (Spearman correlation).
    df: DataFrame con colonne ['Layer', 'Pseudotime_Corr']
    model_name: nome modello per la legenda
    ci: intervallo di confidenza (opzionale, array o None)
    ax: matplotlib axis (opzionale)
    """
    df = df.copy()
    df['Layer'] = df['Layer'].astype(str)
    df = df.sort_values(by='Layer', key=lambda x: x.map(natural_sort_key))
    x = df['Layer']
    y = df['Pseudotime_Corr']
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker='o', color='#0072B2', linewidth=2, label=model_name or 'Model')
    if ci is not None:
        ax.fill_between(x, y-ci, y+ci, color='#0072B2', alpha=0.2)
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('Pseudotime correlation (Spearman)', fontsize=13)
    ax.set_title('Layer-wise pseudotime preservation', fontsize=15)
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    if model_name is not None:
        ax.legend(loc='upper left', bbox_to_anchor=(1,1), frameon=False)
    plt.tight_layout()
    if ax is None:
        plt.show()


def plot_umap_snapshots(adata, layer_keys, ref_pseudotime, use_rep_prefix='X_layer_', main_temporal='Reference pseudotime', figsize=(15,5)):
    """
    Three-panel UMAP/PCA colored by reference pseudotime: early, best, late layer.
    adata: AnnData
    layer_keys: [early, best, late] layer keys
    ref_pseudotime: array di pseudotempo di riferimento
    """
    import scanpy as sc
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for i, key in enumerate(layer_keys):
        sc.pp.neighbors(adata, use_rep=key, n_neighbors=15)
        sc.tl.umap(adata)
        adata.obsm[f'X_umap_{key}'] = adata.obsm['X_umap'].copy()
        sc.pl.umap(adata, color=ref_pseudotime, ax=axes[i], show=False,
                  title=f'{main_temporal}\n{key}', cmap='viridis', frameon=True)
    plt.tight_layout()
    plt.show()


def plot_distance_vs_pseudotime(df, layer, n_points=2000, ax=None):
    """
    Geometry plot: scatter di |Δ pseudotime| vs embedding distance per layer.
    df: DataFrame con colonne ['PT_i', 'PT_j', 'Dist']
    layer: nome layer
    n_points: quanti punti campionare
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    if len(df) > n_points:
        df = df.sample(n_points, random_state=42)
    ax.scatter(np.abs(df['PT_i']-df['PT_j']), df['Dist'], alpha=0.2, s=10, color='#1f77b4')
    ax.set_xlabel('|Δ pseudotime|')
    ax.set_ylabel('Embedding distance')
    ax.set_title(f'Distance vs pseudotime difference\n{layer}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    if ax is None:
        plt.show()
