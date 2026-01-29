"""
Core visualization for perturbation analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional


def plot_layer_correlation(
    correlation_df: pd.DataFrame,
    output_path: str,
    model_name: str = "Model"
):
    """
    Figure 1: Layer-wise semantic alignment (main result).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(correlation_df['layer'], correlation_df['correlation'], 
           'o-', linewidth=2.5, markersize=8, label=model_name)
    
    # Mark peak
    best_idx = correlation_df['correlation'].idxmax()
    best_layer = correlation_df.loc[best_idx, 'layer']
    best_corr = correlation_df.loc[best_idx, 'correlation']
    ax.axvline(best_layer, color='red', linestyle='--', alpha=0.5)
    ax.plot(best_layer, best_corr, 'r*', markersize=15)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Spearman Correlation with DE', fontsize=12)
    ax.set_title('Perturbation Semantic Alignment Across Layers', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_held_out_generalization(
    correlation_df: pd.DataFrame,
    held_out_results: Dict[int, Dict],
    output_path: str,
    model_name: str = "Model"
):
    """
    Figure 2: Held-out perturbation robustness.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Full set
    ax.plot(correlation_df['layer'], correlation_df['correlation'], 
           'o-', linewidth=2.5, label=f'{model_name} (full)', color='steelblue')
    
    # Held-out
    held_out_df = pd.DataFrame([
        {'layer': k, 'corr': v['mean_corr'], 'std': v['std_corr']}
        for k, v in held_out_results.items()
    ])
    ax.plot(held_out_df['layer'], held_out_df['corr'], 
           's--', linewidth=2.5, label=f'{model_name} (held-out)', color='orange')
    ax.fill_between(held_out_df['layer'],
                    held_out_df['corr'] - held_out_df['std'],
                    held_out_df['corr'] + held_out_df['std'],
                    alpha=0.2, color='orange')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title('Generalization to Held-Out Perturbations', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_similarity_heatmaps(
    biological_sim: pd.DataFrame,
    embedding_sims: Dict[int, pd.DataFrame],
    best_layer: int,
    output_path: str,
    n_display: int = 40,
    seed: int = 42
):
    """
    Figure 3: Qualitative similarity comparison.
    """
    np.random.seed(seed)
    
    perturbations = biological_sim.index
    if len(perturbations) > n_display:
        display = np.random.choice(perturbations, n_display, replace=False)
    else:
        display = perturbations
    
    layers = sorted(embedding_sims.keys())
    early, final = layers[0], layers[-1]
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    
    panels = [
        ('DE (Ground Truth)', biological_sim, 'RdBu_r', 0, (-1, 1)),
        (f'Layer {early} (Early)', embedding_sims[early], 'viridis', None, (0.5, 1)),
        (f'Layer {best_layer} (Best)', embedding_sims[best_layer], 'viridis', None, (0.5, 1)),
        (f'Layer {final} (Final)', embedding_sims[final], 'viridis', None, (0.5, 1))
    ]
    
    for ax, (title, mat, cmap, center, vrange) in zip(axes, panels):
        subset = mat.loc[display, display]
        sns.heatmap(subset, ax=ax, cmap=cmap, center=center,
                   vmin=vrange[0], vmax=vrange[1],
                   xticklabels=False, yticklabels=False,
                   cbar_kws={'label': 'Similarity'})
        ax.set_title(title, fontweight='bold')
    
    plt.suptitle('Perturbation Similarity Structure Across Layers', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pathway_gap(
    embedding_sims: Dict[int, pd.DataFrame],
    pathway_dict: Dict[str, set],
    output_path: str
):
    """
    Figure 4: Within vs between pathway similarity gap.
    """
    results = []
    
    for layer, sim in sorted(embedding_sims.items()):
        within, between = [], []
        
        perturbations = list(sim.index)
        for i, p1 in enumerate(perturbations):
            pathways1 = pathway_dict.get(p1, set())
            if not pathways1:
                continue
                
            for p2 in perturbations[i+1:]:
                pathways2 = pathway_dict.get(p2, set())
                if not pathways2:
                    continue
                
                similarity = sim.loc[p1, p2]
                if pathways1 & pathways2:  # Shared pathway
                    within.append(similarity)
                else:
                    between.append(similarity)
        
        if within and between:
            results.append({
                'layer': layer,
                'gap': np.mean(within) - np.mean(between)
            })
    
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['layer'], df['gap'], 'o-', linewidth=2.5, markersize=8, color='darkgreen')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Within − Between Pathway Similarity', fontsize=12)
    ax.set_title('Pathway Structure Capture Across Layers', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()