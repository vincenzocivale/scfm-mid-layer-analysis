import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns

def natural_sort_key(s):
    """Ordina stringhe come Layer_1, Layer_2, ..., Layer_10 in modo naturale."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def load_results(results_path):
    """
    Carica un file CSV di risultati della pipeline di classificazione.
    Restituisce un DataFrame ordinato per layer.
    """
    df = pd.read_csv(results_path)
    if 'Layer' in df.columns:
        # Non ordinare qui, lo facciamo dopo aver separato la baseline
        df['Layer'] = df['Layer'].astype(str)
    return df

def plot_metric_by_layer(
    results_path,
    metric_col,
    model_name=None,
    title=None,
    ylabel=None,
    ax=None,
    plot_type='line' # 'line' or 'bar'
):
    """
    Crea un plot di una metrica specifica per ogni layer, con una linea per la baseline.
    """
    df = load_results(results_path)

    if metric_col not in df.columns:
        raise ValueError(f"Metrica '{metric_col}' non trovata nel file dei risultati.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Separa baseline e layer del modello
    baseline_df = df[df['Layer'] == 'Baseline_PCA']
    layers_df = df[df['Layer'] != 'Baseline_PCA'].copy()
    
    # Ordina i layer del modello in modo naturale
    if not layers_df.empty:
        layers_df.sort_values(by='Layer', key=lambda x: x.map(natural_sort_key), inplace=True)
        x = layers_df['Layer']
        y = layers_df[metric_col]

        if plot_type == 'line':
            ax.plot(x, y, marker='o', linestyle='-', label=model_name or 'Model Embeddings')
        elif plot_type == 'bar':
            sns.barplot(x=x, y=y, ax=ax, color=sns.color_palette("muted")[0])
        else:
            raise ValueError("plot_type deve essere 'line' o 'bar'")
    
    # Plotta la baseline come linea orizzontale
    if not baseline_df.empty:
        baseline_value = baseline_df[metric_col].iloc[0]
        ax.axhline(baseline_value, color='red', linestyle='--', linewidth=2, label='Baseline (PCA)')

    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel(ylabel or metric_col, fontsize=13)
    ax.set_title(title or f'{metric_col} by Layer', fontsize=15)
    
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.legend(loc='best', frameon=False)
        
    plt.tight_layout()

    if 'fig' in locals():
        plt.show()


def plot_all_metrics(results_path, model_name=None):
    """
    Crea un pannello con i grafici per tutte le metriche principali.
    """
    metrics = {
        'Accuracy': 'Accuracy',
        'F1_macro': 'Macro F1-Score',
        'Silhouette_Score': 'Silhouette Score',
        'ARI': 'Adjusted Rand Index'
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, (metric, label) in enumerate(metrics.items()):
        plot_metric_by_layer(
            results_path,
            metric_col=metric,
            model_name=model_name,
            title=f'Layer-wise {label}',
            ylabel=label,
            ax=axes[i],
            plot_type='line'
        )
    
    fig.suptitle(f'Classification Evaluation for {model_name or "Model"}', fontsize=20, y=1.03)
    plt.tight_layout()
    plt.show()

# Esempio di utilizzo (da notebook o script):
# from cell_type_classification_benchmark.plotting import plot_all_metrics
# plot_all_metrics(
#     results_path="data/classification_results/my_dataset_classification_results.csv",
#     model_name="My Model"
# )