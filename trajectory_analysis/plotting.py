"""
Funzioni per la visualizzazione dei risultati trajectory evaluation layer-wise.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric_vs_layer(df, metric, ylabel=None, title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(x='Layer', y=metric, data=df, marker='o', ax=ax)
    ax.set_xlabel('Layer')
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f'{metric} vs Layer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if ax is None:
        plt.show()

def plot_multi_metrics(df, metrics, ylabel=None, title=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in metrics:
        sns.lineplot(x='Layer', y=metric, data=df, marker='o', label=metric, ax=ax)
    ax.set_xlabel('Layer')
    ax.set_ylabel(ylabel or 'Metric value')
    ax.set_title(title or 'Metrics vs Layer')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_metric_bar(df, metric, ylabel=None, title=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x='Layer', y=metric, data=df, ax=ax)
    ax.set_xlabel('Layer')
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f'{metric} per Layer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_heatmap_metrics(df, metrics, title=None):
    data = df.set_index('Layer')[metrics]
    plt.figure(figsize=(len(metrics)*1.5+3, len(df)*0.4+2))
    sns.heatmap(data, annot=True, cmap='viridis', fmt='.2f')
    plt.title(title or 'Metrics Heatmap')
    plt.ylabel('Layer')
    plt.xlabel('Metric')
    plt.tight_layout()
    plt.show()

def load_trajectory_results(filepath):
    return pd.read_csv(filepath)
