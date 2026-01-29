import pandas as pd
import matplotlib.pyplot as plt
import re


# =========================
# Utilities
# =========================

def _extract_layer_index(layer_name: str) -> int:
    """
    Estrae l'indice numerico dal nome del layer (es. X_layer_12 -> 12).
    """
    match = re.search(r"(\d+)$", layer_name)
    if match is None:
        raise ValueError(f"Impossibile estrarre indice da: {layer_name}")
    return int(match.group(1))


def load_trajectory_metrics(csv_path: str) -> pd.DataFrame:
    """
    Carica il CSV e restituisce un DataFrame ordinato per layer.
    """
    df = pd.read_csv(csv_path)
    df["layer_idx"] = df["Layer"].apply(_extract_layer_index)
    df = df.sort_values("layer_idx").reset_index(drop=True)
    return df


# =========================
# Plot functions (one metric = one function)
# =========================

def plot_pseudotime_correlation(
    df: pd.DataFrame,
    save_path: str = None,
):
    """
    Plot: preservazione dell'ordinamento pseudotime.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(
        df["layer_idx"],
        df["Pseudotime_Corr"],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Layer")
    plt.ylabel("Spearman correlation")
    plt.title("Pseudotime Ordering Preservation")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_neighborhood_overlap(
    df: pd.DataFrame,
    save_path: str = None,
):
    """
    Plot: preservazione della continuità locale (kNN overlap).
    """
    plt.figure(figsize=(6, 4))
    plt.plot(
        df["layer_idx"],
        df["Neighborhood_Overlap"],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Layer")
    plt.ylabel("Mean kNN overlap")
    plt.title("Local Continuity Preservation")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_global_geometry_alignment(
    df: pd.DataFrame,
    save_path: str = None,
):
    """
    Plot: allineamento geometrico globale (distanza vs pseudotime).
    """
    plt.figure(figsize=(6, 4))
    plt.plot(
        df["layer_idx"],
        df["Global_Geom_Corr"],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Layer")
    plt.ylabel("Spearman correlation")
    plt.title("Global Geometry Alignment")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
