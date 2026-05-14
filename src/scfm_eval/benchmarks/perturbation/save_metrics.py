import os
import pandas as pd

def save_metrics(metrics_dict, out_dir, filename_prefix: str):
    """
    Salva le metriche in modo organizzato, usando un prefisso per i nomi dei file.
    Args:
        metrics_dict: dict con chiavi = nome metrica, valori = DataFrame.
        out_dir: directory di output.
        filename_prefix: Prefisso per i file di output (es. nome del file h5ad).
    """
    os.makedirs(out_dir, exist_ok=True)
    for metric_name, df in metrics_dict.items():
        out_path = os.path.join(out_dir, f"{filename_prefix}_{metric_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {metric_name} to {out_path}")
