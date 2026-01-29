import os
import pandas as pd

def save_metrics(metrics_dict, out_dir):
    """
    Salva le metriche in modo organizzato per ciascun layer.
    Args:
        metrics_dict: dict con chiavi = nome metrica, valori = DataFrame con colonna 'layer'
        out_dir: directory di output
    """
    os.makedirs(out_dir, exist_ok=True)
    for metric_name, df in metrics_dict.items():
        out_path = os.path.join(out_dir, f"{metric_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {metric_name} to {out_path}")
