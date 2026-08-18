from __future__ import annotations

import numpy as np
import pandas as pd


GROUP = ["dataset", "model", "model_size", "task", "metric", "split"]


def add_final_layer_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    """Annotate every model-layer metric relative to that model's final layer.

    `signed_gain_vs_final > 0` always means the intermediate layer is better,
    regardless of whether the native metric is maximized or minimized.
    """
    df = metrics.copy()

    model_rows = df[
        (df["representation"] == "model_layer")
        & df["layer"].notna()
        & df["n_layers_total"].notna()
    ].copy()
    if model_rows.empty:
        df["final_value"] = np.nan
        df["raw_delta_vs_final"] = np.nan
        df["signed_gain_vs_final"] = np.nan
        df["relative_gain_vs_final"] = np.nan
        df["beats_final"] = False
        return df

    model_rows["is_final"] = (
        model_rows["layer"].astype(int)
        == model_rows["n_layers_total"].astype(int) - 1
    )

    finals = (
        model_rows[model_rows["is_final"]]
        .loc[:, GROUP + ["value"]]
        .rename(columns={"value": "final_value"})
    )

    if finals.duplicated(GROUP).any():
        dup = finals[finals.duplicated(GROUP, keep=False)]
        raise ValueError(f"Multiple final-layer rows for the same metric group:\n{dup}")

    out = df.merge(finals, on=GROUP, how="left")
    out["raw_delta_vs_final"] = out["value"] - out["final_value"]
    direction = np.where(out["higher_is_better"].astype(bool), 1.0, -1.0)
    out["signed_gain_vs_final"] = out["raw_delta_vs_final"] * direction
    denom = out["final_value"].abs().replace(0, np.nan)
    out["relative_gain_vs_final"] = out["signed_gain_vs_final"] / denom
    out["beats_final"] = out["signed_gain_vs_final"] > 0
    return out
