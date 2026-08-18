#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from scfm_eval.v2.compare import add_final_layer_comparison
from scfm_eval.v2.results import load_metric_tables


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("artifacts/results"))
    p.add_argument("--output", type=Path, default=Path("artifacts/summary/metrics_all.csv"))
    args = p.parse_args()

    df = load_metric_tables(args.results_root)
    if df.empty:
        raise SystemExit(f"No metrics found below {args.results_root}")

    df = add_final_layer_comparison(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    winners = df[
        (df["representation"] == "model_layer")
        & (df["beats_final"])
    ].copy()
    print(f"Wrote {len(df)} metric rows -> {args.output}")
    if not winners.empty:
        cols = [
            "dataset", "model", "model_size", "task", "metric",
            "layer", "relative_depth", "value", "final_value",
            "signed_gain_vs_final",
        ]
        print("\nIntermediate layers beating the final layer:")
        print(
            winners[cols]
            .sort_values("signed_gain_vs_final", ascending=False)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
