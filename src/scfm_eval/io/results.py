"""Standardized output paths and writers for per-task benchmark results.

Each task pipeline produces a CSV with one row per evaluated layer plus
optional baseline rows. The aggregator (`scripts/aggregate_results.py`) parses
filenames back into (dataset, model, task) so the naming convention here is
load-bearing — change with care.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


# Per-task output directory and filename suffix relative to repo root.
TASK_OUTPUT_DIRS = {
    'classification': Path('data/classification_results'),
    'pseudotime':     Path('data/pseudotime_results'),
    'perturbation':   Path('data/perturbation_metrics'),
}

TASK_SUFFIX = {
    'classification': '_classification_results.csv',
    'pseudotime':     '_results.csv',
    # perturbation writes multiple files via its own save_metrics helper; this
    # is the canonical "single representative" suffix used by the driver to
    # check whether a run has produced output.
    'perturbation':   '_semantic_similarity.csv',
}


def results_csv_path(task: str, input_stem: str, repo_root: Union[str, Path] = '.') -> Path:
    """Canonical output CSV path for a (task, input_stem) pair."""
    if task not in TASK_OUTPUT_DIRS:
        raise ValueError(f"Unknown task {task!r}. Known: {sorted(TASK_OUTPUT_DIRS)}")
    return Path(repo_root) / TASK_OUTPUT_DIRS[task] / f"{input_stem}{TASK_SUFFIX[task]}"


def write_results_csv(df: pd.DataFrame, task: str, input_stem: str,
                      *, repo_root: Union[str, Path] = '.', index: bool = False) -> Path:
    """Write a benchmark result DataFrame to the canonical location."""
    out = results_csv_path(task, input_stem, repo_root=repo_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=index)
    return out
