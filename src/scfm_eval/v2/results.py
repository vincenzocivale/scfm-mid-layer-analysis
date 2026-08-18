from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .specs import MetricRecord, RunSpec


RESULT_COLUMNS = [
    "dataset",
    "model",
    "model_size",
    "task",
    "representation",
    "layer",
    "n_layers_total",
    "relative_depth",
    "metric",
    "value",
    "higher_is_better",
    "n_obs",
    "split",
    "notes",
]


class ResultWriter:
    """One run directory, one canonical long-format metric table."""

    def __init__(self, root: Path, run: RunSpec):
        self.root = Path(root)
        self.run = run
        self.run_dir = self.root / run.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.meta_path = self.run_dir / "run.json"

    def write(self, records: Iterable[MetricRecord], metadata: dict | None = None) -> Path:
        rows = [r.to_dict() for r in records]
        if not rows:
            raise ValueError("No metric records to write")

        df = pd.DataFrame(rows)
        for col in RESULT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[RESULT_COLUMNS]

        fd, tmp = tempfile.mkstemp(prefix=".metrics.", suffix=".csv", dir=self.run_dir)
        os.close(fd)
        try:
            df.to_csv(tmp, index=False)
            os.replace(tmp, self.metrics_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

        meta = {
            "run_id": self.run.run_id,
            "dataset": self.run.dataset,
            "model": self.run.model,
            "model_size": self.run.model_size,
            "task": self.run.task,
            "artifact_dir": self.run.artifact_dir,
            "dataset_path": self.run.dataset_path,
        }
        if metadata:
            meta.update(metadata)
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        return self.metrics_path


def load_metric_tables(results_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(Path(results_root).glob("*/metrics.csv")):
        df = pd.read_csv(path)
        df["source_file"] = path.as_posix()
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS + ["source_file"])
    return pd.concat(frames, ignore_index=True)
