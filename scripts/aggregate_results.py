"""
Walk per-task result directories and consolidate every CSV into a single
long-format `data/results_all.csv` with columns:

    dataset, model, model_size, task, layer, n_layers_total,
    relative_depth, metric, value, source_file

Parsing strategy
----------------
Filenames are of the form `<input_stem>_<task suffix>.csv` where `<input_stem>`
encodes the (dataset, model) tuple. We parse `<input_stem>` against the model
registry (config/models.yaml) and the dataset registry (config/datasets.yaml)
so that adding new models/datasets is a config-only change.

Layer column
------------
- Real layers are reported as their integer index from `X_layer_{i}`.
- The classification baseline `Baseline_PCA` is reported as layer=-1 (sentinel).
- `relative_depth = layer / (n_layers_total - 1)` when n_layers_total is known
  (looked up in models.yaml or recorded in `adata.uns['layer_embeddings']`);
  NaN otherwise (e.g. for the baseline).
"""
import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

TASK_DIRS = {
    'classification': REPO_ROOT / 'data' / 'classification_results',
    'pseudotime': REPO_ROOT / 'data' / 'pseudotime_results',
    'perturbation': REPO_ROOT / 'data' / 'perturbation_metrics',
}

# Filename suffixes used by each pipeline. The aggregator strips these to
# recover <input_stem>, then parses <input_stem> as <dataset_key>_<model_tag>.
TASK_SUFFIXES = {
    'classification': ['_classification_results'],
    'pseudotime': ['_results'],
    # perturbation produces multiple CSVs per run (semantic_similarity,
    # dose_response, pathway_clustering, reference_de_similarity)
    'perturbation': [
        '_semantic_similarity', '_dose_response', '_pathway_clustering',
    ],
}


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_tag_patterns(models_cfg: dict) -> List[Tuple[re.Pattern, str, Optional[str]]]:
    """
    Returns a list of (regex, model_key, size) tuples to match a model tag
    inside an input_stem. Longer tags first to disambiguate (tahoe_1b before tahoe).
    """
    patterns = []
    for model_key, spec in models_cfg.get('models', {}).items():
        sizes = spec.get('sizes', [None]) or [None]
        for size in sizes:
            if size in (None, 'default'):
                tag = model_key
                size_val = None
            else:
                tag = f"{model_key}_{size}"
                size_val = size
            # Tag may also appear as "<model>_embeddings" in legacy output names.
            patterns.append((re.compile(rf"_{re.escape(tag)}(?:_embeddings)?$"), model_key, size_val))
    # Sort by tag-length desc so "tahoe_1b" matches before "tahoe".
    patterns.sort(key=lambda p: -len(p[0].pattern))
    return patterns


def parse_input_stem(stem: str, datasets_cfg: dict, model_patterns: list) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (dataset_key, model, model_size) parsed from an input file stem.
    """
    for pat, model_key, model_size in model_patterns:
        m = pat.search(stem)
        if m:
            dataset_part = stem[:m.start()]
            # Match dataset against the registry; we allow input_stem to be a
            # strict prefix of the dataset filename stem.
            for ds_key, ds_spec in datasets_cfg.get('datasets', {}).items():
                ds_filename = Path(ds_spec['path']).stem
                if dataset_part == ds_filename or dataset_part == ds_key:
                    return ds_key, model_key, model_size
            # Dataset not in registry — return raw dataset_part so user can fix it.
            return dataset_part, model_key, model_size
    return None, None, None


def parse_layer_column(layer_str) -> int:
    """`X_layer_5` -> 5; bare int (or str of int) -> int; anything else -> -1 sentinel (baseline)."""
    s = str(layer_str)
    m = re.match(r'X_layer_(\d+)', s)
    if m:
        return int(m.group(1))
    if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
        return int(s)
    return -1


# Columns we drop before melting: garbage from CSV serialization or duplicate
# layer encodings already captured by the canonical layer column.
_DROP_COLS = {'Unnamed: 0', 'layer_idx', 'p_value'}


def melt_metrics(df: pd.DataFrame, layer_col_candidates=('Layer', 'layer')) -> pd.DataFrame:
    """Long-format: one row per (Layer, metric)."""
    layer_col = next((c for c in layer_col_candidates if c in df.columns), None)
    if layer_col is None:
        return pd.DataFrame()
    keep = [c for c in df.columns if c not in _DROP_COLS]
    df = df[keep]
    long = df.melt(id_vars=[layer_col], var_name='metric', value_name='value')
    long = long.rename(columns={layer_col: 'layer_label'})
    long['layer'] = long['layer_label'].map(parse_layer_column)
    return long


def aggregate(repo_root: Path, output_path: Path):
    models_cfg = load_yaml(repo_root / 'config' / 'models.yaml')
    datasets_cfg = load_yaml(repo_root / 'config' / 'datasets.yaml')
    model_patterns = build_model_tag_patterns(models_cfg)

    rows = []
    for task, task_dir in TASK_DIRS.items():
        if not task_dir.is_dir():
            print(f"[skip] {task_dir} not present")
            continue
        for csv_path in sorted(task_dir.glob('*.csv')):
            # Strip the task-specific suffix to recover <input_stem>
            stem = csv_path.stem
            metric_suffix = None
            for sfx in TASK_SUFFIXES[task]:
                if stem.endswith(sfx):
                    stem = stem[: -len(sfx)]
                    metric_suffix = sfx.lstrip('_')
                    break
            if metric_suffix is None and task == 'perturbation':
                # Skip files we don't recognize (e.g. reference_de_similarity).
                print(f"[skip] {csv_path.name} — unrecognized perturbation output")
                continue

            ds_key, model_key, size = parse_input_stem(stem, datasets_cfg, model_patterns)
            if model_key is None:
                print(f"[skip] {csv_path.name} — could not parse (dataset, model) from {stem!r}")
                continue

            df = pd.read_csv(csv_path)
            long = melt_metrics(df)
            if long.empty:
                print(f"[skip] {csv_path.name} — no Layer column found")
                continue

            long['dataset'] = ds_key
            long['model'] = model_key
            long['model_size'] = size or ''
            long['task'] = task
            long['source_file'] = csv_path.relative_to(repo_root).as_posix()
            rows.append(long)

    if not rows:
        sys.exit("No result CSVs found. Have you run any benchmarks yet?")

    all_long = pd.concat(rows, ignore_index=True)

    # n_layers_total: prefer to look it up from models.yaml when possible.
    # For now we let it be NaN unless the user pre-populates it; the notebook
    # can derive it from the data itself (max(layer)+1 per (dataset, model)).
    n_layers = (
        all_long[all_long['layer'] >= 0]
        .groupby(['dataset', 'model', 'model_size'])['layer']
        .max()
        .add(1)
        .rename('n_layers_total')
        .reset_index()
    )
    all_long = all_long.merge(n_layers, on=['dataset', 'model', 'model_size'], how='left')
    all_long['relative_depth'] = (
        all_long['layer'] / (all_long['n_layers_total'] - 1)
    ).where(all_long['layer'] >= 0)

    cols = ['dataset', 'model', 'model_size', 'task', 'layer', 'layer_label',
            'n_layers_total', 'relative_depth', 'metric', 'value', 'source_file']
    all_long = all_long[cols].sort_values(['dataset', 'model', 'task', 'metric', 'layer']).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_long.to_csv(output_path, index=False)
    print(f"Wrote {len(all_long)} rows to {output_path}")
    print("Summary by (task, dataset, model):")
    print(all_long.groupby(['task', 'dataset', 'model', 'model_size']).size().to_string())


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-task result CSVs into a single long-format file.")
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'data' / 'results_all.csv')
    args = parser.parse_args()
    aggregate(REPO_ROOT, args.output)


if __name__ == '__main__':
    main()
