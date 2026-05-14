"""
Driver: read config/experiments.yaml and dispatch the full grid of
(dataset, model[, model_size], task) runs.

Each run is executed in a fresh subprocess for clean GPU memory isolation
between models. A CSV manifest of attempted runs is appended to
data/run_manifest.csv with status/timestamps/error info.

Usage examples
--------------
    # Run the whole grid declared in config/experiments.yaml
    python scripts/run_experiment_grid.py

    # Dry-run: print the plan, don't execute
    python scripts/run_experiment_grid.py --dry-run

    # Only Tahoe-1b on the brain dataset, classification only
    python scripts/run_experiment_grid.py --filter model=tahoe --filter model_size=1b --filter dataset=brain --filter task=classification

    # Force re-run even if output already exists
    python scripts/run_experiment_grid.py --no-resume
"""
import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable  # use the same interpreter that runs the driver

TASK_LAUNCHERS = {
    'classification': REPO_ROOT / 'scripts' / 'run_cell_type_classification.py',
    'pseudotime': REPO_ROOT / 'scripts' / 'run_pseudotime_analysis.py',
    'perturbation': REPO_ROOT / 'scripts' / 'run_perturbation_analysis.py',
}

TASK_OUTPUT_DIRS = {
    'classification': REPO_ROOT / 'data' / 'classification_results',
    'pseudotime': REPO_ROOT / 'data' / 'pseudotime_results',
    'perturbation': REPO_ROOT / 'data' / 'perturbation_metrics',
}

TASK_OUTPUT_SUFFIX = {
    'classification': '_classification_results.csv',
    'pseudotime': '_results.csv',
    'perturbation': '_semantic_similarity.csv',   # representative output
}


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def embeddings_dir_for(dataset_path: Path, model: str, model_size: Optional[str]) -> Path:
    suffix = f"_{model}" if model_size in (None, '', 'default') else f"_{model}_{model_size}"
    return REPO_ROOT / 'data' / 'embeddings' / f"{dataset_path.stem}{suffix}"


def merged_h5ad_for(embeddings_dir: Path) -> Path:
    return embeddings_dir / f"{embeddings_dir.name}.h5ad"


def task_output_path(task: str, merged_h5ad: Path) -> Path:
    return TASK_OUTPUT_DIRS[task] / f"{merged_h5ad.stem}{TASK_OUTPUT_SUFFIX[task]}"


def parse_filters(filter_args: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for f in filter_args:
        if '=' not in f:
            sys.exit(f"--filter expects key=value, got {f!r}")
        k, v = f.split('=', 1)
        out[k.strip()] = v.strip()
    return out


def matches_filter(run: dict, task: str, filters: Dict[str, str]) -> bool:
    for k, v in filters.items():
        if k == 'task':
            if task != v:
                return False
        elif k == 'model_size':
            if str(run.get('model_size', '')) != v:
                return False
        elif str(run.get(k, '')) != v:
            return False
    return True


def run_subprocess(cmd: List[str], log_prefix: str) -> tuple:
    """Run cmd in a fresh subprocess. Returns (returncode, elapsed_seconds, error_msg)."""
    print(f"\n[{log_prefix}] $ {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return proc.returncode, elapsed, f"exit code {proc.returncode}"
        return 0, elapsed, ''
    except Exception as e:
        return -1, time.time() - t0, repr(e)


def append_manifest(manifest_path: Path, row: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    exists = manifest_path.exists()
    with open(manifest_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def ensure_extraction(dataset_path: Path, model: str, model_size: Optional[str],
                       extraction_defaults: dict, resume: bool, dry: bool) -> Path:
    """Run stage 1 if needed and return the merged-h5ad path expected by stage 2."""
    out_dir = embeddings_dir_for(dataset_path, model, model_size)
    merged = merged_h5ad_for(out_dir)

    if resume and merged.exists():
        print(f"  ✓ embeddings already present: {merged.relative_to(REPO_ROOT)}")
        return merged

    # Stage 1: chunked extraction
    defaults = extraction_defaults.get(model, {})
    env = {
        'MODELS': model,
        'CHUNK_SIZE': str(defaults.get('chunk_size', 50000)),
        'BATCH_SIZE': str(defaults.get('batch_size', 1)),
    }
    if model == 'tahoe' and model_size:
        env['TAHOE_MODEL_SIZE'] = model_size

    extra = ['--no-fp16'] if defaults.get('fp16') is False else []
    cmd = ['bash', str(REPO_ROOT / 'run_embedding_extraction.sh'), str(dataset_path), *extra]

    if dry:
        print(f"  [DRY] would extract -> {out_dir}/")
        print(f"        env: {env}")
        print(f"        cmd: {' '.join(cmd)}")
        return merged

    full_env = os.environ.copy()
    full_env.update(env)
    print(f"\n[extract] {model}{('-'+model_size) if model_size else ''} on {dataset_path.name}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=full_env)
    if proc.returncode != 0:
        raise RuntimeError(f"extraction failed for {dataset_path.name} / {model}/{model_size}")
    print(f"  extraction done in {time.time()-t0:.1f}s")

    # Stage 1.5: merge chunks into a single h5ad if not already there
    if not merged.exists():
        chunks = sorted(out_dir.glob('chunk_*.h5ad'))
        if not chunks:
            raise RuntimeError(f"no chunks produced in {out_dir}")
        merge_cmd = [PY, str(REPO_ROOT / 'scripts' / 'merge_chunks.py'),
                     '--input-dir', str(out_dir), '--output', str(merged)]
        rc, _, err = run_subprocess(merge_cmd, log_prefix='merge')
        if rc != 0:
            raise RuntimeError(f"merge_chunks failed: {err}")

    return merged


def build_task_cmd(task: str, merged_h5ad: Path, task_cfg: dict) -> List[str]:
    launcher = TASK_LAUNCHERS[task]
    if task == 'classification':
        return [PY, str(launcher),
                '--input', str(merged_h5ad),
                '--cell_type_column', task_cfg['cell_type_column']]
    if task == 'pseudotime':
        return [PY, str(launcher),
                '--input', str(merged_h5ad),
                '--time_column', task_cfg['time_column']]
    if task == 'perturbation':
        cmd = [PY, str(launcher),
               '--input', str(merged_h5ad),
               '--output_dir', str(TASK_OUTPUT_DIRS['perturbation']),
               '--perturb_key', task_cfg['perturb_key'],
               '--control_label', task_cfg['control_label']]
        if 'dose_key' in task_cfg:
            cmd += ['--dose_key', task_cfg['dose_key']]
        if 'pathway_json' in task_cfg:
            cmd += ['--pathway_json', task_cfg['pathway_json']]
        return cmd
    raise ValueError(f"Unknown task: {task}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'config' / 'experiments.yaml')
    parser.add_argument('--datasets', type=Path, default=REPO_ROOT / 'config' / 'datasets.yaml')
    parser.add_argument('--filter', action='append', default=[],
                        help="Filter runs, e.g. --filter dataset=brain --filter task=classification. Repeatable.")
    parser.add_argument('--no-resume', action='store_true',
                        help="Re-run every cell, even if its output CSV is already present.")
    parser.add_argument('--dry-run', action='store_true', help="Print the plan; do not execute.")
    parser.add_argument('--manifest', type=Path, default=REPO_ROOT / 'data' / 'run_manifest.csv')
    args = parser.parse_args()

    runs_cfg = load_yaml(args.config)
    datasets_cfg = load_yaml(args.datasets)['datasets']
    filters = parse_filters(args.filter)
    extraction_defaults = runs_cfg.get('extraction_defaults', {})

    # Expand each run × task into a flat plan
    plan: List[dict] = []
    for run in runs_cfg.get('runs', []):
        ds_key = run['dataset']
        if ds_key not in datasets_cfg:
            print(f"[skip] unknown dataset {ds_key!r} in run {run}")
            continue
        for task in run.get('tasks', []):
            if not matches_filter(run, task, filters):
                continue
            if task not in datasets_cfg[ds_key].get('tasks', {}):
                print(f"[skip] dataset {ds_key!r} declares no config for task {task!r}")
                continue
            plan.append({
                'dataset': ds_key,
                'model': run['model'],
                'model_size': run.get('model_size'),
                'task': task,
                'task_cfg': datasets_cfg[ds_key]['tasks'][task],
                'dataset_path': REPO_ROOT / datasets_cfg[ds_key]['path'],
            })

    if not plan:
        sys.exit("Plan is empty — check --filter and config/experiments.yaml.")

    print(f"Plan: {len(plan)} cells to run")
    for p in plan:
        size_str = f"-{p['model_size']}" if p['model_size'] else ''
        print(f"  - {p['dataset']:>14s} × {p['model']}{size_str:<6s} × {p['task']}")

    if args.dry_run:
        return

    resume = not args.no_resume
    for cell in plan:
        ds_key = cell['dataset']
        model = cell['model']
        size = cell['model_size']
        task = cell['task']
        run_id = f"{ds_key}/{model}{('-'+size) if size else ''}/{task}"
        started = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')

        try:
            merged_h5ad = ensure_extraction(
                cell['dataset_path'], model, size, extraction_defaults,
                resume=resume, dry=False,
            )

            out = task_output_path(task, merged_h5ad)
            if resume and out.exists():
                print(f"\n[skip] {run_id} — output already exists: {out.relative_to(REPO_ROOT)}")
                append_manifest(args.manifest, {
                    'run_id': run_id, 'dataset': ds_key, 'model': model,
                    'model_size': size or '', 'task': task,
                    'status': 'skipped', 'started_at': started, 'finished_at': started,
                    'elapsed_s': 0, 'error': '',
                })
                continue

            cmd = build_task_cmd(task, merged_h5ad, cell['task_cfg'])
            rc, elapsed, err = run_subprocess(cmd, log_prefix=run_id)
            status = 'ok' if rc == 0 else 'fail'

        except Exception as e:
            rc, elapsed, err, status = -1, 0.0, repr(e), 'fail'

        finished = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
        append_manifest(args.manifest, {
            'run_id': run_id, 'dataset': ds_key, 'model': model,
            'model_size': size or '', 'task': task,
            'status': status, 'started_at': started, 'finished_at': finished,
            'elapsed_s': round(elapsed, 1), 'error': err,
        })

    print("\nGrid complete. Manifest:", args.manifest.relative_to(REPO_ROOT))


if __name__ == '__main__':
    main()
