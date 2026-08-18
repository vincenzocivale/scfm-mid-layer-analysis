"""Open an h5ad and report whether it meets the benchmark contract.

Usage:
    python scripts/dataset_downloads/verify.py <name> [<name> ...]
    python scripts/dataset_downloads/verify.py --all

Reads scripts/dataset_downloads/manifest.yaml.  For each named dataset:
  * if target file missing -> MISSING
  * else open with anndata.read_h5ad (backed='r' when possible) and report:
      n_obs, n_vars, present obs columns, missing required obs columns,
      whether `counts_state` looks consistent with X (heuristic: integer dtype
      / max value), and (perturbation only) presence of `obs_control_label`
      values in the perturb column.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "scripts/dataset_downloads/manifest.yaml"


def load_manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text())["datasets"]


def _counts_state_guess(X) -> str:
    # Sample a few values; if all integer-valued and max > 50 -> raw_counts;
    # if float with max < ~20 -> log1p_normalized; else unknown.
    try:
        head = X[:200]
        sample = head.toarray() if hasattr(head, "toarray") else np.asarray(head)
    except Exception:
        return "err: cannot slice X"
    if sample.size == 0:
        return "empty"
    is_int = np.all(np.equal(np.mod(sample, 1), 0))
    mx = float(sample.max())
    if is_int and mx > 50:
        return "raw_counts"
    if not is_int and mx < 25:
        return "log1p_normalized"
    return "unknown"


def verify_one(name: str, entry: dict) -> dict:
    target = REPO / entry["target"]
    result = {"name": name, "task": entry["task"], "target": str(target)}
    if not target.exists():
        result["status"] = "MISSING"
        return result
    try:
        a = ad.read_h5ad(target, backed="r")
    except Exception as e:  # noqa: BLE001
        result["status"] = f"READ_ERROR: {e}"
        return result
    result["n_obs"] = int(a.n_obs)
    result["n_vars"] = int(a.n_vars)
    obs_cols = list(a.obs.columns)
    required = entry.get("obs_required", [])
    missing = [c for c in required if c not in obs_cols]
    result["obs_required"] = required
    result["obs_missing"] = missing
    result["obs_sample"] = obs_cols[:15]
    result["counts_state_declared"] = entry.get("counts_state", "?")
    try:
        result["counts_state_guess"] = _counts_state_guess(a.X)
    except Exception as e:  # noqa: BLE001
        result["counts_state_guess"] = f"err: {e}"
    if entry["task"] == "perturbation":
        control = entry.get("obs_control_label")
        if required and required[0] in obs_cols and control is not None:
            vals = set(a.obs[required[0]].astype(str).unique()[:5000])
            result["control_label_present"] = control in vals
    result["status"] = "OK" if not missing else "OBS_MISSING"
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    manifest = load_manifest()
    names = list(manifest.keys()) if args.all else args.names
    if not names:
        p.error("pass dataset names or --all")

    rows = []
    for n in names:
        if n not in manifest:
            print(f"[{n}] not in manifest", file=sys.stderr)
            continue
        rows.append(verify_one(n, manifest[n]))

    # Compact print
    for r in rows:
        head = f"[{r['status']:>13s}] {r['name']:<20s} task={r['task']:<14s}"
        if r["status"] == "MISSING":
            print(head + f" -> {r['target']} (not downloaded yet)")
            continue
        if r["status"].startswith("READ_ERROR"):
            print(head + f" {r['status']}")
            continue
        extra = (f" n_obs={r['n_obs']:>8d} n_vars={r['n_vars']:>6d}"
                 f" counts: declared={r['counts_state_declared']}"
                 f" guess={r['counts_state_guess']}")
        if r.get("obs_missing"):
            extra += f"  MISSING obs cols: {r['obs_missing']}"
        else:
            extra += f"  obs ok: {r['obs_required']}"
        if "control_label_present" in r:
            extra += f"  control_in_obs={r['control_label_present']}"
        print(head + extra)


if __name__ == "__main__":
    main()
