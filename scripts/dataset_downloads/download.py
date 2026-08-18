"""Download datasets declared in manifest.yaml.

Source policy on this server: GEO + HuggingFace only.

Usage:
    python scripts/dataset_downloads/download.py <name> [<name> ...]
    python scripts/dataset_downloads/download.py --task classification
    python scripts/dataset_downloads/download.py --all
    python scripts/dataset_downloads/download.py --list

For `method: geo` the supplementary files declared in `files:` are downloaded
into `data/raw/<task>/_geo/<accession>/`.  The downloader does NOT attempt
to assemble them into the final h5ad — that lives in per-dataset assembler
scripts under `scripts/dataset_downloads/assemble_<name>.py`, because the
metadata layout varies dataset by dataset.

For `method: huggingface` we call snapshot_download(repo_type='dataset') and
copy any *.h5ad found into the manifest target.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "scripts/dataset_downloads/manifest.yaml"

GEO_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"


def load_manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text())["datasets"]


# --------------------------------------------------------------------------- helpers

def _http_download(url: str, dest: Path, *, force: bool = False) -> Path:
    """Stream-download with Range-resume."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"   exists, skipping: {dest.name}")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "Mozilla/5.0 (scfm-eval downloader)"}
    mode = "wb"
    pos = 0
    if part.exists():
        pos = part.stat().st_size
        headers["Range"] = f"bytes={pos}-"
        mode = "ab"
        print(f"   resuming at {pos / 1e6:.1f} MB")

    with requests.get(url, stream=True, headers=headers, timeout=120,
                      allow_redirects=True) as r:
        if r.status_code == 416:        # range not satisfiable -> already complete
            shutil.move(part, dest)
            return dest
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + pos
        done = pos
        next_report = pos + 100 * 1024 * 1024
        with open(part, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    pct = 100 * done / total if total else 0
                    print(f"   {done / 1e6:>9.1f} MB"
                          + (f" / {total / 1e6:.1f} MB ({pct:4.1f}%)"
                             if total else ""))
                    next_report = done + 100 * 1024 * 1024
    shutil.move(part, dest)
    return dest


# --------------------------------------------------------------------------- methods

def _geo_url(acc: str, fname: str) -> str:
    # FTP layout: GSE98nnn/GSE98638/suppl/<file>
    bucket = acc[:-3] + "nnn"
    return f"{GEO_FTP_BASE}/{bucket}/{acc}/suppl/{fname}"


def download_geo(name: str, entry: dict, *, force: bool):
    acc = entry["source"]
    files = entry.get("files", [])
    out_dir = REPO / "data" / "raw" / entry["task"] / "_geo" / acc
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"-> [geo] {name}: {acc} -> {out_dir}")
    if not files:
        print(f"   manifest lists no files for {acc}.  Browse "
              f"{GEO_FTP_BASE}/{acc[:-3]}nnn/{acc}/suppl/ and populate "
              f"the `files:` list.")
        return
    for fn in files:
        url = _geo_url(acc, fn)
        try:
            _http_download(url, out_dir / fn, force=force)
        except Exception as e:  # noqa: BLE001
            print(f"   !! {fn} FAILED: {e}")
    print(f"   downloaded {len(files)} file(s) to {out_dir}")
    target = REPO / entry["target"]
    print(f"   NOTE: assemble {target.name} with the dataset-specific script")


def download_huggingface(name: str, entry: dict, *, force: bool):
    target = REPO / entry["target"]
    repo_id = entry["source"]
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"-> [huggingface] {name}: {repo_id}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("   huggingface_hub not installed — pip install huggingface_hub")
        return
    cache_dir = target.parent / f".{name}_hf"
    local = snapshot_download(repo_id=repo_id, repo_type="dataset",
                              allow_patterns=["*.h5ad", "*.h5"],
                              local_dir=str(cache_dir))
    cand = sorted(Path(local).rglob("*.h5ad"))
    if not cand:
        print(f"   no h5ad found under {local}")
        print(f"   contents: {[p.name for p in Path(local).rglob('*') if p.is_file()][:10]}")
        return
    if target.exists() and not force:
        print(f"   target exists, skipping copy: {target.name}")
    else:
        shutil.copy(cand[0], target)
        print(f"   copied {cand[0].name} -> {target}")


DISPATCH = {
    "geo": download_geo,
    "huggingface": download_huggingface,
}


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--task", choices=["classification", "pseudotime",
                                      "perturbation"])
    p.add_argument("--list", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    manifest = load_manifest()
    if args.list:
        for n, e in manifest.items():
            tgt = REPO / e["target"]
            mark = "✓" if tgt.exists() else " "
            print(f"  [{mark}] {n:<14s} task={e['task']:<14s} method={e['method']:<11s}"
                  f"  src={e['source']}")
        return

    if args.task:
        names = [n for n, e in manifest.items() if e["task"] == args.task]
    elif args.all:
        names = list(manifest.keys())
    else:
        names = args.names

    if not names:
        p.error("pass dataset names, --task, --all, or --list")

    for n in names:
        if n not in manifest:
            print(f"!! {n} not in manifest", file=sys.stderr)
            continue
        e = manifest[n]
        fn = DISPATCH.get(e["method"])
        if not fn:
            print(f"!! {n} unknown method {e['method']}", file=sys.stderr)
            continue
        try:
            fn(n, e, force=args.force)
        except Exception as exc:  # noqa: BLE001
            print(f"!! {n} FAILED: {exc}")


if __name__ == "__main__":
    main()
