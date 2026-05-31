"""
Consolidate per-chunk h5ad files (output of stage 1) into a single h5ad
with X_layer_* in obsm, ready for the stage-2 benchmark pipelines.

Usage:
    python scripts/merge_chunks.py --input-dir data/embeddings/brain_dataset_scfoundation/
    python scripts/merge_chunks.py --input-dir <dir> --output <path>   # explicit output
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import anndata as ad
import scanpy as sc


def main():
    parser = argparse.ArgumentParser(description="Merge chunked embedding h5ads into one file.")
    parser.add_argument('--input-dir', type=Path, required=True,
                        help="Directory with chunk_*.h5ad files (output of run_embedding_extraction.sh).")
    parser.add_argument('--output', type=Path, default=None,
                        help="Output h5ad path. Defaults to <input-dir>/<input-dir-name>.h5ad")
    parser.add_argument('--pattern', type=str, default='*_chunk_*.h5ad',
                        help="Glob pattern for chunk files (default: *_chunk_*.h5ad).")
    parser.add_argument('--compression', type=str, default='gzip')
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input directory does not exist: {args.input_dir}")

    chunk_paths = sorted(args.input_dir.glob(args.pattern))
    if not chunk_paths:
        sys.exit(f"No chunks matching {args.pattern} in {args.input_dir}")

    print(f"Found {len(chunk_paths)} chunks in {args.input_dir}")

    out_path = args.output or (args.input_dir / f"{args.input_dir.name}.h5ad")
    if out_path in chunk_paths:
        sys.exit(f"Output path collides with a chunk file: {out_path}")

    adatas = []
    layer_meta = None
    for p in chunk_paths:
        print(f"  reading {p.name} ...", end=' ', flush=True)
        a = sc.read_h5ad(p)
        adatas.append(a)
        print(f"({a.n_obs} cells)")
        if layer_meta is None and 'layer_embeddings' in a.uns:
            layer_meta = dict(a.uns['layer_embeddings'])

    print(f"Concatenating {len(adatas)} chunks ...")
    merged = ad.concat(adatas, axis=0, join='outer', merge='same', uns_merge='same')

    # Restore layer-level metadata (concat drops chunk-specific bits anyway).
    if layer_meta is not None:
        layer_meta.pop('chunk_index', None)
        layer_meta.pop('cell_range', None)
        layer_meta['n_cells_total'] = int(merged.n_obs)
        merged.uns['layer_embeddings'] = layer_meta

    print(f"Writing {out_path} ({merged.n_obs} cells x {merged.n_vars} genes) ...")
    merged.write_h5ad(out_path, compression=args.compression)
    print("Done.")


if __name__ == '__main__':
    main()
