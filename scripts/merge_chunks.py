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
import h5py
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata.experimental import read_elem, write_elem


def _var_index(path):
    with h5py.File(path, 'r') as f:
        return read_elem(f['var']).index


def _x_kind(chunk_paths):
    """'dense' if every chunk's X is a dense dataset, 'sparse' if every chunk's
    X is a sparse group (CSR/CSC), or None if mixed (caller should fall back
    to in-memory concat)."""
    kinds = set()
    for p in chunk_paths:
        with h5py.File(p, 'r') as f:
            kinds.add('dense' if isinstance(f['X'], h5py.Dataset) else 'sparse')
    return kinds.pop() if len(kinds) == 1 else None


def _merge_metadata(chunk_paths):
    """Merge obs/var/uns across chunks via lightweight placeholder AnnData
    objects whose X is reduced to 0 (obs side) or 1 (var side) rows. This
    reuses anndata's exact `join`/`merge`/`uns_merge` semantics without ever
    loading the large X/obsm arrays.
    """
    obs_uns_adatas = []
    var_adatas = []
    n_obs_per_chunk = []
    layer_meta = None
    for p in chunk_paths:
        with h5py.File(p, 'r') as f:
            obs_i = read_elem(f['obs'])
            var_i = read_elem(f['var'])
            uns_i = dict(read_elem(f['uns'])) if 'uns' in f else {}
        n_obs_per_chunk.append(len(obs_i))
        if layer_meta is None and 'layer_embeddings' in uns_i:
            layer_meta = dict(uns_i['layer_embeddings'])
        obs_uns_adatas.append(ad.AnnData(X=np.zeros((len(obs_i), 0), dtype=np.float32), obs=obs_i, uns=uns_i))
        var_adatas.append(ad.AnnData(X=np.zeros((1, len(var_i)), dtype=np.float32), var=var_i))

    merged_meta = ad.concat(obs_uns_adatas, axis=0, join='outer', merge='same', uns_merge='same')
    merged_var = ad.concat(var_adatas, axis=0, join='inner', merge='same').var
    merged_uns = dict(merged_meta.uns)

    if layer_meta is not None:
        layer_meta.pop('chunk_index', None)
        layer_meta.pop('cell_range', None)
        layer_meta['n_cells_total'] = sum(n_obs_per_chunk)
        merged_uns['layer_embeddings'] = layer_meta

    return merged_meta.obs, merged_var, merged_uns, n_obs_per_chunk


def _x_and_obsm_specs(chunk_paths, x_kind):
    """dtype/shape of X and each obsm array, promoted across chunks (cheap: header-only)."""
    x_dtypes = []
    n_vars = None
    obsm_dtypes = {}
    obsm_dims = {}
    obsm_keys = None
    for p in chunk_paths:
        with h5py.File(p, 'r') as f:
            if x_kind == 'dense':
                x_dtypes.append(f['X'].dtype)
                if n_vars is None:
                    n_vars = f['X'].shape[1]
            else:
                x_dtypes.append(f['X']['data'].dtype)
                if n_vars is None:
                    n_vars = int(f['X'].attrs['shape'][1])
            keys = set(f['obsm'].keys())
            obsm_keys = keys if obsm_keys is None else (obsm_keys & keys)
            for k in keys:
                obsm_dtypes.setdefault(k, []).append(f['obsm'][k].dtype)
                obsm_dims[k] = f['obsm'][k].shape[1]
    x_dtype = np.result_type(*x_dtypes)
    obsm_specs = {k: (np.result_type(*obsm_dtypes[k]), obsm_dims[k]) for k in obsm_keys}
    return x_dtype, n_vars, obsm_specs


def _write_streamed(chunk_paths, out_path, merged_obs, merged_var, merged_uns, n_obs_per_chunk, compression, x_kind):
    """Write the merged h5ad incrementally: obs/var/uns (small) are written
    once, while obsm['X_layer_*'] (the dominant memory cost: dense
    n_cells x hidden_dim per layer) are streamed chunk-by-chunk into
    resizable on-disk datasets. Peak memory is bounded by a single chunk's
    obsm arrays instead of the sum of all chunks.

    X (gene expression) is dense for some models and sparse (CSR) for others.
    Dense X is streamed the same way as obsm. Sparse X is read per-chunk
    (scipy sparse, much smaller than its dense equivalent) and concatenated
    via scipy.sparse.vstack once all chunks are read.
    """
    n_obs_total = sum(n_obs_per_chunk)
    x_dtype, n_vars, obsm_specs = _x_and_obsm_specs(chunk_paths, x_kind)

    with h5py.File(out_path, 'w') as fout:
        fout.attrs['encoding-type'] = 'anndata'
        fout.attrs['encoding-version'] = '0.1.0'

        write_elem(fout, 'obs', merged_obs)
        write_elem(fout, 'var', merged_var)

        uns_group = fout.create_group('uns')
        uns_group.attrs['encoding-type'] = 'dict'
        uns_group.attrs['encoding-version'] = '0.1.0'
        for k, v in merged_uns.items():
            write_elem(uns_group, k, v)

        for grp_name in ('layers', 'obsp', 'varm', 'varp'):
            g = fout.create_group(grp_name)
            g.attrs['encoding-type'] = 'dict'
            g.attrs['encoding-version'] = '0.1.0'

        if x_kind == 'dense':
            X_ds = fout.create_dataset('X', shape=(n_obs_total, n_vars), dtype=x_dtype,
                                        chunks=True, compression=compression)
            X_ds.attrs['encoding-type'] = 'array'
            X_ds.attrs['encoding-version'] = '0.2.0'
        else:
            x_parts = []

        obsm_group = fout.create_group('obsm')
        obsm_group.attrs['encoding-type'] = 'dict'
        obsm_group.attrs['encoding-version'] = '0.1.0'
        obsm_ds = {}
        for k, (dtype, dim) in obsm_specs.items():
            ds = obsm_group.create_dataset(k, shape=(n_obs_total, dim), dtype=dtype,
                                            chunks=True, compression=compression)
            ds.attrs['encoding-type'] = 'array'
            ds.attrs['encoding-version'] = '0.2.0'
            obsm_ds[k] = ds

        row = 0
        for p, n in zip(chunk_paths, n_obs_per_chunk):
            print(f"  writing {p.name} ({n} cells) ...", end=' ', flush=True)
            with h5py.File(p, 'r') as fin:
                if x_kind == 'dense':
                    X_ds[row:row + n, :] = fin['X'][:]
                else:
                    x_parts.append(read_elem(fin['X']))
                for k, ds in obsm_ds.items():
                    ds[row:row + n, :] = fin['obsm'][k][:]
            row += n
            print("done")

        if x_kind == 'sparse':
            X_full = sp.vstack(x_parts, format=x_parts[0].format)
            write_elem(fout, 'X', X_full, dataset_kwargs={'compression': compression})


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

    # Default: data/embeddings/<dataset>/<dataset>_<model_suffix>.h5ad
    # input_dir is expected to be .../embeddings/<dataset>/chunks/<model_suffix>
    dataset_stem = args.input_dir.parent.parent.name
    model_suffix = args.input_dir.name
    out_path = args.output or (args.input_dir.parent.parent / f"{dataset_stem}_{model_suffix}.h5ad")
    if out_path in chunk_paths:
        sys.exit(f"Output path collides with a chunk file: {out_path}")

    if out_path.exists():
        out_path.unlink()

    # All chunks of one dataset/model share the same gene set, so an inner
    # join over var is equivalent to the original outer join here, and lets
    # us stream X/obsm chunk-by-chunk without any gene reindexing. X must also
    # be uniformly dense or uniformly sparse across chunks.
    first_vars = _var_index(chunk_paths[0])
    same_vars = all(first_vars.equals(_var_index(p)) for p in chunk_paths[1:])
    x_kind = _x_kind(chunk_paths)
    streamable = same_vars and x_kind is not None

    if streamable:
        print(f"Merging {len(chunk_paths)} chunks (streamed, one chunk in RAM at a time, X={x_kind}) ...")
        merged_obs, merged_var, merged_uns, n_obs_per_chunk = _merge_metadata(chunk_paths)
        _write_streamed(chunk_paths, out_path, merged_obs, merged_var, merged_uns, n_obs_per_chunk, args.compression, x_kind)
        n_obs, n_vars = sum(n_obs_per_chunk), len(merged_var)
    else:
        print("Gene sets differ across chunks; falling back to in-memory concat ...")
        adatas = []
        layer_meta = None
        for p in chunk_paths:
            print(f"  reading {p.name} ...", end=' ', flush=True)
            a = sc.read_h5ad(p)
            adatas.append(a)
            print(f"({a.n_obs} cells)")
            if layer_meta is None and 'layer_embeddings' in a.uns:
                layer_meta = dict(a.uns['layer_embeddings'])

        merged = ad.concat(adatas, axis=0, join='outer', merge='same', uns_merge='same')
        if layer_meta is not None:
            layer_meta.pop('chunk_index', None)
            layer_meta.pop('cell_range', None)
            layer_meta['n_cells_total'] = int(merged.n_obs)
            merged.uns['layer_embeddings'] = layer_meta

        n_obs, n_vars = merged.n_obs, merged.n_vars
        merged.write_h5ad(out_path, compression=args.compression)

    print(f"Wrote {out_path} ({n_obs} cells x {n_vars} genes).")
    print("Done.")


if __name__ == '__main__':
    main()
