"""Chunked, all-layer embedding extraction.

Processes the input h5ad in chunks (memory-bounded) and writes one h5ad per
chunk into the output directory. Each chunk's adata has `X_layer_{i}` in
`obsm` and rich metadata in `uns['layer_embeddings']`.
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import scanpy as sc

from scfm_eval.embedders import get_embedder, list_embedders
from scfm_eval.io.embeddings import build_layer_metadata, write_embedded_h5ad


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-layer embeddings from a foundation model, chunked."
    )
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', required=True, choices=list_embedders())
    parser.add_argument('--model-size', default=None,
                        help="Required for models with size variants (e.g. tahoe 1b/70m/3b).")
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                        help="Layer indices to extract; default: all layers of the model.")
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=50000)
    parser.add_argument('--no-fp16', action='store_true')

    args = parser.parse_args()
    fp16 = not args.no_fp16

    input_name = args.input.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening AnnData in backed mode: {args.input}")
    adata_backed = sc.read_h5ad(args.input, backed="r")

    n_cells = adata_backed.n_obs
    n_chunks = (n_cells - 1) // args.chunk_size + 1
    print(f"Cells       : {n_cells}")
    print(f"Chunk size  : {args.chunk_size}")
    print(f"Chunks      : {n_chunks}")

    print(f"Initializing model: {args.model}")
    embedder_kwargs = {'fp16': fp16}
    if args.model_size is not None:
        embedder_kwargs['model_size'] = args.model_size
    embedder = get_embedder(args.model, **embedder_kwargs)

    if args.layers is None:
        args.layers = embedder.get_all_layer_indices()
    print(f"Layers      : {args.layers}")

    # ==================================================
    # MAIN LOOP: CHUNK → ALL LAYERS
    # ==================================================
    for chunk_idx, start in enumerate(range(0, n_cells, args.chunk_size)):
        end = min(start + args.chunk_size, n_cells)

        chunk_path = args.output_dir / f"{input_name}_chunk_{chunk_idx:04d}.h5ad"

        if chunk_path.exists():
            print(f"Skipping existing chunk {chunk_idx}")
            continue

        print("")
        print("=" * 70)
        print(f"Processing chunk {chunk_idx} [{start}:{end}]")
        print("=" * 70)

        # --------------------------------------------------
        # Load chunk
        # --------------------------------------------------
        adata_chunk = adata_backed[start:end].to_memory()

        if 'feature_name' not in adata_chunk.var.columns and 'gene_name' in adata_chunk.var.columns:
            adata_chunk.var['feature_name'] = adata_chunk.var['gene_name']

        # --------------------------------------------------
        # Prepare data
        # --------------------------------------------------
        adata_prepared = embedder.prepare_data(adata_chunk)
        del adata_chunk
        gc.collect()

        # --------------------------------------------------
        # Extract ALL layers in one call
        # --------------------------------------------------
        layer_embeddings = embedder.extract_embeddings_for_layers(
            adata_prepared,
            layer_indices=args.layers,
            batch_size=args.batch_size
        )

        adata_out = sc.AnnData(
            X=adata_prepared.X,
            obs=adata_prepared.obs,
            var=adata_prepared.var,
        )
        meta = build_layer_metadata(
            embedder, layers_extracted=args.layers, fp16=fp16,
            chunk_index=chunk_idx, cell_range=[start, end],
        )
        # Force the model name to the CLI choice so 'tahoe' (vs 'tahoe-x1-1b') is canonical.
        meta.model = args.model
        if args.model == 'tahoe':
            meta.model_size = args.model_size

        print(f"Saving chunk -> {chunk_path}")
        write_embedded_h5ad(adata_out, layer_embeddings, meta, chunk_path)

        # --------------------------------------------------
        # Cleanup
        # --------------------------------------------------
        del adata_prepared, adata_out, layer_embeddings
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # --------------------------------------------------
    # Final cleanup
    # --------------------------------------------------
    del embedder, adata_backed
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    print("")
    print("ALL CHUNKS PROCESSED SUCCESSFULLY ✅")


if __name__ == '__main__':
    main()
