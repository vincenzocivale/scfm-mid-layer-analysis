"""
Estrae embeddings di TUTTI i layer richiesti in UNA chiamata,
processando il dataset a CHUNK per evitare saturazione RAM.
"""

import sys
from pathlib import Path
import os
import gc
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, './models')

import scanpy as sc
import numpy as np

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Estrae embedding per tutti i layer richiesti, processando l'h5ad a chunk."
    )
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', required=True, choices=['tahoe', 'scfoundation', 'scgpt'])
    parser.add_argument('--model-size', default='70m')
    parser.add_argument('--layers', nargs='+', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=50000)
    parser.add_argument('--no-fp16', action='store_true')

    args = parser.parse_args()
    fp16 = not args.no_fp16

    # --------------------------------------------------
    # Output setup
    # --------------------------------------------------
    input_name = args.input.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Open AnnData backed
    # --------------------------------------------------
    print(f"Opening AnnData in backed mode: {args.input}")
    adata_backed = sc.read_h5ad(args.input, backed="r")

    n_cells = adata_backed.n_obs
    n_chunks = (n_cells - 1) // args.chunk_size + 1

    print(f"Cells       : {n_cells}")
    print(f"Chunk size  : {args.chunk_size}")
    print(f"Chunks      : {n_chunks}")
    print(f"Layers      : {args.layers}")

    # --------------------------------------------------
    # Init model ONCE
    # --------------------------------------------------
    print(f"Initializing model: {args.model}")
    if args.model == 'tahoe':
        from tahoe_embedder import TahoeEmbedder
        embedder = TahoeEmbedder(
            model_size=args.model_size,
            fp16=fp16
        )
    elif args.model == 'scfoundation':
        from scfoundation_embedder import scFoundationEmbedder
        embedder = scFoundationEmbedder(fp16=fp16)
    elif args.model == 'scgpt':
        from scgpt_embedder import scGPTEmbedder
        embedder = scGPTEmbedder(fp16=fp16)
    else:
        raise ValueError("Unsupported model")

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

        # --------------------------------------------------
        # Build output AnnData
        # --------------------------------------------------
        adata_out = sc.AnnData(
            X=adata_prepared.X,
            obs=adata_prepared.obs,
            var=adata_prepared.var
        )

        for layer, emb in layer_embeddings.items():
            adata_out.obsm[f"X_layer_{layer}"] = emb.astype(np.float16)

        adata_out.uns['layer_embeddings'] = {
            'model': args.model,
            'model_size': args.model_size if args.model == 'tahoe' else None,
            'layers': args.layers,
            'chunk_index': chunk_idx,
            'cell_range': [start, end],
        }

        # --------------------------------------------------
        # Save chunk
        # --------------------------------------------------
        print(f"Saving chunk -> {chunk_path}")
        adata_out.write_h5ad(chunk_path, compression="gzip")

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
