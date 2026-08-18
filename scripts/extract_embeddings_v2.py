#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import scanpy as sc

from scfm_eval.embedders import get_embedder, list_embedders
from scfm_eval.v2.specs import EmbeddingSpec
from scfm_eval.v2.store import EmbeddingStoreWriter


def main():
    p = argparse.ArgumentParser(
        description="Extract all scFM layers into a resumable sharded embedding store."
    )
    p.add_argument("--dataset-key", required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True, choices=list_embedders())
    p.add_argument("--model-size", default=None)
    p.add_argument("--chunk-size", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--no-fp16", action="store_true")
    args = p.parse_args()

    adata = sc.read_h5ad(args.input, backed="r")
    fp16 = not args.no_fp16

    kwargs = {"fp16": fp16}
    if args.model_size:
        kwargs["model_size"] = args.model_size
    embedder = get_embedder(args.model, **kwargs)
    layers = [int(x) for x in embedder.get_all_layer_indices()]

    hidden_dim = int(getattr(embedder, "hidden_dim"))
    spec = EmbeddingSpec(
        dataset=args.dataset_key,
        model=args.model,
        model_size=args.model_size,
        n_layers_total=len(layers),
        hidden_dim=hidden_dim,
        pooling=getattr(embedder, "pooling", None),
        expected_input=getattr(embedder, "expected_input", None),
        dtype="float16" if fp16 else "float32",
        source_path=str(args.input),
    )
    writer = EmbeddingStoreWriter(args.output, spec)

    for start in range(0, adata.n_obs, args.chunk_size):
        end = min(start + args.chunk_size, adata.n_obs)
        if writer.has_range(start, end):
            print(f"[skip] {start}:{end}")
            continue

        print(f"[extract] {start}:{end}")
        original = adata[start:end].to_memory()
        cell_ids = original.obs_names.astype(str).tolist()

        # prepare_data may reorder/filter genes, but cell identity must remain intact.
        prepared = embedder.prepare_data(original)
        if prepared.n_obs != len(cell_ids):
            raise ValueError(
                f"Model preprocessing changed cell count: {len(cell_ids)} -> {prepared.n_obs}"
            )
        if list(prepared.obs_names.astype(str)) != cell_ids:
            raise ValueError("Model preprocessing changed cell ordering")

        embeddings = embedder.extract_embeddings_for_layers(
            prepared, layer_indices=layers, batch_size=args.batch_size
        )
        writer.write_shard(
            start=start,
            end=end,
            cell_ids=cell_ids,
            embeddings={int(k): np.asarray(v) for k, v in embeddings.items()},
        )

        del original, prepared, embeddings
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    writer.finalize(expected_n_obs=adata.n_obs)
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
