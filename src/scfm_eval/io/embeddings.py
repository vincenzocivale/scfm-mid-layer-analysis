"""Standardized read/write for embedded h5ads.

Schema contract (see docs/architecture.md):

    adata.obsm['X_layer_{i}']   for each extracted layer i  (n_cells, hidden_dim)
    adata.uns['layer_embeddings'] = {
        'model':            str,        # e.g. 'scfoundation', 'tahoe'
        'model_size':       str | None, # e.g. '1b'; None if not applicable
        'n_layers_total':   int,        # total transformer blocks of the model
        'layers_extracted': list[int],
        'hidden_dim':       int | None,
        'pooling':          str | None, # 'cls_token' | 's_token' | 'mean' | ...
        'expected_input':   str | None, # 'raw_counts' | 'log1p_normalized' | ...
        'fp16':             bool,
        'genes_matched':    int | None,
        # chunked output only:
        'chunk_index':      int,
        'cell_range':       [start, end],
    }
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import anndata as ad
import numpy as np
import scanpy as sc


LAYER_KEY_PREFIX = 'X_layer_'
UNS_KEY = 'layer_embeddings'


@dataclass
class LayerEmbeddingsMeta:
    """Structured metadata for `adata.uns['layer_embeddings']`."""
    model: str
    n_layers_total: int
    layers_extracted: List[int]
    model_size: Optional[str] = None
    hidden_dim: Optional[int] = None
    pooling: Optional[str] = None
    expected_input: Optional[str] = None
    fp16: bool = True
    genes_matched: Optional[int] = None
    chunk_index: Optional[int] = None
    cell_range: Optional[List[int]] = None
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        extra = d.pop('extra')
        d.update(extra)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> 'LayerEmbeddingsMeta':
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(extra=extra, **kwargs)


def build_layer_metadata(embedder, layers_extracted: List[int],
                          *, fp16: bool, chunk_index: Optional[int] = None,
                          cell_range: Optional[List[int]] = None) -> LayerEmbeddingsMeta:
    """Inspect an embedder instance and produce a LayerEmbeddingsMeta."""
    return LayerEmbeddingsMeta(
        model=getattr(embedder, 'model_name', type(embedder).__name__).split('-')[0],
        model_size=getattr(embedder, 'model_size', None),
        n_layers_total=int(getattr(embedder, 'n_layers_total', len(layers_extracted))),
        layers_extracted=list(layers_extracted),
        hidden_dim=(int(embedder.hidden_dim) if getattr(embedder, 'hidden_dim', None) else None),
        pooling=getattr(embedder, 'pooling', None),
        expected_input=getattr(embedder, 'expected_input', None),
        fp16=bool(fp16),
        genes_matched=getattr(embedder, 'genes_matched', None),
        chunk_index=chunk_index,
        cell_range=list(cell_range) if cell_range is not None else None,
    )


def write_embedded_h5ad(
    adata: ad.AnnData,
    layer_embeddings: Dict[int, np.ndarray],
    meta: LayerEmbeddingsMeta,
    output_path: Union[str, Path],
    *,
    dtype=np.float16,
    compression: str = 'gzip',
) -> Path:
    """Attach per-layer embeddings to `adata.obsm` + metadata to `adata.uns` and write to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for layer_idx, emb in layer_embeddings.items():
        if emb.shape[0] != adata.n_obs:
            raise ValueError(
                f"Layer {layer_idx}: embedding has {emb.shape[0]} rows, expected {adata.n_obs}"
            )
        adata.obsm[f'{LAYER_KEY_PREFIX}{layer_idx}'] = emb.astype(dtype)

    adata.uns[UNS_KEY] = meta.to_dict()
    adata.write_h5ad(output_path, compression=compression)
    return output_path


def read_embedded_h5ad(path: Union[str, Path]) -> tuple:
    """Read an embedded h5ad and return (adata, meta). Raises if the schema is missing."""
    adata = sc.read_h5ad(path)
    if UNS_KEY not in adata.uns:
        raise ValueError(f"{path} has no {UNS_KEY!r} in uns — was it produced by extract?")
    layer_keys = [k for k in adata.obsm if k.startswith(LAYER_KEY_PREFIX)]
    if not layer_keys:
        raise ValueError(f"{path} has no {LAYER_KEY_PREFIX}* keys in obsm.")
    return adata, LayerEmbeddingsMeta.from_dict(dict(adata.uns[UNS_KEY]))


def list_layer_keys(adata: ad.AnnData) -> List[str]:
    """Sorted layer keys present in adata.obsm."""
    return sorted(
        (k for k in adata.obsm if k.startswith(LAYER_KEY_PREFIX)),
        key=lambda k: int(k[len(LAYER_KEY_PREFIX):])
    )
