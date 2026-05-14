"""I/O helpers: standardized serialization of embeddings and benchmark results."""
from .embeddings import (
    LayerEmbeddingsMeta,
    build_layer_metadata,
    read_embedded_h5ad,
    write_embedded_h5ad,
)
from .results import results_csv_path, write_results_csv

__all__ = [
    'LayerEmbeddingsMeta',
    'build_layer_metadata',
    'write_embedded_h5ad',
    'read_embedded_h5ad',
    'results_csv_path',
    'write_results_csv',
]
