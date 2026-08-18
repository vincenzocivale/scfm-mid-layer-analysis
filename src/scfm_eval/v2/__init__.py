"""Second-generation experiment architecture for scFM mid-layer analysis."""

from .specs import EmbeddingSpec, MetricRecord, RunSpec
from .store import EmbeddingStore, EmbeddingStoreWriter
from .results import ResultWriter, load_metric_tables
from .compare import add_final_layer_comparison

__all__ = [
    "EmbeddingSpec",
    "MetricRecord",
    "RunSpec",
    "EmbeddingStore",
    "EmbeddingStoreWriter",
    "ResultWriter",
    "load_metric_tables",
    "add_final_layer_comparison",
]
