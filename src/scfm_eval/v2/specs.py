from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EmbeddingSpec:
    dataset: str
    model: str
    model_size: Optional[str]
    n_layers_total: int
    hidden_dim: int
    pooling: Optional[str]
    expected_input: Optional[str]
    dtype: str
    source_path: str

    @property
    def model_variant(self) -> str:
        return self.model if not self.model_size else f"{self.model}-{self.model_size}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    model: str
    model_size: Optional[str]
    task: str
    artifact_dir: str
    dataset_path: str

    @property
    def model_variant(self) -> str:
        return self.model if not self.model_size else f"{self.model}-{self.model_size}"

    @property
    def run_id(self) -> str:
        return f"{self.dataset}__{self.model_variant}__{self.task}"


@dataclass(frozen=True)
class MetricRecord:
    dataset: str
    model: str
    model_size: Optional[str]
    task: str
    representation: str
    layer: Optional[int]
    n_layers_total: Optional[int]
    metric: str
    value: float
    higher_is_better: bool
    n_obs: Optional[int] = None
    split: str = "eval"
    notes: str = ""

    @property
    def relative_depth(self) -> Optional[float]:
        if self.layer is None or self.n_layers_total is None:
            return None
        if self.n_layers_total <= 1:
            return 0.0
        return self.layer / (self.n_layers_total - 1)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["relative_depth"] = self.relative_depth
        return d
