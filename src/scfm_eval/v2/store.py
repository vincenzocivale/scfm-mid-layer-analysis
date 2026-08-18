from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from .specs import EmbeddingSpec


MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 2


def _decode_ids(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O"}:
        return np.asarray([
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in values
        ], dtype=object)
    return values.astype(str)


class EmbeddingStoreWriter:
    """Append-only writer for resumable embedding extraction.

    A store contains one small manifest and independent HDF5 shards.
    The source expression matrix is intentionally *not* copied into the store.
    """

    def __init__(self, root: Path, spec: EmbeddingSpec):
        self.root = Path(root)
        self.shards_dir = self.root / "shards"
        self.root.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.manifest_path = self.root / MANIFEST_NAME
        self._manifest = self._load_or_init()

    def _load_or_init(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                m = json.load(f)
            if m.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Unsupported embedding-store schema: {m.get('schema_version')}")
            existing = m["embedding_spec"]
            for key in ("dataset", "model", "model_size", "n_layers_total", "hidden_dim"):
                if existing.get(key) != self.spec.to_dict().get(key):
                    raise ValueError(f"Store/spec mismatch for {key}: {existing.get(key)!r} != {self.spec.to_dict().get(key)!r}")
            return m
        m = {
            "schema_version": SCHEMA_VERSION,
            "embedding_spec": self.spec.to_dict(),
            "layers": [],
            "n_obs": 0,
            "shards": [],
            "complete": False,
        }
        self._write_manifest(m)
        return m

    def _write_manifest(self, manifest: dict) -> None:
        fd, tmp = tempfile.mkstemp(prefix=".manifest.", suffix=".json", dir=self.root)
        os.close(fd)
        try:
            with open(tmp, "w") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
            os.replace(tmp, self.manifest_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @property
    def completed_ranges(self) -> List[Tuple[int, int]]:
        return [(int(x["start"]), int(x["end"])) for x in self._manifest["shards"]]

    def has_range(self, start: int, end: int) -> bool:
        return (int(start), int(end)) in set(self.completed_ranges)

    def write_shard(
        self,
        *,
        start: int,
        end: int,
        cell_ids: Sequence[str],
        embeddings: Dict[int, np.ndarray],
    ) -> Path:
        if end <= start:
            raise ValueError("end must be > start")
        if len(cell_ids) != end - start:
            raise ValueError("cell_ids length does not match [start:end]")
        if self.has_range(start, end):
            return self.shards_dir / f"{start:012d}_{end:012d}.h5"

        layers = sorted(int(k) for k in embeddings)
        if not layers:
            raise ValueError("No layer embeddings supplied")

        n = end - start
        for layer in layers:
            arr = np.asarray(embeddings[layer])
            if arr.ndim != 2 or arr.shape[0] != n:
                raise ValueError(f"Layer {layer} has invalid shape {arr.shape}; expected ({n}, hidden_dim)")
            if arr.shape[1] != self.spec.hidden_dim:
                raise ValueError(
                    f"Layer {layer} hidden dim {arr.shape[1]} != declared {self.spec.hidden_dim}"
                )

        target = self.shards_dir / f"{start:012d}_{end:012d}.h5"
        tmp = target.with_suffix(".tmp.h5")
        if tmp.exists():
            tmp.unlink()

        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(tmp, "w") as f:
            f.attrs["schema_version"] = SCHEMA_VERSION
            f.attrs["start"] = int(start)
            f.attrs["end"] = int(end)
            f.create_dataset("cell_id", data=np.asarray(cell_ids, dtype=object), dtype=string_dtype)
            g = f.create_group("layers")
            for layer in layers:
                arr = np.asarray(embeddings[layer])
                g.create_dataset(
                    str(layer),
                    data=arr,
                    compression="gzip",
                    shuffle=True,
                    chunks=True,
                )
        os.replace(tmp, target)

        self._manifest["shards"].append({
            "path": target.relative_to(self.root).as_posix(),
            "start": int(start),
            "end": int(end),
            "n_obs": int(n),
        })
        self._manifest["shards"] = sorted(self._manifest["shards"], key=lambda x: x["start"])
        self._manifest["layers"] = sorted(set(self._manifest["layers"]) | set(layers))
        self._manifest["n_obs"] = int(sum(x["n_obs"] for x in self._manifest["shards"]))
        self._write_manifest(self._manifest)
        return target

    def finalize(self, expected_n_obs: Optional[int] = None) -> None:
        shards = self._manifest["shards"]
        if not shards:
            raise ValueError("Cannot finalize an empty store")

        expected_start = 0
        for shard in shards:
            if shard["start"] != expected_start:
                raise ValueError(
                    f"Non-contiguous store: expected next shard at {expected_start}, got {shard['start']}"
                )
            expected_start = shard["end"]

        if expected_n_obs is not None and expected_start != int(expected_n_obs):
            raise ValueError(f"Store ends at {expected_start}, expected {expected_n_obs}")

        expected_layers = list(range(self.spec.n_layers_total))
        if self._manifest["layers"] != expected_layers:
            raise ValueError(
                f"Layer coverage mismatch. Found {self._manifest['layers']}, expected {expected_layers}"
            )

        self._manifest["complete"] = True
        self._manifest["n_obs"] = expected_start
        self._write_manifest(self._manifest)


class EmbeddingStore:
    """Read-only view over a sharded embedding store."""

    def __init__(self, root: Path):
        self.root = Path(root)
        with open(self.root / MANIFEST_NAME) as f:
            self.manifest = json.load(f)
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported embedding store schema")
        self.spec = EmbeddingSpec(**self.manifest["embedding_spec"])

    @property
    def layers(self) -> List[int]:
        return [int(x) for x in self.manifest["layers"]]

    @property
    def n_obs(self) -> int:
        return int(self.manifest["n_obs"])

    @property
    def complete(self) -> bool:
        return bool(self.manifest.get("complete", False))

    def iter_shards(self, layer: Optional[int] = None) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
        for shard in self.manifest["shards"]:
            with h5py.File(self.root / shard["path"], "r") as f:
                ids = _decode_ids(f["cell_id"][...])
                arr = None if layer is None else f["layers"][str(int(layer))][...]
                yield ids, arr

    def cell_ids(self) -> np.ndarray:
        return np.concatenate([ids for ids, _ in self.iter_shards()])

    def load_layer(self, layer: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
        if int(layer) not in self.layers:
            raise KeyError(f"Layer {layer} not in store; available={self.layers}")
        ids, arrays = [], []
        for cell_ids, emb in self.iter_shards(layer=int(layer)):
            ids.append(cell_ids)
            arrays.append(np.asarray(emb, dtype=dtype))
        return np.concatenate(ids), np.concatenate(arrays, axis=0)

    def assert_alignment(self, obs_names: Sequence[str]) -> None:
        expected = np.asarray(obs_names, dtype=str)
        actual = self.cell_ids().astype(str)
        if len(expected) != len(actual):
            raise ValueError(f"Cell count mismatch: dataset={len(expected)}, embeddings={len(actual)}")
        bad = np.flatnonzero(expected != actual)
        if len(bad):
            i = int(bad[0])
            raise ValueError(
                f"Cell-order mismatch at row {i}: dataset={expected[i]!r}, embeddings={actual[i]!r}"
            )
