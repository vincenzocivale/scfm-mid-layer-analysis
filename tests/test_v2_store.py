from pathlib import Path

import numpy as np

from scfm_eval.v2.specs import EmbeddingSpec
from scfm_eval.v2.store import EmbeddingStore, EmbeddingStoreWriter


def test_sharded_store_roundtrip(tmp_path: Path):
    spec = EmbeddingSpec(
        dataset="toy",
        model="toyfm",
        model_size=None,
        n_layers_total=2,
        hidden_dim=3,
        pooling="cls_token",
        expected_input="raw_counts",
        dtype="float32",
        source_path="toy.h5ad",
    )
    writer = EmbeddingStoreWriter(tmp_path / "store", spec)

    writer.write_shard(
        start=0,
        end=2,
        cell_ids=["a", "b"],
        embeddings={
            0: np.ones((2, 3), dtype=np.float32),
            1: np.full((2, 3), 2, dtype=np.float32),
        },
    )
    writer.write_shard(
        start=2,
        end=3,
        cell_ids=["c"],
        embeddings={
            0: np.ones((1, 3), dtype=np.float32),
            1: np.full((1, 3), 2, dtype=np.float32),
        },
    )
    writer.finalize(expected_n_obs=3)

    store = EmbeddingStore(tmp_path / "store")
    assert store.complete
    ids, layer1 = store.load_layer(1)
    assert ids.tolist() == ["a", "b", "c"]
    assert layer1.shape == (3, 3)
    assert np.all(layer1 == 2)
