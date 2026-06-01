"""CellFM embedder — MindSpore Retention-based single-cell foundation model.

CellFM requires a dedicated conda environment (envs/cellfm-env.yml) because
it uses MindSpore instead of PyTorch.

Architecture: 40 RetentionLayer blocks, hidden_dim=1536.
Pooling: mean over non-zero-expression gene positions (no CLS token in Encoder).
Input: raw counts mapped to the CellFM gene vocabulary (vendor/cellfm/csv/gene_info.csv).

Checkpoint (CELLFM_CKPT env var): data/checkpoints/cellfm/base_weight.ckpt
Download from: https://huggingface.co/ShangguanNingyuan/CellFM/tree/main
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import issparse
from tqdm import tqdm


def _preload_mindspore_gpu_plugin() -> None:
    """Pre-load libmindspore_gpu.so before MindSpore is imported.

    MindSpore 2.9's GPU plugin requires being loaded via ctypes BEFORE the
    _c_expression C extension initialises, otherwise the device-discovery
    callback cannot register GPU as a supported backend.  We search for the
    first loadable variant (11.6 preferred, then 11.1) in the package's
    plugin dir.  Failures are silently ignored — MindSpore falls back to CPU.
    """
    try:
        import importlib.util
        ms_spec = importlib.util.find_spec("mindspore")
        if ms_spec is None:
            return
        plugin_dir = Path(ms_spec.origin).parent / "lib" / "plugin"
        for candidate in ["libmindspore_gpu.so.11.6", "libmindspore_gpu.so.11.1"]:
            path = plugin_dir / candidate
            if path.exists():
                try:
                    ctypes.CDLL(str(path))
                    return
                except OSError:
                    continue
    except Exception:
        pass


_preload_mindspore_gpu_plugin()

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "cellfm"
_DEFAULT_CKPT = _REPO_ROOT / "data" / "checkpoints" / "cellfm" / "base_weight.ckpt"
_DEFAULT_GENE_INFO = _VENDOR_DIR / "csv" / "gene_info.csv"

_N_LAYERS = 40
_HIDDEN_DIM = 1536


class CellFMEmbedder:
    """Per-layer cell embeddings from CellFM via MindSpore PYNATIVE_MODE."""

    pooling = "mean_nonzero"
    expected_input = "raw_counts"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        gene_info_path: Optional[str] = None,
        device: str = "auto",
        fp16: bool = True,  # accepted for API compatibility; MindSpore dtype handled separately
    ):
        self.model_name = "cellfm"
        self.device = None  # MindSpore device is set globally via ms.set_context

        self.checkpoint_path = checkpoint_path or os.environ.get(
            "CELLFM_CKPT", str(_DEFAULT_CKPT)
        )
        self.gene_info_path = gene_info_path or str(_DEFAULT_GENE_INFO)

        if device in ("cuda", "gpu", "GPU"):
            self._ms_device = "GPU"
        elif device == "cpu":
            self._ms_device = "CPU"
        else:  # auto
            try:
                import mindspore as ms
                # Try GPU; fall back to CPU if not available
                ms.set_context(device_target="GPU")
                self._ms_device = "GPU"
            except Exception:
                self._ms_device = "CPU"

        self.encoder = None
        self.geneset: Optional[dict] = None
        self._n_model_genes: Optional[int] = None
        self._cfg = None
        self.genes_matched: Optional[int] = None

        self.load_model()

    # ── metadata properties (read by build_layer_metadata) ──────────────────

    @property
    def n_layers_total(self) -> int:
        return _N_LAYERS

    @property
    def hidden_dim(self) -> int:
        return _HIDDEN_DIM

    # ── BaseEmbedder abstract interface ─────────────────────────────────────

    def load_model(self) -> None:
        """Load gene vocabulary and Encoder weights from checkpoint."""
        import mindspore as ms
        import pandas as pd

        if str(_VENDOR_DIR) not in sys.path:
            sys.path.insert(0, str(_VENDOR_DIR))

        from config import Config           # CellFM/config.py
        from encoder_infer import Encoder   # CellFM/encoder_infer.py (safe import, no module-level exec)

        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=self._ms_device)
        ms.set_seed(0)

        cfg = Config()
        cfg.recompute = False  # disable graph-mode recompute flag (no-op in PYNATIVE)

        gene_info = pd.read_csv(self.gene_info_path, index_col=0, header=0)
        self.geneset = {g: i + 1 for i, g in enumerate(gene_info.index)}
        self._n_model_genes = len(self.geneset)
        self._cfg = cfg

        # Build Encoder with a 1-element placeholder; prepare_data updates used_gene.
        placeholder_gene = np.zeros(1, dtype=np.int32)
        self.encoder = Encoder(self._n_model_genes, placeholder_gene, cfg)

        para = ms.load_checkpoint(self.checkpoint_path)
        err, _ = ms.load_param_into_net(self.encoder, para)
        if err:
            # err is a list of param names in MindSpore >=2.x
            names = list(err.keys()) if hasattr(err, 'keys') else list(err)
            print(f"CellFM: {len(names)} params not loaded (first 5: {names[:5]})")

        self.encoder.set_train(False)
        print(f"CellFM loaded — vocab={self._n_model_genes} genes, "
              f"layers={_N_LAYERS}, dim={_HIDDEN_DIM}, device={self._ms_device}")

    def prepare_data(self, adata):
        """Map dataset gene names to CellFM vocabulary indices.

        Returns a copy of adata subset to matched genes only, which reduces
        the RetentionLayer sequence length and speeds up extraction significantly.
        """
        import mindspore as ms

        # Try var_names first; fall back to feature_name column for Ensembl-ID datasets.
        candidates = list(adata.var_names)
        gene = np.array([self.geneset.get(g, 0) for g in candidates], dtype=np.int32)
        if gene.sum() == 0 and "feature_name" in adata.var.columns:
            candidates = list(adata.var["feature_name"])
            gene = np.array([self.geneset.get(g, 0) for g in candidates], dtype=np.int32)

        matched = int((gene > 0).sum())
        self.genes_matched = matched
        print(f"CellFM: matched {matched}/{len(gene)} genes "
              f"({matched / max(len(gene), 1) * 100:.1f}%)")

        # Subset to matched genes to reduce RetentionLayer sequence length.
        # Unmatched genes (ID=0) get the UNK embedding and add O(n²) overhead
        # for no biological signal — filtering them out is correct and faster.
        matched_mask = gene > 0
        adata_f = adata[:, matched_mask].copy()
        matched_gene = gene[matched_mask]

        self.encoder.used_gene = ms.Tensor(matched_gene, ms.int32)
        return adata_f

    def extract_embeddings_for_layers(
        self,
        adata,
        layer_indices: List[int],
        batch_size: int = 8,
    ) -> Dict[int, np.ndarray]:
        """Extract mean-pooled cell embeddings at requested layer depths."""
        import mindspore as ms

        layer_set = set(layer_indices)
        max_layer = max(layer_indices)
        layer_outputs: Dict[int, list] = {idx: [] for idx in layer_indices}
        enc = self.encoder
        n_obs = adata.n_obs

        for start in tqdm(range(0, n_obs, batch_size), desc="CellFM layers"):
            end = min(start + batch_size, n_obs)

            if issparse(adata.X):
                expr_np = adata.X[start:end].toarray().astype(np.float32)
            else:
                expr_np = np.asarray(adata.X[start:end], dtype=np.float32)

            b, l = expr_np.shape

            expr_ms = ms.Tensor(expr_np, ms.float32)
            zero_mask = ms.Tensor((expr_np != 0).astype(np.float32), ms.float32)

            # --- Replicate Encoder.construct initial embedding steps ---
            # len_scale: (b, 1, 1, 1) — per-cell scaling factor for RetentionLayer
            len_scale = enc.detach(
                enc.rsqrt(enc.sum(zero_mask, -1)).reshape(b, 1, 1, 1)
            )
            # gene_emb: (1, l, hidden_dim) — vocabulary embedding for each dataset gene
            gene_emb = enc.gather1(enc.gene_emb, enc.used_gene, 0).reshape(1, l, -1)
            init_emb, _ = enc.value_enc(expr_ms)
            expr_emb = init_emb + gene_emb  # MindSpore __add__ in PYNATIVE_MODE

            # --- Layer-by-layer forward ---
            nonzero_mask = zero_mask.reshape(b, l, 1)  # (b, l, 1) for pooling
            n_nonzero = nonzero_mask.sum(axis=1)         # (b, 1)

            for layer_idx in range(max_layer + 1):
                expr_emb = enc.encoder[layer_idx](
                    expr_emb,
                    v_pos=len_scale,
                    seq_mask=None,
                )
                if layer_idx in layer_set:
                    # Mean pool over expressed genes → (b, hidden_dim)
                    cell_emb = (expr_emb * nonzero_mask).sum(axis=1) / (n_nonzero + 1e-8)
                    layer_outputs[layer_idx].append(cell_emb.asnumpy())

        return {
            idx: np.concatenate(embs, axis=0)
            for idx, embs in layer_outputs.items()
            if embs
        }

    def extract_layer_embeddings(self, adata, layer_idx: int) -> np.ndarray:
        return self.extract_embeddings_for_layers(adata, [layer_idx])[layer_idx]

    def extract_all_layers(
        self,
        adata,
        output_path: str,
        layers: Optional[List[int]] = None,
    ) -> None:
        import h5py

        if layers is None:
            layers = self.get_all_layer_indices()

        with h5py.File(output_path, "w") as f:
            f.attrs["model_name"] = self.model_name
            f.attrs["n_cells"] = adata.n_obs
            for layer_idx in layers:
                embs = self.extract_layer_embeddings(adata, layer_idx)
                f.create_dataset(f"layer_{layer_idx}", data=embs, compression="gzip")

        print(f"Saved to {output_path}")

    def get_all_layer_indices(self) -> List[int]:
        return list(range(_N_LAYERS))
