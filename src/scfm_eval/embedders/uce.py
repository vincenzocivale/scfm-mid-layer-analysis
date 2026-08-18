"""UCE (Universal Cell Embeddings) embedder.

Reference: Rosen et al., "Universal Cell Embeddings: A Foundation Model for Cell Biology"
           https://www.biorxiv.org/content/10.1101/2023.11.28.568918
Source repo: https://github.com/snap-stanford/UCE (MIT License)

Architecture summary
--------------------
  - 4-layer Transformer (default pretrained model; 33-layer variant also exists)
  - Token dim: 5120 (ESM2 protein embeddings per gene)
  - d_model: 1280, nhead: 20, d_hid: 5120, output_dim: 1280
  - Input: "cell sentence" – sequence of gene tokens sampled weighted by expression,
    ordered by chromosome/genomic position.  First token = CLS.
  - Pooling: CLS token at position 0 (sequence-first format).
  - Final published embedding: decoder(transformer_output)[0, :, :], L2-normalised.
  - Here (intermediate layers): raw transformer_encoder.layers[i] output at CLS position,
    consistent with X_layer_i semantics in this framework (pre-head, post block i).

Weights and model files
-----------------------
  Fetched automatically from figshare on first use (auto_download=True).
  Default storage: data/checkpoints/uce/
  Override via env vars:
    UCE_MODEL_CKPT          – path to 4layer_model.torch
    UCE_MODEL_FILES_DIR     – directory containing all_tokens.torch,
                               species_chrom.csv, species_offsets.pkl,
                               protein_embeddings/

Stochasticity note
------------------
  Cell sentences are built by weighted random sampling of genes.  A fixed
  numpy seed (42) is set during extraction for reproducibility, then restored.
"""
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import issparse
from tqdm import tqdm

from .base import BaseEmbedder

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CKPT = _REPO_ROOT / "data" / "checkpoints" / "uce" / "4layer_model.torch"
_DEFAULT_MODEL_FILES_DIR = _REPO_ROOT / "data" / "checkpoints" / "uce" / "model_files"

# Figshare file IDs for model artefacts (4-layer pretrained model)
_FIGSHARE = {
    "species_chrom.csv":       "https://figshare.com/ndownloader/files/42706558",
    "species_offsets.pkl":     "https://figshare.com/ndownloader/files/42706555",
    "all_tokens.torch":        "https://figshare.com/ndownloader/files/42706585",
    "protein_embeddings.tar.gz": "https://figshare.com/ndownloader/files/42715213",
    "4layer_model.torch":      "https://figshare.com/ndownloader/files/42706576",
}

# Model hyperparameters (4-layer pretrained variant)
_TOKEN_DIM = 5120
_D_MODEL = 1280
_NHEAD = 20
_D_HID = 5120
_OUTPUT_DIM = 1280
_DROPOUT = 0.05
_PAD_LENGTH = 1536
_SAMPLE_SIZE = 1024

# Special token indices in the UCE vocabulary
_PAD_TOKEN_IDX = 0
_CHROM_TOKEN_LEFT_IDX = 1
_CHROM_TOKEN_RIGHT_IDX = 2
_CLS_TOKEN_IDX = 3
_CHROM_TOKEN_OFFSET = 143574   # chromosome tokens start here in all_tokens


class UCEEmbedder(BaseEmbedder):
    # Documented in docs/models.md — see layer-indexing convention there.
    pooling = "cls_token"
    expected_input = "raw_counts"

    def __init__(
        self,
        device: str = "auto",
        fp16: bool = True,
        model_loc: str = None,
        model_files_dir: str = None,
        species: str = "human",
        nlayers: int = 4,
        auto_download: bool = True,
    ):
        super().__init__("uce", device)
        self.fp16 = fp16
        self.species = species
        self.nlayers = nlayers
        self.auto_download = auto_download
        self.model_loc = model_loc or os.environ.get("UCE_MODEL_CKPT", str(_DEFAULT_CKPT))
        self.model_files_dir = Path(
            model_files_dir or os.environ.get("UCE_MODEL_FILES_DIR", str(_DEFAULT_MODEL_FILES_DIR))
        )
        self.genes_matched = None
        # Set by prepare_data(); required before extract_embeddings_for_layers()
        self._pe_row_idxs: torch.Tensor = None
        self._dataset_chroms: np.ndarray = None
        self._dataset_starts: np.ndarray = None
        self.load_model()

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def n_layers_total(self) -> int:
        return len(self.model.transformer_encoder.layers)

    @property
    def hidden_dim(self) -> int:
        return _D_MODEL

    # ------------------------------------------------------------------ #
    #  Model loading                                                       #
    # ------------------------------------------------------------------ #

    def _ensure_model_files(self) -> None:
        """Download all UCE model artefacts from figshare if not already present.

        Note: if figshare is network-blocked, delete the 0-byte placeholder files
        created by a failed download so this function retries. Alternative: set
        UCE_MODEL_CKPT and UCE_MODEL_FILES_DIR to paths with pre-downloaded files.
        """
        from .vendor.uce.download import figshare_download

        self.model_files_dir.mkdir(parents=True, exist_ok=True)

        for filename, url in [
            ("species_chrom.csv",       _FIGSHARE["species_chrom.csv"]),
            ("species_offsets.pkl",     _FIGSHARE["species_offsets.pkl"]),
            ("all_tokens.torch",        _FIGSHARE["all_tokens.torch"]),
        ]:
            path = self.model_files_dir / filename
            # Remove 0-byte placeholder left by an interrupted download so it
            # retries instead of silently skipping.
            if path.exists() and path.stat().st_size == 0:
                path.unlink()
            figshare_download(url, str(path))

        prot_dir = self.model_files_dir / "protein_embeddings"
        if not prot_dir.exists():
            archive = self.model_files_dir / "protein_embeddings.tar.gz"
            if archive.exists() and archive.stat().st_size == 0:
                archive.unlink()
            figshare_download(_FIGSHARE["protein_embeddings.tar.gz"], str(archive))

        model_path = Path(self.model_loc)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.exists() and model_path.stat().st_size == 0:
            model_path.unlink()
        figshare_download(_FIGSHARE["4layer_model.torch"], str(model_path))

    def load_model(self) -> None:
        if self.auto_download:
            self._ensure_model_files()

        model_path = Path(self.model_loc)
        if not model_path.exists() or model_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"UCE checkpoint not found or empty: {model_path}\n"
                "Figshare may be network-blocked. Download manually from:\n"
                f"  {_FIGSHARE['4layer_model.torch']}\n"
                "and place it at the path above (or set UCE_MODEL_CKPT)."
            )

        from torch import nn
        from .vendor.uce.model import TransformerModel

        model = TransformerModel(
            token_dim=_TOKEN_DIM,
            d_model=_D_MODEL,
            nhead=_NHEAD,
            d_hid=_D_HID,
            nlayers=self.nlayers,
            dropout=_DROPOUT,
            output_dim=_OUTPUT_DIM,
        )

        # Placeholder pe_embedding (the real weights come from the state_dict)
        empty_pe = torch.zeros(145469, _TOKEN_DIM)
        empty_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)

        state_dict = torch.load(self.model_loc, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

        # Safety: if the model file did not contain pe_embedding (older variant),
        # load it from the ESM2 token file + generated chromosome tokens.
        all_pe = self._load_token_embeddings()
        if all_pe.shape[0] != 145469:
            all_pe.requires_grad = False
            model.pe_embedding = nn.Embedding.from_pretrained(all_pe)

        dtype = torch.float16 if (self.device.type == "cuda" and self.fp16) else torch.float32
        self.model = model.to(self.device, dtype=dtype)
        self.model.eval()
        print(f"UCE: loaded {self.nlayers}-layer model from {self.model_loc}")

    def _load_token_embeddings(self) -> torch.Tensor:
        """Load gene token embeddings and append random chromosome token embeddings."""
        token_file = self.model_files_dir / "all_tokens.torch"
        all_pe = torch.load(str(token_file), map_location="cpu")
        if all_pe.shape[0] == 143574:
            # Append 1895 chromosome tokens with the fixed seed from the original code.
            torch.manual_seed(23)
            chrom_tensors = torch.normal(mean=0, std=1, size=(1895, _TOKEN_DIM))
            all_pe = torch.vstack((all_pe, chrom_tensors))
            all_pe.requires_grad = False
        return all_pe

    # ------------------------------------------------------------------ #
    #  Data preparation                                                    #
    # ------------------------------------------------------------------ #

    def prepare_data(self, adata):
        """Subset adata to genes with UCE protein embeddings for *self.species*.

        Computes per-gene token indices, chromosome codes, and genomic start
        positions that are later used to build cell sentences.  These are stored
        on the embedder instance and consumed by extract_embeddings_for_layers.

        Expects adata.X to contain raw (un-normalised) counts.
        Gene symbols are resolved via adata.var['feature_name'] if present,
        otherwise adata.var_names.  Matching is case-insensitive.
        """
        from .vendor.uce.data_utils import load_species_pe_genes, get_spec_chrom_csv

        # ── 1. Load species protein-embedding gene list ─────────────────────
        spec_pe_genes_raw = load_species_pe_genes(
            self.model_files_dir / "protein_embeddings", self.species
        )
        spec_pe_genes_upper = [g.upper() for g in spec_pe_genes_raw]
        spec_pe_gene_set = set(spec_pe_genes_upper)
        pe_gene_to_idx: Dict[str, int] = {g: i for i, g in enumerate(spec_pe_genes_upper)}

        # ── 2. Load species offset into global UCE vocabulary ────────────────
        with open(self.model_files_dir / "species_offsets.pkl", "rb") as fh:
            species_to_offsets = pickle.load(fh)
        if self.species not in species_to_offsets:
            raise KeyError(
                f"Species {self.species!r} not found in species_offsets.pkl. "
                f"Available: {sorted(species_to_offsets)}"
            )
        offset = species_to_offsets[self.species]

        # ── 3. Resolve gene symbols from adata ──────────────────────────────
        if "feature_name" in adata.var.columns:
            raw_gene_symbols = list(adata.var["feature_name"])
        else:
            raw_gene_symbols = list(adata.var_names)
        gene_symbols_upper = [g.upper() for g in raw_gene_symbols]

        # ── 4. Filter to genes present in protein-embedding vocabulary ───────
        pe_valid_mask = [g in spec_pe_gene_set for g in gene_symbols_upper]
        pe_valid_idx = [i for i, v in enumerate(pe_valid_mask) if v]
        if not pe_valid_idx:
            raise ValueError(
                f"No genes in adata match UCE protein-embedding vocabulary for "
                f"species={self.species!r}. Ensure var_names / var['feature_name'] "
                "contain HGNC gene symbols (e.g. 'CD3E', not Ensembl IDs)."
            )
        adata_f = adata[:, pe_valid_idx].copy()
        filtered_symbols = [gene_symbols_upper[i] for i in pe_valid_idx]

        # ── 5. Load chromosome map and filter further ────────────────────────
        gene_to_chrom_pos = get_spec_chrom_csv(str(self.model_files_dir / "species_chrom.csv"))
        spec_chrom_df = (
            gene_to_chrom_pos[gene_to_chrom_pos["species"] == self.species]
            .set_index("gene_symbol")
        )
        chrom_gene_set = set(spec_chrom_df.index)

        chrom_valid_mask = [g in chrom_gene_set for g in filtered_symbols]
        chrom_valid_idx = [i for i, v in enumerate(chrom_valid_mask) if v]
        if not chrom_valid_idx:
            raise ValueError(
                f"No genes have chromosome info for species={self.species!r} "
                "in species_chrom.csv."
            )
        adata_f = adata_f[:, chrom_valid_idx].copy()
        final_symbols = [filtered_symbols[i] for i in chrom_valid_idx]

        self.genes_matched = len(final_symbols)
        print(
            f"UCE: matched {self.genes_matched}/{len(raw_gene_symbols)} genes "
            f"with protein embeddings + chromosome info"
        )

        # ── 6. Compute per-gene UCE token indices ────────────────────────────
        self._pe_row_idxs = torch.tensor(
            [pe_gene_to_idx[g] + offset for g in final_symbols], dtype=torch.long
        )

        # ── 7. Compute chromosome codes and genomic start positions ──────────
        gene_chrom = spec_chrom_df.loc[final_symbols]
        self._dataset_chroms = gene_chrom["spec_chrom"].cat.codes.values.astype(np.int64)
        self._dataset_starts = gene_chrom["start"].values.astype(np.float64)

        return adata_f

    # ------------------------------------------------------------------ #
    #  Embedding extraction                                                #
    # ------------------------------------------------------------------ #

    def extract_embeddings_for_layers(
        self, adata, layer_indices: List[int], batch_size: int = 25
    ) -> Dict[int, np.ndarray]:
        """Extract per-layer CLS embeddings using forward hooks on transformer blocks.

        X_layer_i = output of i-th TransformerEncoderLayer at CLS position,
        shape (n_cells, 1280).  Pre-decoder, consistent with framework convention.

        A fixed numpy seed is set during extraction so cell-sentence sampling is
        reproducible across runs; the original seed is restored on exit.
        """
        if self._pe_row_idxs is None:
            raise RuntimeError("Call prepare_data() before extract_embeddings_for_layers().")

        X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X, dtype=np.float32)
        n_cells = X.shape[0]

        layer_outputs: Dict[int, List[torch.Tensor]] = {li: [] for li in layer_indices}

        def make_hook(layer_idx: int):
            def hook_fn(module, inp, output):
                # TransformerEncoderLayer may return a tuple in newer PyTorch
                tensor = output[0] if isinstance(output, tuple) else output
                # seq-first: [seq_len, batch, d_model] → CLS at position 0
                layer_outputs[layer_idx].append(tensor[0, :, :].detach().float().cpu())
            return hook_fn

        hooks = []
        for li in layer_indices:
            if li >= self.n_layers_total:
                warnings.warn(f"UCE layer {li} out of range (n_layers={self.n_layers_total}). Skipping.")
                continue
            hooks.append(
                self.model.transformer_encoder.layers[li].register_forward_hook(make_hook(li))
            )

        if not hooks:
            warnings.warn("No valid layer indices — aborting UCE extraction.")
            return {}

        dtype = torch.float16 if (self.device.type == "cuda" and self.fp16) else torch.float32
        # Save and fix RNG state for reproducible gene sampling
        rng_state = np.random.get_state()
        np.random.seed(42)

        try:
            with torch.no_grad():
                for start in tqdm(range(0, n_cells, batch_size), desc="Extracting UCE layers"):
                    batch_X = X[start: start + batch_size]
                    cell_sentences, mask = self._make_cell_sentences(batch_X)

                    # [seq_len, batch] → pe_embedding → [seq_len, batch, token_dim]
                    cell_sentences = cell_sentences.permute(1, 0).to(self.device)
                    token_embs = self.model.pe_embedding(cell_sentences)
                    token_embs = F.normalize(token_embs.to(dtype), dim=2)
                    # PAD token (idx=0) has an all-zero embedding; F.normalize produces NaN.
                    # Replace NaN with 0 — PAD positions are masked in attention anyway.
                    token_embs = torch.nan_to_num(token_embs, nan=0.0)

                    mask = mask.to(self.device, dtype=dtype)
                    self.model.forward(token_embs, mask=mask)
        finally:
            for h in hooks:
                h.remove()
            np.random.set_state(rng_state)

        final: Dict[int, np.ndarray] = {}
        for li, embs in layer_outputs.items():
            if not embs:
                continue
            arr = torch.cat(embs, dim=0).numpy()
            nan_count = int(np.isnan(arr).sum())
            if nan_count > 0:
                raise ValueError(
                    f"UCE layer {li}: {nan_count}/{arr.size} NaN values detected. "
                    "Ensure adata.X contains non-negative raw counts."
                )
            final[li] = arr
        return final

    # ------------------------------------------------------------------ #
    #  Cell-sentence construction                                          #
    # ------------------------------------------------------------------ #

    def _make_cell_sentences(
        self, counts_batch: np.ndarray
    ):
        """Build token-index cell sentences for a batch of cells.

        counts_batch: (batch_size, n_genes) raw counts (float32 OK)
        Returns:
          cell_sentences : LongTensor [batch_size, seq_len]
          mask           : FloatTensor [batch_size, seq_len]  (1=attend, 0=ignore)
        """
        batch_size, n_genes = counts_batch.shape
        pe_idxs = self._pe_row_idxs.numpy()
        chroms = self._dataset_chroms     # (n_genes,) int64
        starts = self._dataset_starts     # (n_genes,) float64

        # Pre-fill with CLS (mirrors original UCE: position 0 = CLS)
        cell_sentences = np.full((batch_size, _PAD_LENGTH), _CLS_TOKEN_IDX, dtype=np.int64)
        mask = np.zeros((batch_size, _PAD_LENGTH), dtype=np.float32)
        longest = 0

        for c in range(batch_size):
            weights = np.log1p(counts_batch[c]).astype(np.float64)
            total = weights.sum()
            if total == 0:
                # Zero-expression cell: keep only CLS token
                mask[c, 0] = 1.0
                longest = max(longest, 1)
                continue
            weights /= total

            # Sample _SAMPLE_SIZE genes, weighted by expression
            choice_idx = np.random.choice(n_genes, size=_SAMPLE_SIZE, p=weights, replace=True)
            chosen_chrom = chroms[choice_idx]

            # Sort by chromosome first
            chrom_sort = np.argsort(chosen_chrom)
            choice_idx = choice_idx[chrom_sort]
            new_chrom = chroms[choice_idx]
            chosen_starts = starts[choice_idx]

            i = 1  # position 0 is CLS
            uq_chroms = np.unique(new_chrom)
            np.random.shuffle(uq_chroms)

            for chrom in uq_chroms:
                if i >= _PAD_LENGTH - 2:
                    break
                # Chromosome open token
                cell_sentences[c, i] = int(chrom) + _CHROM_TOKEN_OFFSET
                i += 1
                # Genes on this chromosome, sorted by start position
                loc = np.where(new_chrom == chrom)[0]
                sort_by_start = np.argsort(chosen_starts[loc])
                to_add = choice_idx[loc[sort_by_start]]
                room = _PAD_LENGTH - i - 1  # reserve 1 for closing token
                to_add = to_add[:room]
                cell_sentences[c, i: i + len(to_add)] = pe_idxs[to_add]
                i += len(to_add)
                # Chromosome close token
                cell_sentences[c, i] = _CHROM_TOKEN_RIGHT_IDX
                i += 1

            longest = max(longest, i)
            mask[c, :i] = 1.0
            cell_sentences[c, i:] = _PAD_TOKEN_IDX  # PAD the remainder

        return (
            torch.from_numpy(cell_sentences[:, :longest]),
            torch.from_numpy(mask[:, :longest]),
        )

    # ------------------------------------------------------------------ #
    #  Misc                                                                #
    # ------------------------------------------------------------------ #

    def get_all_layer_indices(self) -> List[int]:
        return list(range(self.n_layers_total))
