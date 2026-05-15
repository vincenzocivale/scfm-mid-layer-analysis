"""GeneCompass embedder — per-layer intermediate embedding extraction.

GeneCompass is a BERT-based single-cell foundation model that integrates
biological prior knowledge (promoter sequences, gene co-expression, gene
families, PECA GRN) into the embedding layer.

Architecture summary:
  - 12 transformer encoder layers (BertEncoder), hidden_size=768
  - CLS token prepended at position 0 (use_cls_token=True in the pretrained config)
  - Knowledge-enriched embedding layer (KnowledgeBertEmbeddings)
  - Input: gene Ensembl IDs + normalised expression values

References:
  Chen B. et al. "GeneCompass: Deciphering Universal Gene Regulatory Logic
  by Integrating Multi-species Single-Cell RNA Sequencing Data". Cell 2024.
"""

import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from .base import BaseEmbedder

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CKPT_DIR = _REPO_ROOT / "data" / "checkpoints" / "genecompass"
_DEFAULT_PRIOR_DIR = _REPO_ROOT / "data" / "checkpoints" / "genecompass" / "prior_knowledge"

# Path to the vendored GeneCompass Python package.
_VENDOR_DIR = Path(__file__).parent / "vendor"


class GeneCompassEmbedder(BaseEmbedder):
    pooling = "cls_token"
    expected_input = "raw_counts"

    def __init__(self, device="auto", fp16=True, model_dir=None, prior_dir=None, max_input_size=1000):
        super().__init__("genecompass", device)
        self.fp16 = fp16
        # cap to model's max_position_embeddings - 1 (for CLS)
        self.max_input_size = max_input_size
        self.model_dir = Path(
            model_dir or os.environ.get("GENECOMPASS_CKPT", str(_DEFAULT_CKPT_DIR))
        )
        self.prior_dir = Path(
            prior_dir or os.environ.get("GENECOMPASS_PRIOR_DIR", str(_DEFAULT_PRIOR_DIR))
        )
        self.token_dictionary = None
        self.gene_to_token = None
        self.genes_matched = None
        self.load_model()

    # ------------------------------------------------------------------ #
    # Properties required by the pipeline                                  #
    # ------------------------------------------------------------------ #

    @property
    def n_layers_total(self) -> int:
        return self.model.bert.config.num_hidden_layers

    @property
    def hidden_dim(self) -> int:
        return self.model.bert.config.hidden_size

    # ------------------------------------------------------------------ #
    # Model loading                                                         #
    # ------------------------------------------------------------------ #

    def _ensure_vendor_on_path(self):
        vendor_str = str(_VENDOR_DIR)
        if vendor_str not in sys.path:
            sys.path.insert(0, vendor_str)

    def _load_token_dictionary(self):
        token_dict_path = self.prior_dir / "h&m_token1000W.pickle"
        if not token_dict_path.exists():
            raise FileNotFoundError(
                f"GeneCompass token dictionary not found: {token_dict_path}\n"
                "Copy the prior_knowledge/ directory from the GeneCompass repo (or LFS pull) to "
                f"{self.prior_dir} or set GENECOMPASS_PRIOR_DIR."
            )
        with open(token_dict_path, "rb") as f:
            self.token_dictionary = pickle.load(f)
        self.gene_to_token = {
            k: v for k, v in self.token_dictionary.items()
            if k not in ("<pad>", "<mask>")
        }

    def _load_knowledges(self, config_dict: dict) -> dict:
        """Load prior-knowledge tensors required by KnowledgeBertEmbeddings."""
        self._ensure_vendor_on_path()
        from genecompass.utils import load_prior_embedding

        uses_knowledge = any(
            config_dict.get(k, False)
            for k in ("use_promoter", "use_co_exp", "use_gene_family", "use_peca_grn")
        )
        if not uses_knowledge:
            return {}

        prior = self.prior_dir
        out = load_prior_embedding(
            name2promoter_human_path=str(prior / "promoter_emb" / "human_emb_768.pickle"),
            name2promoter_mouse_path=str(prior / "promoter_emb" / "mouse_emb_768.pickle"),
            name2coexp_human_path=str(prior / "gene_co_express_emb" / "Human_dim_768_gene_28291_random.pickle"),
            name2coexp_mouse_path=str(prior / "gene_co_express_emb" / "Mouse_dim_768_gene_27444_random.pickle"),
            name2family_human_path=str(prior / "gene_family" / "Human_dim_768_gene_28291_random.pickle"),
            name2family_mouse_path=str(prior / "gene_family" / "Mouse_dim_768_gene_27934_random.pickle"),
            name2peca_human_path=str(prior / "PECA2vec" / "human_PECA_vec.pickle"),
            name2peca_mouse_path=str(prior / "PECA2vec" / "mouse_PECA_vec.pickle"),
            id2name_human_mouse_path=str(prior / "gene_list" / "Gene_id_name_dict_human_mouse.pickle"),
            token_dictionary_or_path=self.token_dictionary,
            homologous_gene_path=str(prior / "homologous_hm_token.pickle"),
        )
        return {
            "promoter": out[0],
            "co_exp": out[1],
            "gene_family": out[2],
            "peca_grn": out[3],
            "homologous_gene_human2mouse": out[4],
        }

    def load_model(self):
        self._ensure_vendor_on_path()
        from genecompass.modeling_bert import BertForMaskedLM
        from transformers import BertConfig

        self._load_token_dictionary()

        for path, desc in [
            (self.model_dir / "config.json", "config.json"),
            (self.model_dir / "pytorch_model.bin", "pytorch_model.bin"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"GeneCompass checkpoint file not found: {path}\n"
                    "Download the pre-trained model to "
                    f"{self.model_dir} or set GENECOMPASS_CKPT."
                )

        config = BertConfig.from_pretrained(str(self.model_dir))
        config_dict = config.to_dict()

        # Inject token dictionary size if not already set correctly.
        if config_dict.get("vocab_size") != len(self.token_dictionary):
            warnings.warn(
                f"GeneCompass: config vocab_size={config_dict.get('vocab_size')} "
                f"!= token_dictionary size={len(self.token_dictionary)}. "
                "Using config value."
            )

        knowledges = self._load_knowledges(config_dict)
        self.model = BertForMaskedLM(config, knowledges)

        state_dict = torch.load(
            self.model_dir / "pytorch_model.bin", map_location="cpu"
        )
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            warnings.warn(f"GeneCompass: {len(missing)} missing keys in checkpoint.")
        if unexpected:
            warnings.warn(f"GeneCompass: {len(unexpected)} unexpected keys in checkpoint.")

        self._use_cls_token = config_dict.get("use_cls_token", True)

        dtype = torch.float16 if (self.device.type == "cuda" and self.fp16) else torch.float32
        self.model = self.model.to(device=self.device, dtype=dtype)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Data preparation                                                      #
    # ------------------------------------------------------------------ #

    def prepare_data(self, adata):
        """Map gene identifiers to GeneCompass Ensembl token IDs.

        Matching priority:
          1. var_names that look like Ensembl IDs (ENSG / ENSMUSG).
          2. adata.var['ensembl_id'] or adata.var['feature_id'] column.
          3. Direct var_name lookup in token dictionary (fallback).
        """
        adata = adata.copy()
        var_names = list(adata.var_names)

        def _try_match(candidates):
            ids = [self.gene_to_token.get(g, -1) for g in candidates]
            return ids

        # 1. Ensembl IDs in var_names
        if any(str(g).startswith(("ENSG", "ENSMUSG")) for g in var_names[:20]):
            token_ids = _try_match(var_names)
        # 2. Ensembl column in var
        elif "ensembl_id" in adata.var.columns:
            token_ids = _try_match(adata.var["ensembl_id"].tolist())
        elif "feature_id" in adata.var.columns:
            token_ids = _try_match(adata.var["feature_id"].tolist())
        else:
            # last resort: try var_names directly (may still match some tokens)
            token_ids = _try_match(var_names)
            warnings.warn(
                "GeneCompass: var_names don't look like Ensembl IDs and no "
                "ensembl_id/feature_id column found. Matching may be poor."
            )

        adata.var["gc_token_id"] = token_ids
        matched = int((np.array(token_ids) >= 0).sum())
        self.genes_matched = matched
        print(f"GeneCompass: matched {matched}/{len(var_names)} genes in vocabulary")
        return adata[:, adata.var["gc_token_id"] >= 0]

    # ------------------------------------------------------------------ #
    # Embedding extraction                                                  #
    # ------------------------------------------------------------------ #

    def extract_embeddings_for_layers(
        self, adata, layer_indices: list, batch_size: int = 4
    ) -> dict:
        """Extract CLS-token embeddings from multiple transformer layers via forward hooks.

        Hooks are registered on model.bert.encoder.layer[i], which is a standard
        HuggingFace BertLayer. Its output is a tuple whose first element is the
        hidden-state tensor (batch, seq_len, hidden_size). With use_cls_token=True
        the CLS token is at position 0.
        """
        count_matrix = adata.X
        if hasattr(count_matrix, "toarray"):
            count_matrix = count_matrix.toarray()
        count_matrix = np.array(count_matrix, dtype=np.float32)

        token_ids_arr = np.array(adata.var["gc_token_id"].tolist(), dtype=np.int64)
        pad_id = self.token_dictionary["<pad>"]
        use_cls = self._use_cls_token

        max_genes = min(self.max_input_size, self.model.bert.config.max_position_embeddings - (1 if use_cls else 0))

        def _build_batch(cell_indices):
            batch_ids, batch_vals, batch_masks = [], [], []
            for ci in cell_indices:
                row = count_matrix[ci]
                nonzero_idx = np.nonzero(row)[0]
                # keep at most max_genes non-zero genes, sorted by expression descending
                if len(nonzero_idx) > max_genes:
                    order = np.argsort(row[nonzero_idx])[::-1][:max_genes]
                    nonzero_idx = nonzero_idx[order]

                sel_tokens = token_ids_arr[nonzero_idx]
                sel_values = row[nonzero_idx]
                seq_len = len(sel_tokens)

                ids = np.zeros(max_genes, dtype=np.int64)
                vals = np.zeros(max_genes, dtype=np.float32)
                mask = np.zeros(max_genes, dtype=np.float32)

                ids[:seq_len] = sel_tokens
                vals[:seq_len] = sel_values
                mask[:seq_len] = 1.0

                batch_ids.append(ids)
                batch_vals.append(vals)
                batch_masks.append(mask)

            return (
                torch.from_numpy(np.stack(batch_ids)).long(),
                torch.from_numpy(np.stack(batch_vals)),
                torch.from_numpy(np.stack(batch_masks)),
            )

        n_cells = count_matrix.shape[0]
        layer_outputs = {idx: [] for idx in layer_indices}

        def make_hook(layer_idx):
            def hook_fn(module, inp, output):
                # BertLayer returns (hidden_states, ...) — always a tuple.
                hs = output[0] if isinstance(output, tuple) else output
                # CLS is prepended at pos 0 by BertModel when use_cls_token=True.
                token_pos = 0 if use_cls else None
                if token_pos is not None:
                    emb = hs[:, token_pos, :].detach().float().cpu()
                else:
                    # mean pooling over non-pad positions when no CLS token
                    emb = hs.mean(dim=1).detach().float().cpu()
                layer_outputs[layer_idx].append(emb)
            return hook_fn

        hooks = []
        valid_layers = []
        encoder_layers = self.model.bert.encoder.layer
        for idx in layer_indices:
            if idx < 0 or idx >= len(encoder_layers):
                warnings.warn(f"GeneCompass: layer {idx} out of range [0, {len(encoder_layers)-1}], skipping.")
                continue
            h = encoder_layers[idx].register_forward_hook(make_hook(idx))
            hooks.append(h)
            valid_layers.append(idx)

        if not hooks:
            warnings.warn("GeneCompass: no valid layers to hook, aborting.")
            return {}

        dtype = torch.float16 if (self.device.type == "cuda" and self.fp16) else torch.float32
        # species=0 (human) tensor, shape [batch, 1]
        try:
            with torch.no_grad():
                for start in tqdm(range(0, n_cells, batch_size), desc="Extracting GeneCompass layers"):
                    end = min(start + batch_size, n_cells)
                    cell_idx = list(range(start, end))

                    input_ids, values, attention_mask = _build_batch(cell_idx)
                    bs = input_ids.shape[0]
                    species = torch.zeros(bs, 1, dtype=torch.long)

                    input_ids = input_ids.to(self.device)
                    values = values.to(device=self.device, dtype=dtype)
                    attention_mask = attention_mask.to(self.device)
                    species = species.to(self.device)

                    self.model(
                        input_ids=input_ids,
                        values=values,
                        attention_mask=attention_mask,
                        species=species,
                    )
        finally:
            for h in hooks:
                h.remove()

        return {
            idx: torch.cat(layer_outputs[idx], dim=0).numpy()
            for idx in valid_layers
            if layer_outputs[idx]
        }

    def get_all_layer_indices(self) -> List[int]:
        return list(range(self.n_layers_total))
