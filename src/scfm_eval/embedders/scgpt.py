"""scGPT embedder — per-layer intermediate embedding extraction."""

import json
import os
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from .base import BaseEmbedder

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL_DIR = _REPO_ROOT / "data" / "checkpoints" / "scgpt" / "whole_human"


class scGPTEmbedder(BaseEmbedder):
    # See docs/models.md for what these mean.
    pooling = "cls_token"
    expected_input = "raw_counts"

    def __init__(self, device="auto", fp16=True, model_dir=None):
        super().__init__("scgpt", device)
        self.fp16 = fp16
        self.model_dir = Path(
            model_dir or os.environ.get("SCGPT_MODEL_DIR", str(_DEFAULT_MODEL_DIR))
        )
        self.vocab = None
        self.model_configs = None
        self.genes_matched = None
        self.load_model()

    @property
    def n_layers_total(self):
        return len(self.model.transformer_encoder.layers)

    @property
    def hidden_dim(self):
        return self.model_configs.get("embsize")

    def load_model(self):
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab
        from scgpt.utils import load_pretrained

        vocab_file = self.model_dir / "vocab.json"
        config_file = self.model_dir / "args.json"
        weights_file = self.model_dir / "best_model.pt"

        for f in (vocab_file, config_file, weights_file):
            if not f.exists():
                raise FileNotFoundError(
                    f"scGPT checkpoint file not found: {f}\n"
                    f"Download the pretrained model to {self.model_dir} or set "
                    f"SCGPT_MODEL_DIR to the directory containing vocab.json, "
                    f"args.json, and best_model.pt."
                )

        pad_token = "<pad>"
        self.vocab = GeneVocab.from_file(vocab_file)
        for special in (pad_token, "<cls>", "<eoc>"):
            if special not in self.vocab:
                self.vocab.append_token(special)
        self.vocab.set_default_index(self.vocab[pad_token])

        with open(config_file) as f:
            self.model_configs = json.load(f)

        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.model_configs["embsize"],
            nhead=self.model_configs["nheads"],
            d_hid=self.model_configs["d_hid"],
            nlayers=self.model_configs["nlayers"],
            nlayers_cls=self.model_configs.get("n_layers_cls", 3),
            n_cls=1,
            vocab=self.vocab,
            dropout=self.model_configs["dropout"],
            pad_token=self.model_configs["pad_token"],
            pad_value=self.model_configs["pad_value"],
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=False,
            pre_norm=False,
        )
        load_pretrained(
            self.model,
            torch.load(weights_file, map_location=self.device),
            verbose=False,
        )
        dtype = torch.float16 if (self.device.type == "cuda" and self.fp16) else torch.float32
        self.model = self.model.to(device=self.device, dtype=dtype)
        self.model.eval()

    def prepare_data(self, adata):
        """Map gene names to scGPT vocab IDs, filter to matched genes."""
        from scgpt.tokenizer import GeneVocab  # noqa — ensure vocab is accessible

        adata = adata.copy()

        # Prefer feature_name column if present, otherwise fall back to var_names.
        if "feature_name" in adata.var.columns:
            gene_names = adata.var["feature_name"].tolist()
        else:
            gene_names = adata.var_names.tolist()

        adata.var["id_in_vocab"] = [
            self.vocab[g] if g in self.vocab else -1 for g in gene_names
        ]
        matched = int((adata.var["id_in_vocab"] >= 0).sum())
        self.genes_matched = matched
        print(f"scGPT: matched {matched}/{len(gene_names)} genes in vocabulary")

        return adata[:, adata.var["id_in_vocab"] >= 0]

    def extract_embeddings_for_layers(
        self, adata, layer_indices: list, batch_size: int = 32
    ) -> dict:
        """Extract CLS-token embeddings from multiple transformer layers via forward hooks."""
        import torch.utils.data as data_utils
        from scgpt.data_collator import DataCollator

        count_matrix = adata.X
        if hasattr(count_matrix, "toarray"):
            count_matrix = count_matrix.toarray()

        gene_ids_arr = np.array(adata.var["id_in_vocab"], dtype=int)
        pad_token_id = self.vocab[self.model_configs["pad_token"]]
        pad_value = self.model_configs["pad_value"]
        cls_id = self.vocab["<cls>"]

        class _CellDataset(data_utils.Dataset):
            def __init__(self):
                self.X = count_matrix
                self.gene_ids = gene_ids_arr

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                row = self.X[idx]
                nonzero = np.nonzero(row)[0]
                genes = np.insert(self.gene_ids[nonzero], 0, cls_id)
                values = np.insert(row[nonzero], 0, pad_value)
                return {
                    "id": idx,
                    "genes": torch.from_numpy(genes).long(),
                    "expressions": torch.from_numpy(values.astype(np.float32)),
                }

        dataset = _CellDataset()
        collator = DataCollator(
            do_padding=True,
            pad_token_id=pad_token_id,
            pad_value=pad_value,
            do_mlm=False,
            do_binning=True,
            max_length=1200,
            sampling=True,
            keep_first_n_tokens=1,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            num_workers=0,
        )

        layer_outputs = {idx: [] for idx in layer_indices}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # TransformerEncoderLayer returns a tensor (or tuple depending on version).
                out = output[0] if isinstance(output, tuple) else output
                # PyTorch 2.1+ uses NestedTensor internally when src_key_padding_mask
                # is provided. Convert to padded dense tensor before indexing.
                if out.is_nested:
                    out = out.to_padded_tensor(0.0)
                layer_outputs[layer_idx].append(out[:, 0, :].detach().float().cpu())
            return hook_fn

        hooks = []
        valid_layers = []
        for idx in layer_indices:
            try:
                h = self.model.transformer_encoder.layers[idx].register_forward_hook(
                    make_hook(idx)
                )
                hooks.append(h)
                valid_layers.append(idx)
            except IndexError:
                warnings.warn(f"scGPT: layer {idx} is out of range, skipping.")

        if not hooks:
            warnings.warn("scGPT: no valid layers to hook, aborting extraction.")
            return {}

        dtype = torch.float16 if (self.device.type == "cuda" and self.fp16) else torch.float32
        try:
            with torch.no_grad():
                for batch in tqdm(loader, desc="Extracting scGPT layers"):
                    input_gene_ids = batch["gene"].to(self.device)
                    expressions = batch["expr"].to(device=self.device, dtype=dtype)
                    src_key_padding_mask = input_gene_ids.eq(pad_token_id)
                    self.model._encode(
                        input_gene_ids,
                        expressions,
                        src_key_padding_mask=src_key_padding_mask,
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
        return list(range(len(self.model.transformer_encoder.layers)))
