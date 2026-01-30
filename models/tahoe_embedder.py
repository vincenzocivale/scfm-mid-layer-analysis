"""Tahoe-X1 embedder - FIXED VERSION"""
from models.base_embedder import BaseEmbedder
from tahoe_x1.model import ComposerTX
from tahoe_x1.utils.util import loader_from_adata
import torch
import numpy as np
import warnings
from tqdm import tqdm


class TahoeEmbedder(BaseEmbedder):
    def __init__(self, model_size="70m", device="auto", fp16=True):
        super().__init__(f"tahoe-x1-{model_size}", device)
        self.model_size = model_size
        self.fp16 = fp16
        self.load_model()

    def load_model(self):
        self.model, self.vocab, _, self.coll_cfg = ComposerTX.from_hf(
            "tahoebio/tahoe-x1", self.model_size, return_gene_embeddings=False
        )
        dtype = torch.float16 if (self.device.type == "cuda" and self.fp16) else torch.float32
        self.model = self.model.to(device=self.device, dtype=dtype)
        self.model.eval()
    
    def prepare_data(self, adata):
        """Filter genes to only those in vocabulary"""
        adata = adata.copy()
        adata.var["id_in_vocab"] = [self.vocab[g] if g in self.vocab else -1 for g in adata.var_names]
        
        num_matched = (adata.var["id_in_vocab"] >= 0).sum()
        print(f"Matched {num_matched}/{len(adata.var_names)} genes in Tahoe vocabulary")
        
        return adata[:, adata.var["id_in_vocab"] >= 0]
    
    def extract_embeddings_for_layers(self, adata, layer_indices: list, batch_size: int = 4) -> dict:
        """Extract embeddings from multiple transformer layers in a single forward pass."""
        gene_ids = np.array(adata.var["id_in_vocab"], dtype=int)
        loader = loader_from_adata(
            adata, self.coll_cfg, self.vocab, batch_size=batch_size,
            max_length=2048, gene_ids=gene_ids, num_workers=0, prefetch_factor=None
        )
        
        layer_outputs = {layer: [] for layer in layer_indices}

        def create_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                cls_embedding = output[:, 0, :].detach().float().cpu()
                layer_outputs[layer_idx].append(cls_embedding)
            return hook_fn

        hooks = []
        valid_layers = []
        for layer_idx in layer_indices:
            try:
                hook = self.model.model.transformer_encoder.layers[layer_idx].register_forward_hook(
                    create_hook_fn(layer_idx)
                )
                hooks.append(hook)
                valid_layers.append(layer_idx)
            except IndexError:
                warnings.warn(f"Layer {layer_idx} is out of range for this model. Skipping.")
        
        if not hooks:
            warnings.warn("No valid layers to hook. Aborting extraction for this model.")
            return {}

        dtype = torch.float16 if (self.device.type == "cuda" and self.fp16) else torch.float32
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting Tahoe layers"):
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        if k == "gene":
                            batch[k] = v.to(device=self.device, dtype=torch.long)
                        elif "mask" in k:
                            batch[k] = v.to(device=self.device, dtype=torch.bool)
                        else:
                            batch[k] = v.to(device=self.device, dtype=dtype)
                _ = self.model(batch, skip_decoders=True)
        
        for hook in hooks:
            hook.remove()
        
        final_embeddings = {
            layer: torch.cat(embs, dim=0).numpy() 
            for layer, embs in layer_outputs.items() 
            if embs and layer in valid_layers
        }
        
        return final_embeddings
    
    def predict_gene_expression_for_layers(self, adata, layer_indices: list) -> dict:
        """
        Predict gene expression for multiple layers in a single forward pass.
        """
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        if X.max() < 20:
            warnings.warn(
                f"\n{'='*80}\nWARNING: Data appears NORMALIZED (max={X.max():.2f}). "
                f"Tahoe requires RAW COUNTS for gene expression prediction. Results will be INVALID.\n{'='*80}\n",
                UserWarning
            )
        
        gene_ids = np.array(adata.var["id_in_vocab"], dtype=int)
        loader = loader_from_adata(
            adata, self.coll_cfg, self.vocab, batch_size=4,
            max_length=2048, gene_ids=gene_ids, num_workers=0, prefetch_factor=None
        )
        
        layer_outputs = {layer: {'predictions': [], 'targets': []} for layer in layer_indices}
        batch_layer_embs = {}

        def create_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                batch_layer_embs[layer_idx] = output.detach()
            return hook_fn

        hooks = []
        for layer_idx in layer_indices:
            try:
                hook = self.model.model.transformer_encoder.layers[layer_idx].register_forward_hook(create_hook_fn(layer_idx))
                hooks.append(hook)
            except IndexError:
                warnings.warn(f"Layer {layer_idx} is out of range for this model. Skipping.")

        if not hooks:
            warnings.warn("No valid layers to hook for gene expression prediction.")
            return {}

        dtype = torch.float16 if (self.device.type == "cuda" and self.fp16) else torch.float32
        
        try:
            with torch.no_grad():
                for batch in loader:
                    batch_layer_embs.clear()
                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            batch[k] = v.to(self.device, dtype=torch.long if k == "gene" or "mask" in k else dtype)
                    
                    _ = self.model(batch, skip_decoders=True)

                    for layer_idx, emb in batch_layer_embs.items():
                        pred = self.model.model.expression_decoder(emb.to(dtype))['pred']
                        
                        gene_batch = batch['gene']
                        expr_target = batch['expr_target']
                        pad_token_id = self.coll_cfg['pad_token_id']
                        cls_token_id = self.vocab['<cls>']
                        
                        for cell_idx in range(pred.shape[0]):
                            cell_pred = pred[cell_idx].cpu().numpy()
                            cell_genes = gene_batch[cell_idx].cpu().numpy()
                            cell_expr_target = expr_target[cell_idx].cpu().numpy()
                            
                            valid_mask = (cell_genes != pad_token_id) & (cell_genes != cls_token_id) & (cell_expr_target > 0)
                            
                            if valid_mask.sum() > 0:
                                layer_outputs[layer_idx]['predictions'].append(cell_pred[valid_mask])
                                layer_outputs[layer_idx]['targets'].append(cell_expr_target[valid_mask])
        finally:
            for hook in hooks:
                hook.remove()

        final_results = {}
        for layer_idx, outputs in layer_outputs.items():
            if not outputs['predictions']:
                final_results[f'layer_{layer_idx}'] = {'mse': np.nan, 'mae': np.nan, 'r2': np.nan, 'pearson': np.nan, 'baseline_mse': np.nan, 'n_predictions': 0}
                continue

            pred_all = np.concatenate(outputs['predictions']).astype(np.float64)
            true_all = np.concatenate(outputs['targets']).astype(np.float64)
            
            mse = np.mean((pred_all - true_all) ** 2)
            mae = np.mean(np.abs(pred_all - true_all))
            r2 = 1 - (np.sum((true_all - pred_all) ** 2) / np.sum((true_all - true_all.mean()) ** 2)) if np.sum((true_all - true_all.mean()) ** 2) > 0 else np.nan
            pearson = np.corrcoef(pred_all, true_all)[0, 1] if len(pred_all) > 1 else np.nan
            baseline_mse = np.mean((np.full_like(true_all, true_all.mean()) - true_all) ** 2)
            
            final_results[f'layer_{layer_idx}'] = {'mse': float(mse), 'mae': float(mae), 'r2': float(r2), 'pearson': float(pearson), 'baseline_mse': float(baseline_mse), 'n_predictions': int(len(pred_all))}
            
        return final_results
    
    def get_all_layer_indices(self):
        return list(range(len(self.model.model.transformer_encoder.layers)))
