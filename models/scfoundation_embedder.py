"""scFoundation embedder - FIXED VERSION"""
import sys
sys.path.insert(0, './models')
from models.base_embedder import BaseEmbedder
from load import load_model_frommmf, gatherData
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from tqdm import tqdm

class scFoundationEmbedder(BaseEmbedder):
    
    def __init__(self, device="auto"):
        super().__init__("scfoundation", device)
        self.gene_list = self._load_gene_list()
        self.load_model()
    
    def _load_gene_list(self):
        df = pd.read_csv('./models/OS_scRNA_gene_index.19264.tsv', sep='\t')
        return list(df['gene_name'])
    
    def load_model(self):
        self.model, self.config = load_model_frommmf('./models/models.ckpt', 'cell')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def prepare_data(self, adata):
        """Prepare data for scFoundation - assumes already normalized+log1p"""
        if issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X
        
        # Create mapping: use Ensembl ID as index, feature_name as mapping
        gene_map = dict(zip(adata.var_names, adata.var['feature_name']))
        
        # Build expression matrix using gene symbols
        data_dict = {}
        for i, ensembl_id in enumerate(adata.var_names):
            gene_symbol = gene_map.get(ensembl_id, ensembl_id)
            if gene_symbol in self.gene_list:
                if gene_symbol not in data_dict:
                    data_dict[gene_symbol] = X[:, i]
                else:
                    # Duplicate gene name - take max expression
                    data_dict[gene_symbol] = np.maximum(data_dict[gene_symbol], X[:, i])
        
        # Create DataFrame with model's gene list
        X_new = np.zeros((X.shape[0], len(self.gene_list)))
        for j, gene in enumerate(self.gene_list):
            if gene in data_dict:
                X_new[:, j] = data_dict[gene]
        
        adata_new = sc.AnnData(X_new)
        adata_new.var_names = self.gene_list
        
        # Transfer log_total_count
        if 'log_total_count' in adata.obs.columns:
            adata_new.obs['log_total_count'] = adata.obs['log_total_count'].values
        else:
            if issparse(adata.X):
                original_sum = np.array(adata.X.sum(axis=1)).flatten()
            else:
                original_sum = adata.X.sum(axis=1)
            adata_new.obs['log_total_count'] = np.log10(original_sum + 1)
        
        matched = len([g for g in data_dict.keys() if g in self.gene_list])
        print(f"scFoundation: matched {matched}/{len(self.gene_list)} genes")
        
        return adata_new
    
    def extract_embeddings_for_layers(self, adata, layer_indices: list, batch_size: int = 1) -> dict:
        """
        Extracts embeddings from multiple layers, processing cells in batches.
        """
        layer_outputs = {layer: [] for layer in layer_indices}
        n_obs = adata.n_obs


        from torch.cuda.amp import autocast
        with torch.no_grad():
            for i in tqdm(range(0, n_obs, batch_size), desc=f"Extracting scFoundation layers with batch size {batch_size}"):
                batch_indices = range(i, min(i + batch_size, n_obs))

                # Prepare batch data
                if issparse(adata.X):
                    expr_batch = adata.X[batch_indices].toarray()
                else:
                    expr_batch = adata.X[batch_indices]

                total_batch = adata.obs['log_total_count'].iloc[batch_indices].values

                # Batch processing logic
                batch_tensors = []
                for j in range(len(batch_indices)):
                    expr = expr_batch[j]
                    total = total_batch[j]
                    gene_x = torch.tensor(list(expr) + [4.0, total], device=self.device)
                    batch_tensors.append(gene_x)

                gene_x_batch = torch.stack(batch_tensors)
                gene_ids_batch = torch.arange(19266, device=self.device).unsqueeze(0).repeat(len(batch_indices), 1)

                value_labels = gene_x_batch > 0
                x_batch, x_padding = gatherData(gene_x_batch, value_labels, self.config['pad_token_id'])
                pos_ids_batch, _ = gatherData(gene_ids_batch, value_labels, self.config['pad_token_id'])

                # Forward pass for the batch in mixed precision
                with autocast():
                    x = self.model.token_emb(x_batch.unsqueeze(2).float(), output_weight=0)
                    x += self.model.pos_emb(pos_ids_batch)

                    for idx, mod in enumerate(self.model.encoder.transformer_encoder):
                        x = mod(x, src_key_padding_mask=x_padding)
                        if idx in layer_indices:
                            cell_emb = x[:, -1, :].cpu()
                            layer_outputs[idx].append(cell_emb)


        # Concatenate embeddings for each layer e libera memoria
        final_embeddings = {}

        import gc
        for layer, embs in layer_outputs.items():
            if embs:
                arr = torch.cat(embs, dim=0).numpy()
                final_embeddings[layer] = arr
                # Libera memoria dei batch
                del embs, arr
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        # SUGGERIMENTO: per risparmiare RAM, puoi salvare ogni embedding di layer su disco qui invece di accumulare tutto in final_embeddings
        # Esempio:
        # np.save(f"embedding_layer_{layer}.npy", arr)
        # e poi caricarli uno per volta quando servono

        return final_embeddings
    
    def predict_gene_expression(self, adata, layer_idx: int) -> dict:
        predictions = []
        targets = []
        
        with torch.no_grad():
            for i in range(adata.n_obs):
                expr = adata.X[i] if not issparse(adata.X) else adata.X[i].toarray()[0]
                expr = expr.tolist() if hasattr(expr, 'tolist') else list(expr)
                total = adata.obs['log_total_count'].iloc[i]
                
                gene_x = torch.tensor(expr + [4.0, total]).unsqueeze(0).to(self.device)
                gene_ids = torch.arange(19266, device=self.device).unsqueeze(0)
                
                value_labels = gene_x > 0
                x_gathered, x_padding = gatherData(gene_x, value_labels, self.config['pad_token_id'])
                pos_ids, _ = gatherData(gene_ids, value_labels, self.config['pad_token_id'])
                
                x = self.model.token_emb(x_gathered.unsqueeze(2).float(), output_weight=0)
                x += self.model.pos_emb(pos_ids)
                
                for idx, mod in enumerate(self.model.encoder.transformer_encoder):
                    x = mod(x, src_key_padding_mask=x_padding)
                    if idx == layer_idx:
                        break
                
                x = self.model.decoder_embed(x)
                x = self.model.decoder(x)
                x = self.model.norm(x)
                pred = self.model.to_final(x).squeeze(-1).squeeze(0).cpu().numpy()
                
                true_vals = x_gathered.squeeze(0).cpu().numpy()
                
                predictions.append(pred)
                targets.append(true_vals)
        
        pred_all = np.concatenate(predictions)
        true_all = np.concatenate(targets)
        
        mse = np.mean((pred_all - true_all) ** 2)
        mae = np.mean(np.abs(pred_all - true_all))
        baseline_pred = np.concatenate([np.full_like(t, t.mean()) for t in targets])
        baseline_mse = np.mean((baseline_pred - true_all) ** 2)
        
        return {'mse': float(mse), 'mae': float(mae), 'baseline_mse': float(baseline_mse)}
    
    def get_all_layer_indices(self):
        return list(range(len(self.model.encoder.transformer_encoder)))
