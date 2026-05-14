"""Classe base per estrazione embeddings da qualsiasi modello"""
from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, List
import h5py

class BaseEmbedder(ABC):
    """Interfaccia per estrarre embeddings da layer diversi"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
    
    @abstractmethod
    def load_model(self):
        """Carica il modello"""
        pass
    
    @abstractmethod
    def prepare_data(self, adata):
        """Prepara dati per il modello"""
        pass
    
    def extract_layer_embeddings(self, adata, layer_idx: int) -> np.ndarray:
        """Estrae embeddings da un layer specifico (default: usa extract_embeddings_for_layers)"""
        all_embs = self.extract_embeddings_for_layers(adata, [layer_idx])
        # Compatibilità: restituisci solo l'embedding del layer richiesto
        if layer_idx in all_embs:
            return all_embs[layer_idx]
        elif f'layer_{layer_idx}' in all_embs:
            return all_embs[f'layer_{layer_idx}']
        else:
            raise ValueError(f"Layer {layer_idx} non trovato tra gli embeddings estratti: {list(all_embs.keys())}")
    
    def extract_all_layers(self, adata, output_path: str, layers: List[int] = None):
        """Estrae e salva embeddings da tutti i layer"""
        if layers is None:
            layers = self.get_all_layer_indices()
        
        with h5py.File(output_path, 'w') as f:
            f.attrs['model_name'] = self.model_name
            f.attrs['n_cells'] = adata.n_obs
            
            for layer_idx in layers:
                print(f"Estraendo layer {layer_idx}...")
                embs = self.extract_layer_embeddings(adata, layer_idx)
                f.create_dataset(f'layer_{layer_idx}', data=embs, compression='gzip')
        
        print(f"✓ Salvato in {output_path}")
    
    @abstractmethod
    def get_all_layer_indices(self) -> List[int]:
        """Ritorna indici di tutti i layer"""
        pass
