"""Script principale per estrarre embeddings per singoli layer."""
import sys
from pathlib import Path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, './models')
import scanpy as sc
import argparse
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Estrae embedding per layer specifici e li salva nell'obsm di un unico file .h5ad.")
    parser.add_argument('--input', type=Path, required=True, help="Path al file di input .h5ad")
    parser.add_argument('--output', type=Path, required=True, help="Path al file di output .h5ad che conterrà gli embedding")
    parser.add_argument('--model', required=True, choices=['tahoe', 'scfoundation', 'scgpt'], help="Nome del modello da usare")
    parser.add_argument('--model-size', default='70m', help="Dimensione del modello (es. '70m' per Tahoe)")
    parser.add_argument('--layers', nargs='+', type=int, required=True, help="Lista dei layer da cui estrarre gli embedding")
    parser.add_argument('--batch-size', type=int, default=4, help="Dimensione del batch per l'estrazione degli embedding")
    args = parser.parse_args()

    print(f"Caricamento dati da: {args.input}")
    adata = sc.read_h5ad(args.input)

    if 'feature_name' not in adata.var.columns and 'gene_name' in adata.var.columns:
        adata.var['feature_name'] = adata.var['gene_name']
    
    # Crea la directory di output se non esiste
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Inizializzazione modello: {args.model}")
    if args.model == 'tahoe':
        from tahoe_embedder import TahoeEmbedder
        embedder = TahoeEmbedder(model_size=args.model_size)
    elif args.model == 'scfoundation':
        from scfoundation_embedder import scFoundationEmbedder
        embedder = scFoundationEmbedder()
    elif args.model == 'scgpt':
        from scgpt_embedder import scGPTEmbedder
        embedder = scGPTEmbedder()
    else:
        raise ValueError(f"Modello '{args.model}' non supportato.")

    print("Preparazione dati per il modello...")
    adata_prepared = embedder.prepare_data(adata)

    try:
        # Chiama il metodo per estrarre gli embedding per tutti i layer richiesti
        layer_embeddings = embedder.extract_embeddings_for_layers(
            adata_prepared, layer_indices=args.layers, batch_size=args.batch_size
        )
        
        if not layer_embeddings:
            print("Avviso: nessun embedding è stato estratto.")
        else:
            print(f"Aggiunta di {len(layer_embeddings)} embedding di layer all'oggetto AnnData.")
            
            # Allinea gli embedding all'AnnData originale
            # Utile se `prepare_data` ha filtrato alcune cellule.
            original_obs_names = pd.Series(adata.obs_names)
            prepared_obs_names = pd.Series(adata_prepared.obs_names)
            
            # Trova gli indici per l'allineamento
            _, common_indices_orig, common_indices_prep = np.intersect1d(
                original_obs_names, prepared_obs_names, return_indices=True
            )

            for layer, embedding in layer_embeddings.items():
                if embedding.shape[0] != adata_prepared.n_obs:
                    print(f"Errore: L'embedding per il layer {layer} ha un numero di osservazioni diverso ({embedding.shape[0]}) rispetto all'AnnData preparato ({adata_prepared.n_obs}). Salto questo layer.")
                    continue
                
                # Crea un array vuoto con le dimensioni dell'adata originale
                aligned_embedding = np.full((adata.n_obs, embedding.shape[1]), np.nan, dtype=np.float32)
                
                # Riempie l'array con i valori degli embedding nelle posizioni corrette
                aligned_embedding[common_indices_orig] = embedding[common_indices_prep]

                adata.obsm[f"X_layer_{layer}"] = aligned_embedding

            # Aggiunge metadati sugli embedding a .uns
            adata.uns['layer_embeddings'] = {
                'model': args.model,
                'model_size': args.model_size if args.model == 'tahoe' else None,
                'n_layers': len(layer_embeddings),
                'layer_keys': sorted([f"X_layer_{l}" for l in layer_embeddings.keys()]),
            }

        # Salva l'oggetto AnnData modificato in un unico file
        print(f"Salvataggio di AnnData con gli embedding in -> {args.output}")
        adata.write_h5ad(args.output, compression="gzip")
        

        print(f"Estrazione e salvataggio completati.")

        # Libera memoria GPU
        try:
            import torch
            del embedder, layer_embeddings, adata_prepared
            torch.cuda.empty_cache()
        except Exception as mem_exc:
            print(f"[AVVISO] Impossibile liberare completamente la memoria GPU: {mem_exc}")

    except Exception as e:
        print(f"Errore critico durante l'estrazione degli embedding: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
