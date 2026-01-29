
import pandas as pd
import scanpy as sc
from .preprocessing import prepare_reference_dataset
from .evaluator import EmbeddingEvaluator

def run_evaluation_pipeline(h5ad_path, time_column='week', embedding_prefix='X_layer', output_path=None, use_gpu=False):
    """
    Carica dati, prepara riferimento, ed esegue test su tutti i layer trovati.
    
    Args:
        h5ad_path: Percorso al file .h5ad
        time_column: Nome colonna temporale (per root finding)
        embedding_prefix: Prefisso delle chiavi obsm da testare (es. 'X_layer')
        
    Returns:
        pd.DataFrame con i risultati.
    """
    # 1. Load Data
    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    
    # 2. Preprocessing & Root Finding
    adata_ref, root_idx = prepare_reference_dataset(adata, time_col=time_column, use_gpu=use_gpu)
    
    # 3. Init Evaluator
    evaluator = EmbeddingEvaluator(adata_ref, root_cell_index=root_idx)
    
    # 4. Setup Ground Truth (using PCA of HVGs)
    evaluator.setup_reference(use_rep='X_pca', n_neighbors=30)
    
    # 5. Find Embeddings to Test
    layers_to_test = [k for k in adata_ref.obsm.keys() if k.startswith(embedding_prefix)]
    if not layers_to_test:
        print(f"Warning: No embeddings found starting with '{embedding_prefix}'")
        return pd.DataFrame()
        
    print(f"Found {len(layers_to_test)} layers to test: {layers_to_test}")
    
    # 6. Run Loop
    results = []
    for layer in layers_to_test:
        res = evaluator.evaluate_layer(layer)
        if res is not None:
            results.append(res)

    results = pd.DataFrame(results)
    if output_path is not None:
        results.to_csv(output_path, index=False)
    return results