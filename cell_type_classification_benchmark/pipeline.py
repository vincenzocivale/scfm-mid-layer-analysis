import pandas as pd
import scanpy as sc
from .preprocessing import prepare_classification_dataset
from .evaluator import ClassificationEvaluator

def run_classification_pipeline(h5ad_path, cell_type_column='cell_type', embedding_prefix='X_layer', output_path=None):
    """
    Carica i dati, li prepara, ed esegue i test di classificazione su tutti i layer.
    
    Args:
        h5ad_path (str): Percorso al file .h5ad.
        cell_type_column (str): Nome della colonna con le etichette dei tipi di cellule.
        embedding_prefix (str): Prefisso delle chiavi obsm da testare (es. 'X_layer').
        output_path (str, optional): Percorso dove salvare il CSV con i risultati.
        
    Returns:
        pd.DataFrame: DataFrame con i risultati della valutazione.
    """
    # 1. Carica i dati
    print(f"Loading data from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    
    # 2. Preprocessing
    adata_proc = prepare_classification_dataset(adata, cell_type_col=cell_type_column)

    # 3. Inizializza l'evaluator
    evaluator = ClassificationEvaluator(adata_proc, cell_type_col=cell_type_column)

    # 4. Trova gli embedding da testare
    layers_to_test = [k for k in adata_proc.obsm.keys() if k.startswith(embedding_prefix)]
    if not layers_to_test:
        print(f"Warning: Nessun embedding trovato con prefisso '{embedding_prefix}'")
        
    print(f"Trovati {len(layers_to_test)} layer da testare: {layers_to_test}")
    
    # 5. Esegui il loop di valutazione
    all_keys_to_test = layers_to_test
    if 'X_pca' in adata_proc.obsm and 'X_pca' not in all_keys_to_test:
        all_keys_to_test.append('X_pca')
        print("Aggiunta baseline 'X_pca' alla valutazione.")

    results = []
    for layer_key in all_keys_to_test:
        if layer_key not in evaluator.adata.obsm:
            print(f"Skipping {layer_key}, not found in final adata.obsm after evaluator initialization.")
            continue
        res = evaluator.evaluate_layer(layer_key)
        if res is not None:
            results.append(res)

    if not results:
        print("Nessun risultato generato.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    
    # Rinomina la baseline per chiarezza
    results_df['Layer'] = results_df['Layer'].replace({'X_pca': 'Baseline_PCA'})

    # 6. Salva i risultati
    if output_path is not None:
        try:
            results_df.to_csv(output_path, index=False)
            print(f"Risultati salvati in: {output_path}")
        except Exception as e:
            print(f"Errore durante il salvataggio dei risultati: {e}")
            
    return results_df