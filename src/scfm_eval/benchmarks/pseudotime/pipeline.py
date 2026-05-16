import pandas as pd
import scanpy as sc
from .preprocessing import prepare_reference_dataset
from .evaluator import EmbeddingEvaluator


def run_evaluation_pipeline(h5ad_path, time_column='week', embedding_prefix='X_layer',
                            output_path=None, use_gpu=False, include_pca_baseline=True):
    """Run pseudotime evaluation on every embedding layer in the h5ad.

    The reference DPT is built on PCA-of-HVG; this is a *baseline*, not a biological
    ground truth. The `Pseudotime_Corr_vs_Time` column reports correlation against
    the real `time_column` annotation (week / day / stage), which is the actual
    biological signal where available.
    """
    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)

    adata_ref, root_idx = prepare_reference_dataset(adata, time_col=time_column, use_gpu=use_gpu)

    evaluator = EmbeddingEvaluator(adata_ref, root_cell_index=root_idx, time_col=time_column)
    evaluator.setup_reference(use_rep='X_pca', n_neighbors=30)

    layers_to_test = [k for k in adata_ref.obsm.keys() if k.startswith(embedding_prefix)]
    if include_pca_baseline and 'X_pca' in adata_ref.obsm:
        layers_to_test.append('X_pca')

    if not layers_to_test:
        print(f"Warning: no embeddings found starting with '{embedding_prefix}'")
        return pd.DataFrame()

    print(f"Found {len(layers_to_test)} reps to test: {layers_to_test}")

    results = []
    for layer in layers_to_test:
        res = evaluator.evaluate_layer(layer)
        if res is not None:
            results.append(res)

    results = pd.DataFrame(results)
    if 'Layer' in results.columns:
        results['Layer'] = results['Layer'].replace({'X_pca': 'Baseline_PCA'})
    if output_path is not None:
        results.to_csv(output_path, index=False)
    return results
