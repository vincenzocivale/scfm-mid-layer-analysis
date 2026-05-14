import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

class ClassificationEvaluator:
    def __init__(self, adata, cell_type_col):
        """
        Inizializza l'evaluator per le metriche di classificazione.

        Args:
            adata (AnnData): Oggetto AnnData contenente gli embedding e le etichette.
            cell_type_col (str): Nome della colonna in .obs che contiene le etichette dei tipi di cellule.
        """
        self.adata = adata
        self.cell_type_col = cell_type_col
        self.labels = self._prepare_labels()
        self.k_neighbors = 15 # Default for Leiden clustering

    def _prepare_labels(self):
        """Codifica le etichette in formato numerico."""
        if self.cell_type_col not in self.adata.obs.columns:
            raise ValueError(f"Colonna '{self.cell_type_col}' non trovata in adata.obs")
        
        # Rimuovi celle con etichette NaN
        valid_cells = self.adata.obs[self.cell_type_col].notna()
        self.adata = self.adata[valid_cells].copy()

        encoder = LabelEncoder()
        return encoder.fit_transform(self.adata.obs[self.cell_type_col].values)

    def _metric_1_supervised_classification(self, layer_key):
        """
        Misura performance di classificazione supervisionata con 5-fold cross-validation.
        Utilizza Logistic Regression.
        """
        X = self.adata.obsm[layer_key]
        y = self.labels

        clf = LogisticRegression(solver='liblinear', random_state=42)
        
        # 5-fold stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
        
        return {
            "Accuracy": np.mean(scores['test_accuracy']),
            "F1_macro": np.mean(scores['test_f1_macro']),
            "Precision_macro": np.mean(scores['test_precision_macro']),
            "Recall_macro": np.mean(scores['test_recall_macro'])
        }

    def _metric_2_silhouette(self, layer_key):
        """
        Calcola il Silhouette Score medio usando le etichette dei tipi di cellule.
        """
        X = self.adata.obsm[layer_key]
        
        # Campiona se ci sono troppe cellule per evitare un calcolo troppo lento
        if X.shape[0] > 20000:
            indices = np.random.choice(X.shape[0], 20000, replace=False)
            X_sample = X[indices]
            labels_sample = self.labels[indices]
        else:
            X_sample = X
            labels_sample = self.labels

        return silhouette_score(X_sample, labels_sample, metric='euclidean')

    def _metric_3_ari(self, layer_key):
        """
        Calcola l'Adjusted Rand Index tra i cluster di Leiden e le etichette vere.
        """
        # Crea un AnnData temporaneo per il clustering
        temp_adata = sc.AnnData(self.adata.obsm[layer_key])
        
        # Clustering
        sc.pp.neighbors(temp_adata, n_neighbors=self.k_neighbors, use_rep='X')
        sc.tl.leiden(temp_adata, resolution=1.0, random_state=42)
        
        leiden_clusters = temp_adata.obs['leiden'].values
        
        return adjusted_rand_score(self.labels, leiden_clusters)

    def evaluate_layer(self, layer_key):
        """
        Esegue tutte le valutazioni per un dato layer.
        """
        print(f"--- Evaluating layer: {layer_key} ---")
        
        # Controllo NaN
        embedding_data = self.adata.obsm[layer_key]
        if issparse(embedding_data):
            embedding_data = embedding_data.toarray()
        if np.isnan(embedding_data).any():
            print(f"WARNING: Layer {layer_key} contiene valori NaN. Layer saltato.")
            return None

        # Esegui metriche
        classification_metrics = self._metric_1_supervised_classification(layer_key)
        silhouette = self._metric_2_silhouette(layer_key)
        ari = self._metric_3_ari(layer_key)

        results = {
            "Layer": layer_key,
            **classification_metrics,
            "Silhouette_Score": silhouette,
            "ARI": ari
        }
        
        return results
