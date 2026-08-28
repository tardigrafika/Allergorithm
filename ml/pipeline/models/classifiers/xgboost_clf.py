"""
XGBoost -- preuzeto iz ml/xgboost_blast_kfold_1443.py (podrazumevani
hiperparametri). Isti feature vektor kao Random Forest (abs_diff+cosine
[+blast]).
"""

import numpy as np
import xgboost as xgb

from ...common.features import build_feature_matrix, load_blast_matrices, pairwise_features_batch_same_query
from .base import PairClassifier

DEFAULT_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    n_jobs=-1,
    tree_method="hist",
)


class XGBoostPairClassifier(PairClassifier):
    def __init__(self, params: dict | None = None, extra_features: list | None = None,
                 blast_matrix_path: str | None = None, seed: int = 42, **kwargs):
        super().__init__()
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.seed = seed
        self.extra_features = extra_features or []
        self.blast_matrices = None
        if "blast_identity" in self.extra_features or "blast_score" in self.extra_features:
            assert blast_matrix_path
            self.blast_matrices = load_blast_matrices(blast_matrix_path)
        self.model = None

    def fit(self, positive_pairs, negative_pairs, embedding_matrix, id_to_index, **kwargs):
        self.set_pool(embedding_matrix, id_to_index)
        X, y = build_feature_matrix(positive_pairs, negative_pairs, embedding_matrix, id_to_index,
                                      blast_matrices=self.blast_matrices)
        scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        self.model = xgb.XGBClassifier(random_state=self.seed, scale_pos_weight=scale_pos_weight, **self.params)
        self.model.fit(X, y)

    def score_all(self, query_id) -> np.ndarray:
        query_vec = self.embedding_matrix[self.id_to_index[query_id]]
        X_candidates = pairwise_features_batch_same_query(
            query_vec, query_id, self.embedding_matrix, self.all_ids, blast_matrices=self.blast_matrices)
        return self.model.predict_proba(X_candidates)[:, 1]
