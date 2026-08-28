"""
Random Forest -- preuzeto iz ml/random_forest_baseline.py (podrazumevani
hiperparametri) i ml/random_forest_blast_1443.py (opciono extra_features
prosirenje sa BLAST identity/score kolonama).

Feature vektor: pairwise_features (abs_diff[1280] + cosine[1]
[+ blast_identity[1] + blast_score[1] ako je extra_features postavljen]).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier

from ...common.features import build_feature_matrix, load_blast_matrices, pairwise_features_batch_same_query
from .base import PairClassifier

DEFAULT_PARAMS = dict(
    n_estimators=300,
    max_depth=12,          # plitka stabla: 1281 feature-a, samo ~2600-15000 trening redova
    min_samples_leaf=3,
    class_weight="balanced",
    n_jobs=-1,
)


class RandomForestPairClassifier(PairClassifier):
    def __init__(self, params: dict | None = None, extra_features: list | None = None,
                 blast_matrix_path: str | None = None, seed: int = 42, **kwargs):
        super().__init__()
        self.params = {**DEFAULT_PARAMS, **(params or {}), "random_state": seed}
        self.extra_features = extra_features or []
        self.blast_matrices = None
        if "blast_identity" in self.extra_features or "blast_score" in self.extra_features:
            assert blast_matrix_path, "extra_features trazi blast_identity/blast_score ali blast_matrix_path nije dat"
            self.blast_matrices = load_blast_matrices(blast_matrix_path)
        self.model = SkRandomForestClassifier(**self.params)

    def fit(self, positive_pairs, negative_pairs, embedding_matrix, id_to_index, **kwargs):
        self.set_pool(embedding_matrix, id_to_index)
        X, y = build_feature_matrix(positive_pairs, negative_pairs, embedding_matrix, id_to_index,
                                      blast_matrices=self.blast_matrices)
        self.model.fit(X, y)

    def score_all(self, query_id) -> np.ndarray:
        query_vec = self.embedding_matrix[self.id_to_index[query_id]]
        X_candidates = pairwise_features_batch_same_query(
            query_vec, query_id, self.embedding_matrix, self.all_ids, blast_matrices=self.blast_matrices)
        return self.model.predict_proba(X_candidates)[:, 1]

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_
