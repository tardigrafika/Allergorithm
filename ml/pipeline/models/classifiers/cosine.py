"""
Cosine similarity -- baseline metod korisen u SVIM skriptovima sesije
(analysis/hitsk_and_mrr.py i dalje). fit() je no-op (nema parametara za
treniranje) -- ovo je referentna tacka spram koje se svi ostali klasifikatori
porede.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import PairClassifier


class CosineSimilarity(PairClassifier):
    def __init__(self, **kwargs):
        super().__init__()
        self._cosine_matrix = None

    def fit(self, positive_pairs, negative_pairs, embedding_matrix, id_to_index, **kwargs):
        self.set_pool(embedding_matrix, id_to_index)
        self._cosine_matrix = cosine_similarity(embedding_matrix)

    def score_all(self, query_id) -> np.ndarray:
        qidx = self.id_to_index[query_id]
        return self._cosine_matrix[qidx].copy()
