"""
Apstraktni interfejs za "klasifikatore" (u sirem smislu -- svaki metod koji
za dat query protein rangira SVE kandidate): cosine similarity, Random
Forest, MLP, XGBoost, Hadamard bilinear.

fit() prima gotove parne (positive/negative) podatke -- ne radi split ili
negative sampling sam (to je common/splitting.py i common/negatives.py,
deljeno preko svih modela). score_all() je ono sto se koristi i za
klasifikacione metrike i za retrieval evaluaciju (common/evaluation.py).
"""

from abc import ABC, abstractmethod

import numpy as np


class PairClassifier(ABC):
    """
    embedding_matrix, id_to_index: pool nad kojim se rangira (fiksni candidate
    universe), postavlja se preko set_pool() pre score_all() poziva.
    """

    def __init__(self):
        self.embedding_matrix: np.ndarray | None = None
        self.id_to_index: dict | None = None
        self.all_ids: list | None = None

    def set_pool(self, embedding_matrix: np.ndarray, id_to_index: dict):
        self.embedding_matrix = embedding_matrix
        self.id_to_index = id_to_index
        # eksplicitna, poredana lista (ne oslanjanje na dict.keys() redosled
        # na pozivnom mestu) -- poravnata sa embedding_matrix redovima
        ordered = [None] * len(id_to_index)
        for aid, idx in id_to_index.items():
            ordered[idx] = aid
        self.all_ids = ordered

    @abstractmethod
    def fit(self, positive_pairs: list, negative_pairs: list, embedding_matrix: np.ndarray,
            id_to_index: dict, **kwargs) -> None:
        """positive_pairs: [{"id_1","id_2",...}, ...]. negative_pairs: [(a,b), ...].
        No-op za modele bez treninga (cosine)."""
        raise NotImplementedError

    @abstractmethod
    def score_all(self, query_id) -> np.ndarray:
        """Vraca skor za SVAKOG kandidata u pool-u (poravnato sa embedding_matrix
        redosledom, iz set_pool/fit poziva). Self-skor NIJE maskiran ovde --
        to radi common/evaluation.py (dosledno svuda: self se iskljucuje SAMO
        pri racunanju ranga, ne u sirovom skoru)."""
        raise NotImplementedError

    def predict_proba_pairs(self, X_pairs_query_ids, X_pairs_candidate_ids) -> np.ndarray:
        """Opciono: skor za EKSPLICITNU listu (query,candidate) parova (koristi
        se za klasifikacione metrike na test skupu). Podrazumevana implementacija
        poziva score_all po jedinstvenom query_id-u -- konkretne klase mogu
        override-ovati radi efikasnosti (npr. RF/MLP racunaju direktno na X)."""
        raise NotImplementedError
