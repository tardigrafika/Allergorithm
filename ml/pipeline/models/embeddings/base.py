"""
Apstraktni interfejs za embedding modele (ESM-2, Ankh, ...). Svaki
konkretan embedder ucitava svoj PLM i produkuje pooled vektor po proteinu.

Svi postojeci downstream skriptovi (RF/MLP/XGBoost/RRF/...) rade nad VEC
IZRACUNATIM, keširanim embeddinzima (embeddings.pkl) -- ovaj interfejs
pokriva KORAK GENERISANJA (embeddings/generate_*.py skriptovi), ne
klasifikaciju/rangiranje (to je PairClassifier, videti classifiers/base.py).
"""

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    """model_name, embedding_dim: metapodaci, popunjava svaka konkretna klasa."""

    model_name: str
    embedding_dim: int

    @abstractmethod
    def embed(self, sequences: list[str]) -> np.ndarray:
        """sequences: lista FASTA aminokiselinskih sekvenci.
        Vraca (len(sequences), embedding_dim) pooled vektore (jedan po sekvenci)."""
        raise NotImplementedError
