"""
ESM-2 embedder -- preuzeto iz embeddings/make_emmbedings.py (glavni generator
za embeddings.pkl koji SVI downstream skriptovi koriste): facebook/esm2_t33_650M_UR50D,
mean pooling preko validnih tokena, MAX_LENGTH=1022.
"""

import numpy as np
import torch

from .base import EmbeddingModel

DEFAULT_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEFAULT_MAX_LENGTH = 1022
DEFAULT_BATCH_SIZE = 4


def mean_pool(last_hidden_state, attention_mask):
    """Identicno embeddings/make_emmbedings.py: prosek preko validnih (ne-padding) tokena."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class ESM2Embedder(EmbeddingModel):
    model_name = DEFAULT_MODEL_NAME
    embedding_dim = 1280  # 650M model hidden size

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, max_length: int = DEFAULT_MAX_LENGTH,
                 batch_size: int = DEFAULT_BATCH_SIZE, device: str | None = None):
        from transformers import AutoTokenizer, EsmModel

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def embed(self, sequences: list[str]) -> np.ndarray:
        vectors = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                batch = sequences[start:start + self.batch_size]
                tokens = self.tokenizer(batch, padding=True, truncation=True,
                                          max_length=self.max_length, return_tensors="pt")
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                outputs = self.model(**tokens)
                pooled = mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
                vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors, axis=0)
