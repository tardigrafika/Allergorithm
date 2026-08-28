"""
Ankh embedder -- preuzeto iz embeddings/generate_ankh_embeddings.py:
ElnaggarLab/ankh-base (T5EncoderModel, ~450M), mean pooling, MAX_LENGTH=1022.

VAZNO (iz originalnog skripta): T5-stil arhitekture su poznato numericki
nestabilne pod FP16 (davale su NaN embeddinge) -- FORSIRA SE FP32, nema
model.half()/autocast opcije ovde (za razliku od ESM2Embedder-a).
"""

import numpy as np
import torch

from .base import EmbeddingModel

DEFAULT_MODEL_NAME = "ElnaggarLab/ankh-base"
DEFAULT_MAX_LENGTH = 1022
DEFAULT_BATCH_SIZE = 8


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class AnkhEmbedder(EmbeddingModel):
    model_name = DEFAULT_MODEL_NAME
    embedding_dim = 768  # ankh-base hidden size

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, max_length: int = DEFAULT_MAX_LENGTH,
                 batch_size: int = DEFAULT_BATCH_SIZE, device: str | None = None):
        from transformers import AutoTokenizer, T5EncoderModel

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)  # NE AutoModel -- T5Model
                                                                    # (encoder-decoder) trazi decoder input
        self.model.to(self.device)
        self.model.eval()

    def embed(self, sequences: list[str]) -> np.ndarray:
        vectors = []
        truncated = [s[:self.max_length] for s in sequences]
        with torch.no_grad():  # namerno BEZ autocast/FP16 -- T5 numericka nestabilnost (NaN)
            for start in range(0, len(truncated), self.batch_size):
                batch = truncated[start:start + self.batch_size]
                tokens = self.tokenizer(batch, padding=True, truncation=True,
                                          max_length=self.max_length, return_tensors="pt")
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                outputs = self.model(**tokens)
                pooled = mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
                vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors, axis=0)
