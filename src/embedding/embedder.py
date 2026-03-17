import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


class BaseEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class PritamdekaEmbedder(BaseEmbedder):
    def __init__(self, model_name="pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )


class NeuMLEmbedder(BaseEmbedder):
    def __init__(self, model_name="NeuML/pubmedbert-base-embeddings"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )