"""Embedders behind one interface.

- HashEmbedder: character n-gram hashing. Offline, free, deterministic. A
  weak but honest lexical baseline that works for any script, including
  Kannada. Used for tests and as a retrieval baseline.
- SentenceTransformerEmbedder: real multilingual models (e.g. BGE-M3,
  multilingual-e5). Chosen by a retrieval benchmark, not by guess.
"""
import hashlib
import math
from abc import ABC, abstractmethod

from app.rag.text import char_ngrams


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder(Embedder):
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for g in char_ngrams(text):
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts):
        return [self._one(t) for t in texts]


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def build_embedder(settings) -> Embedder:
    if settings.embedder == "hash":
        return HashEmbedder()
    if settings.embedder == "st":
        return SentenceTransformerEmbedder(settings.embed_model)
    raise ValueError(f"Unknown EMBEDDER '{settings.embedder}' (use hash | st)")
