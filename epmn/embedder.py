"""Thin wrapper around sentence-transformers for EPMN embeddings."""

from __future__ import annotations
import numpy as np
from loguru import logger


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"[Embedder] loading {model_name}")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"[Embedder] dim={self.dim}")

    def encode(self, texts: list[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.astype(np.float32)

    def encode_query(self, symptoms: list[str]) -> np.ndarray:
        """Concatenate symptoms into single query string then embed."""
        query_str = ". ".join(symptoms)
        return self.encode([query_str])[0]
