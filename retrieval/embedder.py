"""Embedding wrapper — handles BGE-style query/passage prefixes."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            log.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode(f"query: {text}")
        return vec.tolist()

    def embed_documents(
        self, texts: list[str], show_progress: bool = True
    ) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(prefixed, show_progress_bar=show_progress)
        return vecs.tolist() if isinstance(vecs, np.ndarray) else vecs
