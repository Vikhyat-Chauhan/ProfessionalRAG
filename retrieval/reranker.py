"""Cross-encoder reranker."""

import logging

from sentence_transformers import CrossEncoder

from config import settings

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.reranker_model
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            log.info("Loading reranker model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[str],
        metadatas: list[dict],
        top_k: int | None = None,
    ) -> list[tuple[str, dict, float]]:
        """Score and sort chunks by relevance. Returns (chunk, meta, score)."""
        k = top_k or settings.top_k
        if not chunks:
            return []

        pairs = [[query, chunk] for chunk in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, chunks, metadatas), key=lambda x: x[0], reverse=True
        )
        return [(chunk, meta, float(score)) for score, chunk, meta in ranked[:k]]
