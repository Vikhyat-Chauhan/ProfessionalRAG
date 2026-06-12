"""Reranker — backed by Voyage AI's hosted reranking API.

Replaces the local cross-encoder. Voyage scores each candidate chunk against the
query server-side and returns them ordered; we map the results back onto the
original `(chunk, meta, score)` tuples by index so the pipeline is unchanged.
"""

import logging

import voyageai

from config import settings

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.reranker_model
        self._client: voyageai.Client | None = None

    @property
    def client(self) -> voyageai.Client:
        if self._client is None:
            self._client = voyageai.Client(api_key=settings.voyage_api_key)
        return self._client

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

        resp = self.client.rerank(
            query=query,
            documents=chunks,
            model=self._model_name,
            top_k=min(k, len(chunks)),
        )

        ranked: list[tuple[str, dict, float]] = []
        for row in resp.results:
            i = row.index
            ranked.append((chunks[i], metadatas[i], float(row.relevance_score)))
        return ranked
