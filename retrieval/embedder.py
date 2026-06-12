"""Embedding wrapper — backed by Voyage AI's hosted embedding API.

No local model weights: embeddings come from a Voyage-hosted model
(`settings.embedding_model`), so the container stays tiny and cold starts fast.
The `input_type` parameter ("query" vs "document") lets Voyage optimise the
embedding for retrieval, replacing the BGE-style text prefixes the old local
model needed.
"""

import logging

import voyageai

from config import settings

log = logging.getLogger(__name__)

# Voyage accepts at most 128 inputs per embed call.
_EMBED_BATCH = 128


class Embedder:
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.embedding_model
        self._client: voyageai.Client | None = None

    @property
    def client(self) -> voyageai.Client:
        if self._client is None:
            self._client = voyageai.Client(api_key=settings.voyage_api_key)
        return self._client

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start : start + _EMBED_BATCH]
            resp = self.client.embed(
                batch, model=self._model_name, input_type=input_type
            )
            out.extend(resp.embeddings)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="query")[0]

    def embed_documents(
        self, texts: list[str], show_progress: bool = True
    ) -> list[list[float]]:
        if show_progress:
            log.info("Embedding %d chunks via %s", len(texts), self._model_name)
        return self._embed(texts, input_type="document")
