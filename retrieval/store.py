"""ChromaDB vector store with fingerprint-based cache invalidation."""

import logging

import chromadb

from config import settings

log = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        path: str | None = None,
        collection_name: str | None = None,
    ):
        self._path = path or settings.chroma_path
        self._collection_name = collection_name or settings.collection_name
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self._path)
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                self._collection_name
            )
        return self._collection

    def needs_ingestion(self, fingerprint: str) -> bool:
        """Check whether the collection already holds data for this fingerprint."""
        if self.collection.count() == 0:
            return True
        existing = self.collection.get(ids=["__fingerprint__"])
        if not existing["documents"]:
            return True
        return existing["documents"][0] != fingerprint

    def ingest(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        fingerprint: str,
    ) -> int:
        """Clear existing data and store new chunks + a fingerprint marker."""
        # Reset collection
        self.client.delete_collection(self._collection_name)
        self._collection = self.client.get_or_create_collection(
            self._collection_name
        )

        # Store chunks in batches (ChromaDB has a batch limit)
        # Chunks go first so the collection dimension is set by real embeddings
        batch_size = 5000
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            self.collection.add(
                ids=[f"chunk_{i}" for i in range(start, end)],
                documents=chunks[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

        # Store fingerprint marker with a zero-vector matching the embedding dim
        embed_dim = len(embeddings[0]) if embeddings else 768
        self.collection.add(
            ids=["__fingerprint__"],
            documents=[fingerprint],
            embeddings=[[0.0] * embed_dim],
            metadatas=[{"type": "fingerprint"}],
        )

        count = self.collection.count() - 1  # exclude fingerprint
        log.info("Ingested %d chunks", count)
        return count

    def query(
        self,
        query_embedding: list[float],
        n_results: int | None = None,
    ) -> tuple[list[str], list[dict]]:
        """Return candidate chunks and their metadata."""
        n = n_results or settings.candidate_count
        # Request extra to account for fingerprint doc
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n + 1, self.collection.count()),
            where={"type": {"$ne": "fingerprint"}},
        )
        chunks = results["documents"][0]
        metas = [m if m else {} for m in results["metadatas"][0]]
        return chunks, metas
