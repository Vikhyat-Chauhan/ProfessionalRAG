"""ChromaDB vector store with per-source fingerprint-based cache invalidation."""

import hashlib
import logging

import chromadb

from config import settings

log = logging.getLogger(__name__)


def _source_tag(fingerprint: str) -> str:
    """Short deterministic tag to identify chunks from a given source."""
    return hashlib.md5(fingerprint.encode()).hexdigest()[:12]


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
        self._next_id: int | None = None

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

    def _get_next_id(self) -> int:
        """Return an auto-incrementing chunk ID that won't collide."""
        if self._next_id is None:
            self._next_id = self.collection.count()
        val = self._next_id
        self._next_id += 1
        return val

    def needs_ingestion(self, fingerprint: str) -> bool:
        """Check whether the collection already holds data for this fingerprint."""
        fp_id = f"__fp_{_source_tag(fingerprint)}__"
        try:
            existing = self.collection.get(ids=[fp_id])
            if existing["documents"] and existing["documents"][0] == fingerprint:
                return False
        except Exception:
            pass
        return True

    def ingest(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        fingerprint: str,
    ) -> int:
        """Add chunks for a source. Removes old chunks from the same source first."""
        tag = _source_tag(fingerprint)

        # Remove previous chunks from this source
        self._remove_source(tag)

        # Tag every chunk with its source
        for meta in metadatas:
            meta["source_tag"] = tag
            meta["type"] = "chunk"

        # Store chunks in batches
        batch_size = 5000
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            self.collection.add(
                ids=[f"chunk_{self._get_next_id()}" for _ in range(start, end)],
                documents=chunks[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

        # Store fingerprint marker
        embed_dim = len(embeddings[0]) if embeddings else 768
        fp_id = f"__fp_{tag}__"
        # Upsert in case it already exists
        self.collection.upsert(
            ids=[fp_id],
            documents=[fingerprint],
            embeddings=[[0.0] * embed_dim],
            metadatas=[{"type": "fingerprint", "source_tag": tag}],
        )

        count = len(chunks)
        log.info("Ingested %d chunks (source tag: %s)", count, tag)
        return count

    def _remove_source(self, tag: str) -> None:
        """Delete all chunks and the fingerprint marker for a given source tag."""
        try:
            self.collection.delete(where={"source_tag": tag})
            log.info("Cleared previous chunks for source tag %s", tag)
        except Exception:
            # No matching docs — nothing to delete
            pass

    def query(
        self,
        query_embedding: list[float],
        n_results: int | None = None,
    ) -> tuple[list[str], list[dict]]:
        """Return candidate chunks and their metadata."""
        n = n_results or settings.candidate_count
        total = self.collection.count()
        if total == 0:
            return [], []
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n, total),
            where={"type": "chunk"},
        )
        chunks = results["documents"][0]
        metas = [m if m else {} for m in results["metadatas"][0]]
        return chunks, metas
