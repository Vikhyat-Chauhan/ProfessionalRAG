"""Pinecone vector store with per-source fingerprint-based cache invalidation."""

import hashlib
import logging

from pinecone import Pinecone, ServerlessSpec

from config import settings

log = logging.getLogger(__name__)


def _source_tag(fingerprint: str) -> str:
    """Short deterministic tag to identify chunks from a given source."""
    return hashlib.md5(fingerprint.encode()).hexdigest()[:12]


class VectorStore:
    def __init__(self, index_name: str | None = None):
        self._index_name = index_name or settings.pinecone_index
        self._pc: Pinecone | None = None
        self._index = None

    @property
    def pc(self) -> Pinecone:
        if self._pc is None:
            self._pc = Pinecone(api_key=settings.pinecone_api_key)
        return self._pc

    @property
    def index(self):
        if self._index is None:
            # Create index if it doesn't exist
            existing = [idx.name for idx in self.pc.list_indexes()]
            if self._index_name not in existing:
                log.info("Creating Pinecone index: %s", self._index_name)
                self.pc.create_index(
                    name=self._index_name,
                    dimension=settings.embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=settings.pinecone_cloud,
                        region=settings.pinecone_region,
                    ),
                )
            self._index = self.pc.Index(self._index_name)
        return self._index

    def count(self) -> int:
        """Return total vector count in the index."""
        stats = self.index.describe_index_stats()
        return stats.total_vector_count

    def needs_ingestion(self, fingerprint: str) -> bool:
        """Check whether the index already holds data for this fingerprint."""
        fp_id = f"__fp_{_source_tag(fingerprint)}__"
        try:
            result = self.index.fetch(ids=[fp_id])
            vec = result.vectors.get(fp_id)
            if vec and vec.metadata.get("fingerprint") == fingerprint:
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

        # Upsert chunks in batches (Pinecone limit: 100 vectors per upsert)
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            vectors = []
            for i in range(start, end):
                vectors.append({
                    "id": f"chunk_{tag}_{i}",
                    "values": embeddings[i],
                    "metadata": {
                        **metadatas[i],
                        "text": chunks[i],
                    },
                })
            self.index.upsert(vectors=vectors)

        # Store fingerprint marker (tiny non-zero vector — Pinecone rejects all-zeros)
        embed_dim = len(embeddings[0]) if embeddings else settings.embedding_dim
        fp_id = f"__fp_{tag}__"
        marker_vec = [1e-7] * embed_dim
        self.index.upsert(vectors=[{
            "id": fp_id,
            "values": marker_vec,
            "metadata": {
                "type": "fingerprint",
                "source_tag": tag,
                "fingerprint": fingerprint,
            },
        }])

        count = len(chunks)
        log.info("Ingested %d chunks (source tag: %s)", count, tag)
        return count

    def _remove_source(self, tag: str) -> None:
        """Delete all chunks and the fingerprint marker for a given source tag."""
        try:
            # Pinecone serverless supports delete by metadata filter
            self.index.delete(filter={"source_tag": {"$eq": tag}})
            log.info("Cleared previous chunks for source tag %s", tag)
        except Exception:
            pass

    def query(
        self,
        query_embedding: list[float],
        n_results: int | None = None,
    ) -> tuple[list[str], list[dict]]:
        """Return candidate chunks and their metadata."""
        n = n_results or settings.candidate_count
        total = self.count()
        if total == 0:
            return [], []

        results = self.index.query(
            vector=query_embedding,
            top_k=min(n, total),
            include_metadata=True,
            filter={"type": {"$eq": "chunk"}},
        )

        chunks = []
        metas = []
        for match in results.matches:
            meta = dict(match.metadata) if match.metadata else {}
            text = meta.pop("text", "")
            chunks.append(text)
            metas.append(meta)

        return chunks, metas
