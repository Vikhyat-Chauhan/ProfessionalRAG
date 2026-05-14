"""Shared test fixtures — stubs AWS, Pinecone, and the LLM so tests run offline."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ProfessionalRAG_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone")


@pytest.fixture
def visits_store():
    """In-memory replacement for the DynamoDB-backed visits table."""
    return []


@pytest.fixture
def stub_visits(monkeypatch, visits_store):
    import monitoring.visits as v
    import api.server as srv

    def fake_write(doc):
        item = {k: val for k, val in doc.items() if val is not None}
        item["pk"] = item.pop("event", "pageview")
        item["timestamp"] = item["timestamp"].isoformat()
        visits_store.append(item)

    def fake_read(days, source=None):
        if source is None:
            return list(visits_store)
        return [e for e in visits_store if e.get("source") == source]

    # Patch both the source module and the names already imported into api.server
    for mod in (v, srv):
        monkeypatch.setattr(mod, "create_table_if_needed", lambda: None, raising=False)
        monkeypatch.setattr(mod, "write_event", fake_write, raising=False)
        monkeypatch.setattr(mod, "read_events", fake_read, raising=False)
    return visits_store


class _FakeStore:
    def __init__(self, count=10):
        self._count = count

    def count(self):
        return self._count

    def query(self, vec):
        return ["chunk-a", "chunk-b"], [{"page": 1}, {"page": 2}]


class _FakeEmbedder:
    def embed_query(self, q):
        return [0.0] * 8


class _FakeReranker:
    def rerank(self, q, chunks, metas, top_k=5):
        return [(c, m, 0.9) for c, m in zip(chunks, metas)][:top_k]


class _FakeLLM:
    def generate_stream(self, q, ctx):
        yield "hello "
        yield "world"


class FakePipeline:
    def __init__(self):
        self.store = _FakeStore()
        self.embedder = _FakeEmbedder()
        self.reranker = _FakeReranker()
        self.llm = _FakeLLM()

    def ingest(self, src, force=False):
        return 1

    def query(self, q, top_k=5):
        return {
            "answer": "stub",
            "sources": [{"page": 1, "score": 0.9, "text": "chunk-a"}],
            "metrics": {"total_ms": 1.0, "llm_ms": 0.5, "cost_usd": 0.0, "top_score": 0.9},
        }


@pytest.fixture
def client(monkeypatch, stub_visits):
    """FastAPI TestClient with all externals stubbed."""
    import api.server as srv

    # Replace the heavy pipeline before the lifespan event runs
    monkeypatch.setattr(srv, "RAGPipeline", FakePipeline)

    from fastapi.testclient import TestClient

    with TestClient(srv.app) as c:
        # Lifespan rebinds module-level `pipeline`; force the fake
        srv.pipeline = FakePipeline()
        yield c


@pytest.fixture
def auth():
    return {"Authorization": "Bearer test-key"}
