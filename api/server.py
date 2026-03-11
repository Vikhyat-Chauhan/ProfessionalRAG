"""FastAPI server — production-ready REST API for the RAG pipeline."""

import json
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from pipeline import RAGPipeline
from monitoring.metrics import metrics
from config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="ProfessionalRAG", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda req, exc: JSONResponse(
    status_code=429, content={"detail": "Rate limit exceeded. Try again later."},
))

ALLOWED_ORIGINS = [
    "https://vikhyatchauhan.com",
    "https://vikhyatchauhan.com/chat",
    "http://localhost:4321",               # local Astro dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

pipeline = RAGPipeline()


@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Reject requests without a valid API key (skip health check)."""
    if request.url.path == "/health":
        return await call_next(request)
    if not settings.api_key:
        return await call_next(request)
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {settings.api_key}":
        return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
    return await call_next(request)


# ── Request / Response models ──────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class IngestRequest(BaseModel):
    pdf_path: str | list[str]
    force: bool = False

class Source(BaseModel):
    page: int | str
    score: float
    text: str

class QueryMetricsResponse(BaseModel):
    total_ms: float
    llm_ms: float
    cost_usd: float
    top_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    metrics: QueryMetricsResponse


class IngestResponse(BaseModel):
    chunks: int
    message: str


class EvalRequest(BaseModel):
    golden_path: str
    use_judge: bool = True


class ChatRequest(BaseModel):
    message: str
    top_k: int = 5


# ── Endpoints ──────────────────────────────────────────────────────────
@app.post("/ingest", response_model=IngestResponse)
@limiter.limit("5/minute")
def ingest(request: Request, req: IngestRequest):
    """Ingest one or more PDFs into the vector store."""
    paths = req.pdf_path if isinstance(req.pdf_path, list) else [req.pdf_path]
    total = 0
    try:
        for path in paths:
            total += pipeline.ingest(path, force=req.force)
        return IngestResponse(chunks=total, message=f"Ingested {total} chunks from {len(paths)} file(s)")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
def query(request: Request, req: QueryRequest):
    """Query the RAG pipeline."""
    if pipeline.store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested. POST /ingest first.",
        )
    result = pipeline.query(req.question, top_k=req.top_k)
    return result


@app.post("/chat")
@limiter.limit("10/minute")
def chat(request: Request, req: ChatRequest):
    """Streaming chat endpoint — returns SSE stream of tokens."""
    if pipeline.store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested. POST /ingest first.",
        )

    metrics.start_query(req.message)

    # Retrieve + rerank
    from config import settings
    k = req.top_k or settings.top_k

    with metrics.track_latency("retrieval"):
        query_vec = pipeline.embedder.embed_query(req.message)
        chunks, metas = pipeline.store.query(query_vec)

    with metrics.track_latency("rerank"):
        ranked = pipeline.reranker.rerank(req.message, chunks, metas, top_k=k)

    top_score = ranked[0][2] if ranked else 0.0
    metrics.record_retrieval(top_score, len(ranked))

    context = [(chunk, meta) for chunk, meta, _ in ranked]
    sources = [
        {"page": m.get("page", "?"), "score": round(s, 4), "text": c[:200]}
        for c, m, s in ranked
    ]

    def event_stream():
        # Send sources first as a JSON event
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Stream LLM tokens
        for token in pipeline.llm.generate_stream(req.message, context):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        # Send done signal with metrics
        query_metrics = metrics.finish_query()
        yield f"data: {json.dumps({'type': 'done', 'metrics': {'total_ms': round(query_metrics.total_latency_ms, 1), 'cost_usd': round(query_metrics.cost_usd, 6), 'top_score': round(top_score, 4)}})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/evaluate")
@limiter.limit("3/minute")
def evaluate(request: Request, req: EvalRequest):
    """Run evaluation against a golden dataset."""
    try:
        return pipeline.evaluate(req.golden_path, use_judge=req.use_judge)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/metrics")
def get_metrics():
    """Return aggregated metrics from recent queries."""
    return metrics.summary()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection_count": pipeline.store.count(),
    }
