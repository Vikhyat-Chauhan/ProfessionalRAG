"""FastAPI server — production-ready REST API for the RAG pipeline."""

import logging

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from pipeline import RAGPipeline
from monitoring.metrics import metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

app = FastAPI(title="ProfileRAG", version="1.0.0")
pipeline = RAGPipeline()


# ── Request / Response models ──────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class IngestRequest(BaseModel):
    pdf_path: str
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


# ── Endpoints ──────────────────────────────────────────────────────────


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """Ingest a PDF into the vector store."""
    try:
        count = pipeline.ingest(req.pdf_path, force=req.force)
        return IngestResponse(chunks=count, message=f"Ingested {count} chunks")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Query the RAG pipeline."""
    if pipeline.store.collection.count() <= 1:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested. POST /ingest first.",
        )
    result = pipeline.query(req.question, top_k=req.top_k)
    return result


@app.post("/evaluate")
def evaluate(req: EvalRequest):
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
        "collection_count": pipeline.store.collection.count(),
    }
