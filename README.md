# ProfessionalRAG

**Production-grade Retrieval-Augmented Generation pipeline with built-in evaluation, cost tracking, and observability.**

A modular RAG system that ingests PDFs, performs semantic retrieval with cross-encoder reranking, generates grounded answers via Claude, and evaluates quality through golden datasets and LLM-as-judge scoring — all with per-query latency and cost instrumentation.

---

## Architecture

```
                          +-----------+
                          |  CLI / API |
                          +-----+-----+
                                |
                     +----------v----------+
                     |    RAG Pipeline      |
                     |    (Orchestrator)    |
                     +----------+----------+
                                |
          +---------------------+---------------------+
          |                     |                     |
+---------v--------+  +---------v--------+  +---------v--------+
|    Ingestion     |  |    Retrieval     |  |   Generation     |
|  PDF Reader      |  |  BGE Embedder   |  |  Claude LLM      |
|  Recursive       |  |  ChromaDB Store |  |  Token + Cost    |
|  Chunker         |  |  Cross-Encoder  |  |  Tracking        |
|  SHA-256 Cache   |  |  Reranker       |  |                  |
+------------------+  +------------------+  +------------------+
                                |
          +---------------------+---------------------+
          |                                           |
+---------v--------+                       +---------v--------+
|   Evaluation     |                       |   Monitoring     |
|  Hit@K / MRR     |                       |  Latency (ms)    |
|  LLM-as-Judge    |                       |  Token Count     |
|  Golden Datasets |                       |  Cost (USD)      |
+------------------+                       |  JSONL Streaming |
                                           +------------------+
```

### Component Breakdown

| Layer | Component | What It Does |
|---|---|---|
| **Ingestion** | `reader.py` | Extracts page-level text from PDFs with SHA-256 fingerprinting for change detection |
| **Ingestion** | `chunker.py` | Zero-dependency recursive character splitter (500 chars, 50 overlap) |
| **Retrieval** | `embedder.py` | BGE embeddings with proper `query:`/`passage:` prefixes |
| **Retrieval** | `store.py` | ChromaDB persistent vector store with fingerprint-based cache invalidation |
| **Retrieval** | `reranker.py` | Cross-encoder reranking (MS MARCO MiniLM) for precision |
| **Generation** | `llm.py` | Claude integration with system-prompt grounding and token tracking |
| **Evaluation** | `golden.py` | Hit@K and MRR computation against golden QA datasets |
| **Evaluation** | `judge.py` | LLM-as-judge scoring (1-5) on Faithfulness, Completeness, Conciseness |
| **Monitoring** | `metrics.py` | Per-query latency breakdown, token counting, cost calculation, JSONL logging |

---

## Key Engineering Decisions

| Decision | Rationale |
|---|---|
| **Two-stage retrieval** (embedding + cross-encoder rerank) | Retrieve 50 candidates fast, then precision-rank the top-k with a cross-encoder — balances recall and relevance |
| **SHA-256 fingerprint caching** | Skip re-ingestion when the source document hasn't changed; avoids redundant embedding computation |
| **Zero-dependency chunker** | Recursive character splitting without LangChain — full control over separator hierarchy and overlap logic |
| **JSONL metrics streaming** | Append-only log for every query; enables post-hoc analysis without impacting query latency |
| **Pydantic Settings** | Type-safe configuration with env var support and validation — no stringly-typed configs |
| **Lazy model loading** | Embedder and reranker models initialize on first use, not at import time |
| **Context manager latency tracking** | `with metrics.track_latency("retrieval"):` — clean, composable instrumentation |

---

## Metrics & Observability

Every query produces a structured metrics record:

```json
{
  "query": "What is the improvement on reach time?",
  "retrieval_ms": 103.9,
  "rerank_ms": 2629.7,
  "llm_ms": 1842.5,
  "total_ms": 6373.2,
  "input_tokens": 530,
  "output_tokens": 137,
  "cost_usd": 0.003645,
  "top_score": -2.4085,
  "chunks": 5
}
```

**What's tracked:**
- **Latency breakdown** — retrieval, reranking, and LLM generation timed independently
- **Token economics** — input/output tokens with configurable per-million-token cost rates
- **Relevance signal** — top cross-encoder score as a retrieval quality proxy
- **Persistent log** — `metrics_log.jsonl` for trend analysis and regression detection

### Evaluation Metrics

| Metric | What It Measures |
|---|---|
| **Hit@K** | Did the expected page appear in the top-k retrieved chunks? |
| **MRR** | Mean Reciprocal Rank — how high did the first relevant result rank? |
| **Judge Score (1-5)** | LLM-rated answer quality across faithfulness, completeness, and conciseness |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.13 |
| **LLM** | Anthropic Claude (claude-sonnet-4-6) |
| **Embeddings** | BAAI/bge-base-en-v1.5 (768-dim) |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Vector Store** | ChromaDB (persistent SQLite backend) |
| **API Framework** | FastAPI + Uvicorn |
| **CLI** | Click |
| **Config** | Pydantic Settings + .env |

---

## Project Structure

```
ProfessionalRAG/
├── api/
│   └── server.py           # FastAPI endpoints (/health, /ingest, /query, /evaluate, /metrics)
├── ingestion/
│   ├── reader.py            # PDF extraction + SHA-256 fingerprinting
│   └── chunker.py           # Recursive text splitter (zero dependencies)
├── retrieval/
│   ├── embedder.py          # BGE embedding wrapper
│   ├── store.py             # ChromaDB vector store with batch ingestion
│   └── reranker.py          # Cross-encoder reranker
├── generation/
│   └── llm.py               # Claude client with cost tracking
├── evaluation/
│   ├── golden.py            # Golden dataset evaluation (Hit@K, MRR)
│   └── judge.py             # LLM-as-judge scoring
├── monitoring/
│   └── metrics.py           # Latency, tokens, cost — JSONL streaming
├── pipeline.py              # RAG orchestrator
├── config.py                # Pydantic settings (env-driven)
├── cli.py                   # Click CLI
├── golden_example.json      # Example evaluation dataset
└── metrics_log.jsonl        # Query metrics history
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | `GET` | Health check + collection count |
| `/ingest` | `POST` | Ingest a PDF (with fingerprint-based skip) |
| `/query` | `POST` | Full retrieve-rerank-generate pipeline |
| `/evaluate` | `POST` | Run evaluation against a golden dataset |
| `/metrics` | `GET` | Aggregated performance statistics |

**Query response shape:**

```json
{
  "answer": "The reach time improved by 15% compared to...",
  "sources": [
    { "page": 12, "score": 0.8742, "text": "..." }
  ],
  "metrics": {
    "total_ms": 4200.3,
    "llm_ms": 1842.5,
    "cost_usd": 0.003645,
    "top_score": 0.8742
  }
}
```

---

## Quickstart

```bash
# Clone & install
git clone https://github.com/vikhyatchauhan/ProfessionalRAG.git
cd ProfessionalRAG
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# Ingest a document
python cli.py ingest /path/to/document.pdf

# Query
python cli.py query "What methodology was used in this research?"

# Start the API server
python cli.py serve

# Run evaluation
python cli.py evaluate golden_example.json

# View aggregated metrics
python cli.py stats --last 50
```

---

## Configuration

All settings are environment-driven via Pydantic:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude API key (required) |
| `LLM_MODEL` | `claude-sonnet-4-6` | LLM model |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `CANDIDATE_COUNT` | `50` | Candidates before reranking |
| `TOP_K` | `5` | Final results returned |
| `COST_PER_M_INPUT_TOKENS` | `3.0` | USD per million input tokens |
| `COST_PER_M_OUTPUT_TOKENS` | `15.0` | USD per million output tokens |

---

## License

MIT
