# ProfessionalRAG

**Production-grade Retrieval-Augmented Generation system with two-stage retrieval, LLM-as-judge evaluation, and per-query cost observability.**

Built from scratch — no LangChain, no LlamaIndex. Every component (chunker, embedder, reranker, vector store, evaluator, metrics collector) is hand-written with clear separation of concerns.

---

## Why This Project Stands Out

| What | How |
|---|---|
| **Two-stage retrieval** | Embedding recall (50 candidates) + cross-encoder reranking (top-k precision) — the same pattern used in production search systems at scale |
| **Zero-dependency chunker** | Recursive character splitter with configurable separator hierarchy and overlap — no framework lock-in |
| **SHA-256 cache invalidation** | Fingerprint-based skip logic prevents redundant embedding computation on unchanged documents |
| **LLM-as-judge evaluation** | Automated answer quality scoring (1-5) across Faithfulness, Completeness, and Conciseness — not just vibes |
| **Per-query cost instrumentation** | Every query logs latency breakdown, token counts, and USD cost to a JSONL stream — production observability from day one |
| **Lazy model loading** | Embedder and reranker initialize on first use, not at import — fast cold starts, testable without GPU |
| **Clean architecture** | Ingestion / Retrieval / Generation / Evaluation / Monitoring — each layer is independently testable and swappable |

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
|  PDF + GitHub    |  |  BGE Embedder   |  |  Claude LLM      |
|  Recursive       |  |  Pinecone Store |  |  Token + Cost    |
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

---

## How It Works

### 1. Ingestion
```
PDF / GitHub Repo → Page Extraction → SHA-256 Fingerprint → Recursive Chunking → BGE Embedding → ChromaDB
```
- Reads PDFs page-by-page or clones GitHub repos and reads source files
- SHA-256 fingerprint check skips re-ingestion if the source hasn't changed
- Recursive character splitter (500 chars, 50 overlap) with separator priority: `\n\n` → `\n` → `. ` → ` ` → `""`
- Chunks are tagged with a source identifier for per-source cache invalidation

### 2. Retrieval (Two-Stage)
```
Query → BGE Embedding → Pinecone ANN Search (50 candidates) → Cross-Encoder Rerank → Top-K Results
```
- **Stage 1:** Fast approximate nearest neighbor search retrieves 50 candidates
- **Stage 2:** Cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores each query-chunk pair for precision reranking
- This retrieve-then-rerank pattern balances recall and relevance — the same approach used in production search at Google, Bing, and Cohere

### 3. Generation
```
Top-K Chunks + Query → Grounded System Prompt → Claude → Answer with Source Citations
```
- System prompt enforces grounded answers — the model must refuse if evidence is insufficient
- Token usage (input/output) tracked per query with configurable cost rates
- Lazy client initialization — no API calls until the first query

### 4. Evaluation
```
Golden Dataset (Q/A/Pages) → Retrieval Metrics (Hit@K, MRR) → LLM-as-Judge (1-5 Score)
```
- **Hit@K:** Did the expected page appear in top-k retrieved chunks?
- **MRR:** Mean Reciprocal Rank — how high did the first relevant result rank?
- **LLM-as-Judge:** Claude rates each answer on Faithfulness, Completeness, and Conciseness (1-5 scale)

---

## Metrics & Observability

Every query produces a structured metrics record appended to `metrics_log.jsonl`:

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
- **Latency breakdown** — retrieval, reranking, and LLM generation timed independently via context manager instrumentation
- **Token economics** — input/output tokens with configurable per-million-token cost rates
- **Relevance signal** — top cross-encoder score as a retrieval quality proxy
- **Persistent log** — append-only JSONL for trend analysis and regression detection

---

## Key Engineering Decisions

| Decision | Rationale |
|---|---|
| **No LangChain / LlamaIndex** | Full control over every component — easier to debug, profile, and optimize |
| **Two-stage retrieval** | Embedding search is fast but imprecise; cross-encoder reranking adds precision without the cost of reranking the entire corpus |
| **SHA-256 fingerprint caching** | Avoids redundant embedding computation — critical when ingestion involves expensive model inference |
| **JSONL metrics streaming** | Append-only, zero-contention logging — enables post-hoc analysis without impacting query latency |
| **Pydantic Settings** | Type-safe, env-driven configuration with validation — no stringly-typed configs or missing key surprises |
| **Context manager latency tracking** | `with metrics.track_latency("retrieval"):` — composable instrumentation that's impossible to forget to close |
| **Lazy model loading** | Models load on first use, not at import — keeps tests fast and startup cheap |
| **Source-tagged vector store** | Each ingested source gets an MD5-based tag, enabling per-source cache invalidation without wiping the entire collection |

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM** | Claude (claude-sonnet-4-6) | Strong instruction-following, structured output, cost-effective |
| **Embeddings** | BAAI/bge-base-en-v1.5 (768-dim) | Top-tier open embedding model with query/passage prefix support |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Fast cross-encoder trained on MS MARCO — strong relevance signal |
| **Vector Store** | Pinecone (serverless) | Managed, scalable, no infra to maintain — persistent across deploys |
| **API** | FastAPI + Uvicorn | Async-ready, auto-generated OpenAPI docs, Pydantic integration |
| **CLI** | Click | Clean subcommand interface with built-in help |
| **Config** | Pydantic Settings + .env | Type-safe env var parsing with defaults and validation |
| **Container** | Docker + Docker Compose | One-command deployment with persistent volume for ChromaDB |

---

## Project Structure

```
ProfessionalRAG/
├── api/
│   └── server.py              # FastAPI REST API (5 endpoints)
├── ingestion/
│   ├── reader.py               # PDF extraction + GitHub repo cloning + SHA-256 fingerprinting
│   └── chunker.py              # Zero-dependency recursive character splitter
├── retrieval/
│   ├── embedder.py             # BGE embedding with query:/passage: prefixes
│   ├── store.py                # Pinecone vector store with per-source fingerprint cache invalidation
│   └── reranker.py             # Cross-encoder reranker (MS MARCO)
├── generation/
│   └── llm.py                  # Claude client with grounded system prompt + token tracking
├── evaluation/
│   ├── golden.py               # Golden dataset eval (Hit@K, MRR)
│   └── judge.py                # LLM-as-judge scoring (Faithfulness/Completeness/Conciseness)
├── monitoring/
│   └── metrics.py              # Per-query latency, tokens, cost — JSONL streaming
├── pipeline.py                 # Orchestrator: ingest → retrieve → rerank → generate → evaluate
├── config.py                   # Pydantic Settings (env-driven, type-safe)
├── cli.py                      # Click CLI (ingest, query, evaluate, stats, serve)
├── Dockerfile                  # Production container (python:3.12-slim)
├── docker-compose.yml          # One-command local deploy
└── golden_example.json         # Example evaluation dataset
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | `GET` | Health check + document count |
| `/ingest` | `POST` | Ingest PDF(s) with fingerprint-based dedup |
| `/query` | `POST` | Full retrieve → rerank → generate pipeline |
| `/evaluate` | `POST` | Run eval against golden dataset |
| `/metrics` | `GET` | Aggregated performance statistics |

**Example query response:**

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
cp .env.example .env   # Add your ANTHROPIC_API_KEY and PINECONE_API_KEY

# Ingest a document
python cli.py ingest /path/to/document.pdf

# Query
python cli.py query "What methodology was used in this research?"

# Start the API server
python cli.py serve

# Run evaluation against a golden dataset
python cli.py evaluate golden_example.json

# View aggregated metrics
python cli.py stats --last 50
```

### Docker

```bash
docker compose up --build
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

---

## Configuration

All settings are environment-driven via Pydantic Settings:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required |
| `PINECONE_API_KEY` | — | Required |
| `PINECONE_INDEX` | `professional-rag` | Pinecone index name |
| `PINECONE_CLOUD` | `aws` | Cloud provider |
| `PINECONE_REGION` | `us-east-1` | Region |
| `LLM_MODEL` | `claude-sonnet-4-6` | LLM model |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `CANDIDATE_COUNT` | `50` | Embedding candidates before reranking |
| `TOP_K` | `5` | Final results after reranking |
| `COST_PER_M_INPUT_TOKENS` | `3.0` | USD per million input tokens |
| `COST_PER_M_OUTPUT_TOKENS` | `15.0` | USD per million output tokens |

---

## License

MIT
