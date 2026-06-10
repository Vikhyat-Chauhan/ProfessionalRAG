"""FastAPI server — production-ready REST API for the RAG pipeline."""

import json
import logging
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from pipeline import RAGPipeline
from monitoring import analytics
from monitoring.metrics import metrics
from monitoring.visits import create_table_if_needed, write_event, read_events, visitor_hash
from config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if not settings.api_key:
        raise RuntimeError(
            "ProfessionalRAG_KEY is not set — refusing to start with auth disabled."
        )
    create_table_if_needed()
    pipeline = RAGPipeline()
    yield


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="ProfessionalRAG", version="1.0.0", lifespan=lifespan)
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


@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Reject requests without a valid API key (skip health + dashboard shell)."""
    # /dashboard serves only static HTML/JS — it carries no data, and the
    # browser-side fetch to /visits still requires the key the user enters.
    if request.url.path in ("/health", "/dashboard"):
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
    source: str | list[str] | None = None
    pdf_path: str | list[str] | None = None  # deprecated alias for source
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


class TrackRequest(BaseModel):
    event: str = "pageview"
    page: str = "/"
    referrer: str = "direct"
    source: Optional[str] = None
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    ref: Optional[str] = None
    # Device info (pageview)
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None
    viewport_width: Optional[int] = None
    viewport_height: Optional[int] = None
    device_type: Optional[str] = None
    language: Optional[str] = None
    theme: Optional[str] = None
    # Tab click
    tab: Optional[str] = None
    # Outbound click
    url: Optional[str] = None
    hostname: Optional[str] = None
    # Chat message
    question: Optional[str] = None
    # Time on site
    seconds: Optional[int] = None


# ── Endpoints ──────────────────────────────────────────────────────────
@app.post("/ingest", response_model=IngestResponse)
@limiter.limit("5/minute")
def ingest(request: Request, req: IngestRequest):
    """Ingest one or more sources (PDF, DOCX, PPTX, CSV, JSON, text, image, URL, repo)."""
    raw = req.source or req.pdf_path
    if not raw:
        raise HTTPException(status_code=422, detail="Provide 'source' (or 'pdf_path').")
    sources = raw if isinstance(raw, list) else [raw]
    total = 0
    try:
        for src in sources:
            total += pipeline.ingest(src, force=req.force)
        return IngestResponse(chunks=total, message=f"Ingested {total} chunks from {len(sources)} source(s)")
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


# ── Visit tracking ───────────────────────────────────────────────────
@app.post("/track", status_code=204)
@limiter.limit("30/minute")
async def track_visit(request: Request, req: TrackRequest):
    """Record a visit event with source attribution and device info."""
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")
    doc: dict = {
        "event": req.event,
        "timestamp": datetime.now(timezone.utc),
        "ip": ip,
        # Privacy-friendly, daily-rotating visitor id (no cookies, non-reversible).
        "visitor_id": visitor_hash(ip, request.headers.get("user-agent")),
    }

    # Strip None values — only store fields that are set
    fields = req.model_dump(exclude={"event"}, exclude_none=True)
    doc.update(fields)

    # Derive source for pageview events
    if req.event == "pageview":
        doc["source"] = req.utm_source or req.ref or req.source or "direct"

    write_event(doc)


@app.get("/visits")
@limiter.limit("10/minute")
def get_visits(
    request: Request,
    days: int = Query(default=30, ge=1, le=365),
    source: Optional[str] = Query(default=None),
    start: Optional[date] = Query(default=None, description="Range start (YYYY-MM-DD)"),
    end: Optional[date] = Query(default=None, description="Range end (YYYY-MM-DD), inclusive"),
):
    """Return visit analytics with event breakdowns.

    Use either the relative `days` window (default) or an explicit `start`/`end`
    date range — both bounds are required together and `start` must not be after
    `end`.
    """
    if (start is None) != (end is None):
        raise HTTPException(status_code=400, detail="Provide both 'start' and 'end', or neither.")
    if start and end:
        if start > end:
            raise HTTPException(status_code=400, detail="'start' must be on or before 'end'.")
        if (end - start).days > 366:
            raise HTTPException(status_code=400, detail="Date range must not exceed 366 days.")
        events = read_events(days=days, source=source, start=start.isoformat(), end=end.isoformat())
        window = {"start": start.isoformat(), "end": end.isoformat()}
    else:
        events = read_events(days=days, source=source)
        window = {"days": days}

    return analytics.aggregate(events, window)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Self-contained analytics dashboard (Chart.js via CDN, no build step).

    Served without a server-side key: the page asks for the API key once, keeps
    it in localStorage, and uses it as the Bearer token for /visits fetches.
    """
    return HTMLResponse(_DASHBOARD_HTML)


# ── Dashboard (static, self-contained) ─────────────────────────────────
_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ProfessionalRAG — Visit Analytics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root { color-scheme: light dark; --bg:#0f1117; --card:#1a1d27; --fg:#e6e8ee; --muted:#8b90a0; --accent:#6c8cff; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:20px 28px; border-bottom:1px solid #262a36; display:flex; flex-wrap:wrap; gap:14px; align-items:center; }
  h1 { font-size:18px; margin:0; font-weight:650; }
  .controls { margin-left:auto; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
  input, select, button { background:var(--card); color:var(--fg); border:1px solid #2c3140; border-radius:8px; padding:7px 10px; font-size:13px; }
  button { cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
  main { padding:24px 28px; max-width:1200px; margin:0 auto; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin-bottom:24px; }
  .card { background:var(--card); border:1px solid #232734; border-radius:12px; padding:16px 18px; }
  .card .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
  .card .value { font-size:26px; font-weight:700; margin-top:6px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:18px; }
  .panel { background:var(--card); border:1px solid #232734; border-radius:12px; padding:16px 18px; }
  .panel h2 { font-size:13px; color:var(--muted); margin:0 0 12px; text-transform:uppercase; letter-spacing:.04em; }
  .wide { grid-column:1 / -1; }
  #status { color:var(--muted); font-size:13px; }
  canvas { max-height:300px; }
</style>
</head>
<body>
<header>
  <h1>📊 Visit Analytics</h1>
  <div class="controls">
    <select id="range">
      <option value="7">Last 7 days</option>
      <option value="30" selected>Last 30 days</option>
      <option value="90">Last 90 days</option>
      <option value="custom">Custom range…</option>
    </select>
    <input type="date" id="start" style="display:none"/>
    <input type="date" id="end" style="display:none"/>
    <select id="source" title="Filter by source / job code">
      <option value="">All sources</option>
    </select>
    <button class="primary" id="reload">Refresh</button>
    <button id="setkey">API key</button>
  </div>
</header>
<main>
  <p id="status">Loading…</p>
  <div class="cards" id="cards"></div>
  <div class="panel wide"><h2>Pageviews &amp; unique visitors over time</h2><canvas id="trend"></canvas></div>
  <div class="grid" style="margin-top:18px">
    <div class="panel"><h2>Top sources</h2><canvas id="sources"></canvas></div>
    <div class="panel"><h2>Top pages</h2><canvas id="pages"></canvas></div>
    <div class="panel"><h2>Top referrers</h2><canvas id="referrers"></canvas></div>
    <div class="panel"><h2>Devices</h2><canvas id="devices"></canvas></div>
  </div>
</main>
<script>
const $ = s => document.querySelector(s);
const charts = {};
function key() {
  let k = localStorage.getItem("rag_api_key");
  if (!k) { k = prompt("Enter the ProfessionalRAG API key:") || ""; if (k) localStorage.setItem("rag_api_key", k); }
  return k;
}
$("#setkey").onclick = () => { localStorage.removeItem("rag_api_key"); key(); load(); };
$("#range").onchange = () => {
  const custom = $("#range").value === "custom";
  $("#start").style.display = $("#end").style.display = custom ? "" : "none";
};
$("#reload").onclick = load;
$("#source").onchange = load;

// Rebuild the source dropdown from an unfiltered response, preserving selection.
// Only repopulate when viewing "All" so a filtered view doesn't shrink the list.
function populateSources(d) {
  if ($("#source").value !== "") return;
  const sel = $("#source");
  const current = sel.value;
  const sources = Object.keys(d.by_source || {});
  sel.innerHTML = '<option value="">All sources</option>' +
    sources.map(s => `<option value="${s}">${s}</option>`).join("");
  sel.value = current;
}

function bar(id, obj, color) {
  const labels = Object.keys(obj).slice(0, 8), data = labels.map(l => obj[l]);
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($("#"+id), {
    type: "bar",
    data: { labels, datasets: [{ data, backgroundColor: color }] },
    options: { indexAxis: "y", plugins: { legend: { display:false } }, scales: { x: { ticks:{ precision:0 } } } }
  });
}

function render(d) {
  const cards = [
    ["Pageviews", d.pageviews],
    ["Unique visitors", d.unique_visitors],
    ["Bounce rate", d.bounce_rate + "%"],
    ["Avg time", d.time_on_site.avg + "s"],
    ["p95 time", d.time_on_site.p95 + "s"],
    ["Resume downloads", d.resume_downloads],
    ["Chat messages", d.chat_messages],
  ];
  $("#cards").innerHTML = cards.map(([l,v]) =>
    `<div class="card"><div class="label">${l}</div><div class="value">${v}</div></div>`).join("");

  const days = Object.keys(d.by_day);
  if (charts.trend) charts.trend.destroy();
  charts.trend = new Chart($("#trend"), {
    type: "line",
    data: { labels: days, datasets: [
      { label:"Pageviews", data: days.map(x=>d.by_day[x].pageviews), borderColor:"#6c8cff", backgroundColor:"rgba(108,140,255,.15)", fill:true, tension:.3 },
      { label:"Unique visitors", data: days.map(x=>d.by_day[x].unique_visitors), borderColor:"#39d98a", tension:.3 },
    ]},
    options: { plugins:{ legend:{ labels:{ color:"#aab" } } }, scales:{ y:{ ticks:{ precision:0 } } } }
  });
  bar("sources", d.by_source, "#6c8cff");
  bar("pages", d.by_page, "#39d98a");
  bar("referrers", d.by_referrer, "#ffb03a");
  bar("devices", d.by_device, "#c46cff");
}

async function load() {
  const k = key();
  if (!k) { $("#status").textContent = "No API key set."; return; }
  let qs;
  if ($("#range").value === "custom" && $("#start").value && $("#end").value) {
    qs = `start=${$("#start").value}&end=${$("#end").value}`;
  } else {
    qs = `days=${$("#range").value === "custom" ? 30 : $("#range").value}`;
  }
  const src = $("#source").value;
  if (src) qs += `&source=${encodeURIComponent(src)}`;
  $("#status").textContent = "Loading…";
  try {
    const r = await fetch(`/visits?${qs}`, { headers: { Authorization: `Bearer ${k}` } });
    if (r.status === 401) { localStorage.removeItem("rag_api_key"); $("#status").textContent = "Invalid API key — click “API key” to re-enter."; return; }
    if (!r.ok) { $("#status").textContent = `Error ${r.status}`; return; }
    const d = await r.json();
    const win = d.start ? `${d.start} → ${d.end}` : `Last ${d.days} days`;
    $("#status").textContent = src ? `${win} · source: ${src}` : win;
    populateSources(d);
    render(d);
  } catch (e) { $("#status").textContent = "Request failed: " + e; }
}
load();
</script>
</body>
</html>"""
