"""Production monitoring — latency, token usage, cost tracking.

Logs structured JSON to stdout (captured by Cloud Logging on Cloud Run)
and keeps an in-memory buffer for the /metrics summary endpoint.
"""

import json
import logging
import os
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

from config import settings

log = logging.getLogger(__name__)

# When set (e.g. in Lambda), per-query metrics are also emitted as CloudWatch
# Embedded Metric Format so latency/cost aggregate across ephemeral instances —
# the in-memory buffer alone resets on every cold start.
_EMF_NAMESPACE = os.environ.get("EMF_NAMESPACE")


@dataclass
class QueryMetrics:
    query: str = ""
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    top_score: float = 0.0
    chunks_retrieved: int = 0


class MetricsCollector:
    """Collects per-query metrics, logs to stdout, keeps in-memory buffer."""

    def __init__(self, buffer_size: int = 200):
        self._current = QueryMetrics()
        self._timers: dict[str, float] = {}
        self._history: deque[dict] = deque(maxlen=buffer_size)

    def start_query(self, query: str) -> None:
        self._current = QueryMetrics(query=query)
        self._timers["total"] = time.perf_counter()

    @contextmanager
    def track_latency(self, stage: str):
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000
        attr = f"{stage}_latency_ms"
        if hasattr(self._current, attr):
            setattr(self._current, attr, elapsed_ms)
        log.debug("%s took %.1f ms", stage, elapsed_ms)

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self._current.input_tokens = input_tokens
        self._current.output_tokens = output_tokens
        self._current.cost_usd = (
            input_tokens * settings.cost_per_m_input_tokens
            + output_tokens * settings.cost_per_m_output_tokens
        ) / 1_000_000

    def record_retrieval(self, top_score: float, chunks_retrieved: int) -> None:
        self._current.top_score = top_score
        self._current.chunks_retrieved = chunks_retrieved

    def finish_query(self) -> QueryMetrics:
        if "total" in self._timers:
            self._current.total_latency_ms = (
                (time.perf_counter() - self._timers["total"]) * 1000
            )
        self._flush()
        result = self._current
        self._current = QueryMetrics()
        self._timers.clear()
        return result

    def _flush(self) -> None:
        entry = {
            "query": self._current.query,
            "retrieval_ms": round(self._current.retrieval_latency_ms, 1),
            "rerank_ms": round(self._current.rerank_latency_ms, 1),
            "llm_ms": round(self._current.llm_latency_ms, 1),
            "total_ms": round(self._current.total_latency_ms, 1),
            "input_tokens": self._current.input_tokens,
            "output_tokens": self._current.output_tokens,
            "cost_usd": round(self._current.cost_usd, 6),
            "top_score": round(self._current.top_score, 4),
            "chunks": self._current.chunks_retrieved,
        }
        # Structured JSON log → Cloud Logging / CloudWatch Logs picks this up
        log.info(json.dumps(entry))
        self._history.append(entry)
        self._emit_emf()

    def _emit_emf(self) -> None:
        """Emit CloudWatch EMF so metrics persist across ephemeral instances."""
        if not _EMF_NAMESPACE:
            return
        emf = {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": _EMF_NAMESPACE,
                    "Dimensions": [[]],
                    "Metrics": [
                        {"Name": "TotalLatencyMs", "Unit": "Milliseconds"},
                        {"Name": "LlmLatencyMs", "Unit": "Milliseconds"},
                        {"Name": "CostUsd", "Unit": "None"},
                        {"Name": "TopScore", "Unit": "None"},
                    ],
                }],
            },
            "TotalLatencyMs": round(self._current.total_latency_ms, 1),
            "LlmLatencyMs": round(self._current.llm_latency_ms, 1),
            "CostUsd": round(self._current.cost_usd, 6),
            "TopScore": round(self._current.top_score, 4),
        }
        # Bare print: EMF must be the whole log line, not wrapped by a formatter.
        print(json.dumps(emf))

    def summary(self, last_n: int = 50) -> dict:
        """Aggregate stats from recent queries (in-memory buffer)."""
        entries = list(self._history)[-last_n:]
        if not entries:
            return {}
        n = len(entries)
        return {
            "queries": n,
            "avg_total_ms": round(sum(e["total_ms"] for e in entries) / n, 1),
            "avg_llm_ms": round(sum(e["llm_ms"] for e in entries) / n, 1),
            "total_cost_usd": round(sum(e["cost_usd"] for e in entries), 4),
            "avg_top_score": round(sum(e["top_score"] for e in entries) / n, 4),
        }


# Singleton
metrics = MetricsCollector()
