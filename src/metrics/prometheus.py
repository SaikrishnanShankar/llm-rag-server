"""
Prometheus instrumentation for the RAG API.

All metrics are registered at module import time. The metrics server
runs on a separate port (default 8001) so Prometheus can scrape it
independently from the main API port.

Design decision: histogram buckets are tuned for LLM workloads where
p50 is ~0.5s (mock) or ~2s (real GPU) and p99 can reach 30s for long
generations. Standard default buckets miss this range entirely.
"""
from __future__ import annotations

import logging

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

logger = logging.getLogger(__name__)

# ─── Latency buckets tuned for LLM serving ────────────────────────────────────
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)

# ─── Metric definitions ───────────────────────────────────────────────────────

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "End-to-end request latency (query → response)",
    labelnames=["chunk_strategy"],
    buckets=_LATENCY_BUCKETS,
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "pgvector similarity search latency",
    labelnames=["chunk_strategy"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

TOKENS_PER_SECOND = Gauge(
    "rag_tokens_per_second",
    "LLM token generation throughput (rolling, last request)",
    labelnames=["model"],
)

ACTIVE_REQUESTS = Gauge(
    "rag_active_requests",
    "Number of requests currently being processed",
)

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total number of requests processed",
    labelnames=["chunk_strategy", "status"],
)

EVAL_FAITHFULNESS = Gauge(
    "rag_eval_faithfulness_score",
    "RAGAS faithfulness score (0-1) for the most recent eval run",
    labelnames=["chunk_strategy"],
)

EVAL_CONTEXT_RELEVANCE = Gauge(
    "rag_eval_context_relevance_score",
    "RAGAS context relevance score (0-1)",
    labelnames=["chunk_strategy"],
)

EVAL_ANSWER_RELEVANCE = Gauge(
    "rag_eval_answer_relevance_score",
    "RAGAS answer relevance score (0-1)",
    labelnames=["chunk_strategy"],
)

INGESTION_CHUNKS = Counter(
    "rag_ingestion_chunks_total",
    "Total chunks inserted into pgvector",
    labelnames=["chunk_strategy"],
)

# ─── Server ───────────────────────────────────────────────────────────────────

_metrics_server_started = False


def start_metrics_server(port: int = 8001) -> None:
    global _metrics_server_started
    if _metrics_server_started:
        return
    start_http_server(port)
    _metrics_server_started = True
    logger.info("Prometheus metrics server started on port %d", port)


# ─── Helper: record eval scores ──────────────────────────────────────────────

def record_eval_scores(chunk_strategy: str, scores: dict) -> None:
    """Push latest RAGAS eval scores into Prometheus gauges."""
    if "faithfulness" in scores:
        EVAL_FAITHFULNESS.labels(chunk_strategy=chunk_strategy).set(scores["faithfulness"])
    if "context_relevance" in scores:
        EVAL_CONTEXT_RELEVANCE.labels(chunk_strategy=chunk_strategy).set(scores["context_relevance"])
    if "answer_relevance" in scores:
        EVAL_ANSWER_RELEVANCE.labels(chunk_strategy=chunk_strategy).set(scores["answer_relevance"])
