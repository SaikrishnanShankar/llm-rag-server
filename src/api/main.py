"""
FastAPI application entry point.

Lifecycle:
  startup  → ensure pgvector schema, start Prometheus metrics server
  shutdown → (cleanup if needed)
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import settings
from src.metrics.prometheus import start_metrics_server
from src.retrieval.vectorstore import ensure_schema

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Production LLM RAG Server",
    description=(
        "RAG system with pgvector retrieval, LangGraph agent, vLLM inference, "
        "RAGAS evals, MLflow tracking, and Prometheus metrics."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    logger.info("Starting RAG API server...")
    # Initialise pgvector schema (idempotent)
    try:
        ensure_schema()
        logger.info("pgvector schema ready")
    except Exception as e:
        logger.warning("Schema init skipped (DB not available yet?): %s", e)

    # Start Prometheus metrics server on a separate port
    start_metrics_server(port=settings.prometheus_port)


@app.get("/")
async def root():
    return {
        "service": "Production LLM RAG Server",
        "docs": "/docs",
        "health": "/api/v1/health",
        "metrics": f"http://localhost:{settings.prometheus_port}/metrics",
        "mlflow": settings.mlflow_tracking_uri,
    }
