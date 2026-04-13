"""
MLflow experiment tracking for the RAG system.

Logs per-request metadata so we can compare chunking strategies via
the MLflow UI at http://localhost:5000.

Design decision: logging is fire-and-forget (background task in FastAPI)
so it never adds latency to the response path. Each request is one MLflow
run nested under the experiment; strategy comparison uses MLflow's built-in
comparison view across runs.
"""
from __future__ import annotations

import logging
from typing import Optional

import mlflow

from src.config import settings

logger = logging.getLogger(__name__)

_experiment_id: Optional[str] = None


def _get_experiment_id() -> str:
    global _experiment_id
    if _experiment_id is not None:
        return _experiment_id

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    if experiment is None:
        _experiment_id = mlflow.create_experiment(settings.mlflow_experiment_name)
        logger.info("Created MLflow experiment: %s (id=%s)", settings.mlflow_experiment_name, _experiment_id)
    else:
        _experiment_id = experiment.experiment_id
    return _experiment_id


def log_request(
    query: str,
    chunk_strategy: str,
    top_k: int,
    latency_seconds: float,
    retrieval_latency_seconds: float,
    tokens_used: int,
    tokens_per_second: float,
    model: str,
    needs_retrieval: bool,
    chunks_retrieved: int,
    run_name: Optional[str] = None,
) -> None:
    """
    Log a single RAG request as one MLflow run.

    Call this as a background task — it is synchronous and may take
    ~50ms to write to the tracking server.
    """
    try:
        experiment_id = _get_experiment_id()
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            # Parameters (hyperparameters / config)
            mlflow.log_params({
                "chunk_strategy": chunk_strategy,
                "top_k": top_k,
                "model": model,
                "needs_retrieval": needs_retrieval,
            })
            # Metrics (numeric, per-request)
            mlflow.log_metrics({
                "latency_seconds": latency_seconds,
                "retrieval_latency_seconds": retrieval_latency_seconds,
                "tokens_used": tokens_used,
                "tokens_per_second": tokens_per_second,
                "chunks_retrieved": chunks_retrieved,
            })
            # Tag with query snippet for filtering
            mlflow.set_tags({
                "query_preview": query[:100],
            })
    except Exception as e:
        # Never let MLflow errors affect the request path
        logger.warning("MLflow logging failed: %s", e)


def log_eval_run(
    chunk_strategy: str,
    faithfulness: float,
    context_relevance: float,
    answer_relevance: float,
    num_questions: int,
    run_name: Optional[str] = None,
) -> str:
    """
    Log a complete RAGAS eval run. Returns the MLflow run_id.

    These runs are what you compare in the MLflow UI to pick the best
    chunking strategy.
    """
    try:
        experiment_id = _get_experiment_id()
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name or f"eval_{chunk_strategy}",
        ) as run:
            mlflow.log_params({
                "chunk_strategy": chunk_strategy,
                "num_questions": num_questions,
                "eval_type": "ragas",
            })
            mlflow.log_metrics({
                "faithfulness": faithfulness,
                "context_relevance": context_relevance,
                "answer_relevance": answer_relevance,
                "composite_score": (faithfulness + context_relevance + answer_relevance) / 3,
            })
            mlflow.set_tags({"run_type": "eval"})
            return run.info.run_id
    except Exception as e:
        logger.warning("MLflow eval logging failed: %s", e)
        return ""


def log_eval_artifact(local_path: str, artifact_name: str = "") -> None:
    """Attach a file (JSON/markdown) to the active MLflow run."""
    try:
        mlflow.log_artifact(local_path, artifact_name)
    except Exception as e:
        logger.warning("MLflow artifact logging failed: %s", e)
