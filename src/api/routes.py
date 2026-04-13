"""
FastAPI route handlers for /query, /ingest, /eval, /health.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import AsyncIterator, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agent.graph import run_agent
from src.config import settings
from src.metrics.prometheus import (
    ACTIVE_REQUESTS,
    INGESTION_CHUNKS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    RETRIEVAL_LATENCY,
    TOKENS_PER_SECOND,
)
from src.retrieval.chunking import chunk_text
from src.retrieval.embeddings import embed, embed_for_chunking
from src.retrieval.vectorstore import ensure_schema, insert_chunks
from src.tracking.mlflow_logger import log_request

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    chunk_strategy: Optional[str] = Field(
        default=None,
        description="Filter retrieval to a specific strategy: fixed | sentence | semantic"
    )
    stream: bool = Field(default=False, description="Enable streaming response")


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Document text to ingest")
    doc_id: str = Field(..., min_length=1, description="Unique document identifier")
    chunk_strategy: str = Field(
        default="sentence",
        description="Chunking strategy: fixed | sentence | semantic"
    )
    metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[dict]
    metadata: dict


class IngestResponse(BaseModel):
    doc_id: str
    chunks_inserted: int
    chunk_strategy: str


# ─── /health ──────────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    return {"status": "healthy", "vllm_url": settings.vllm_base_url}


# ─── /query ───────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Main RAG query endpoint.

    Runs the LangGraph agent pipeline: classify → retrieve → rerank → generate → format.
    Logs metrics to Prometheus and MLflow asynchronously.
    """
    ACTIVE_REQUESTS.inc()
    t0 = time.perf_counter()
    strategy = request.chunk_strategy or settings.default_chunk_strategy

    try:
        result = await run_agent(
            query=request.query,
            top_k=request.top_k,
            chunk_strategy=request.chunk_strategy,
        )

        total_latency = time.perf_counter() - t0
        usage = result.get("metadata", {}).get("usage", {})
        retrieval_lat = result.get("metadata", {}).get("retrieval_latency_seconds", 0.0)

        # Prometheus
        REQUEST_LATENCY.labels(chunk_strategy=strategy).observe(total_latency)
        RETRIEVAL_LATENCY.labels(chunk_strategy=strategy).observe(retrieval_lat)
        tps = usage.get("tokens_per_second", 0)
        if tps > 0:
            TOKENS_PER_SECOND.labels(model=settings.vllm_model).set(tps)
        REQUEST_COUNT.labels(chunk_strategy=strategy, status="success").inc()

        # MLflow (background — never blocks response)
        background_tasks.add_task(
            log_request,
            query=request.query,
            chunk_strategy=strategy,
            top_k=request.top_k,
            latency_seconds=total_latency,
            retrieval_latency_seconds=retrieval_lat,
            tokens_used=usage.get("total_tokens", 0),
            tokens_per_second=tps,
            model=settings.vllm_model,
            needs_retrieval=result.get("metadata", {}).get("needs_retrieval", False),
            chunks_retrieved=result.get("metadata", {}).get("chunks_retrieved", 0),
        )

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result.get("sources", []),
            metadata=result.get("metadata", {}),
        )

    except Exception as e:
        REQUEST_COUNT.labels(chunk_strategy=strategy, status="error").inc()
        logger.exception("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streaming query endpoint — yields tokens via Server-Sent Events.

    Uses the same LangGraph routing but streams the generation step.
    """
    from src.inference.vllm_client import (
        build_direct_messages,
        build_rag_messages,
        generate_stream,
    )
    from src.retrieval.embeddings import embed_query
    from src.retrieval.vectorstore import similarity_search

    strategy = request.chunk_strategy or settings.default_chunk_strategy

    async def _generate() -> AsyncIterator[str]:
        ACTIVE_REQUESTS.inc()
        try:
            # Minimal routing: always retrieve for stream endpoint
            query_vec = embed_query(request.query)
            chunks = similarity_search(
                query_embedding=query_vec,
                top_k=request.top_k,
                chunk_strategy=request.chunk_strategy,
            )
            messages = build_rag_messages(request.query, chunks)
            async for token in generate_stream(messages):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            ACTIVE_REQUESTS.dec()

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ─── /ingest ──────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest_text(request: IngestRequest):
    """Ingest a document text string directly (no file upload required)."""
    ensure_schema()

    embed_fn = embed_for_chunking if request.chunk_strategy == "semantic" else None
    chunks = chunk_text(request.text, strategy=request.chunk_strategy, embed_fn=embed_fn)

    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks produced from document")

    embeddings = embed(chunks)
    inserted = insert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        doc_id=request.doc_id,
        chunk_strategy=request.chunk_strategy,
        metadata=request.metadata,
    )

    INGESTION_CHUNKS.labels(chunk_strategy=request.chunk_strategy).inc(inserted)
    return IngestResponse(
        doc_id=request.doc_id,
        chunks_inserted=inserted,
        chunk_strategy=request.chunk_strategy,
    )


@router.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    chunk_strategy: str = "sentence",
    doc_id: Optional[str] = None,
):
    """Upload and ingest a .txt or .pdf file."""
    import hashlib
    from src.retrieval.ingest import load_pdf, load_txt

    content = await file.read()
    filename = file.filename or "upload"
    effective_doc_id = doc_id or f"{Path(filename).stem}_{hashlib.sha256(content).hexdigest()[:8]}"

    # Decode based on extension
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        text = content.decode("utf-8", errors="ignore")
    elif ext == ".pdf":
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")

    ensure_schema()
    embed_fn = embed_for_chunking if chunk_strategy == "semantic" else None
    chunks = chunk_text(text, strategy=chunk_strategy, embed_fn=embed_fn)
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks produced")

    embeddings = embed(chunks)
    inserted = insert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        doc_id=effective_doc_id,
        chunk_strategy=chunk_strategy,
        metadata={"filename": filename, "original_name": file.filename},
    )
    INGESTION_CHUNKS.labels(chunk_strategy=chunk_strategy).inc(inserted)
    return {"doc_id": effective_doc_id, "chunks_inserted": inserted, "chunk_strategy": chunk_strategy}


# ─── /eval (trigger) ──────────────────────────────────────────────────────────

@router.post("/eval")
async def trigger_eval(background_tasks: BackgroundTasks, strategies: List[str] = None):
    """
    Trigger a RAGAS evaluation run in the background.

    Results are written to eval/results/ and logged to MLflow.
    """
    from src.evals.run_evals import run_full_eval

    if strategies is None:
        strategies = ["fixed", "sentence", "semantic"]

    background_tasks.add_task(run_full_eval, strategies=strategies)
    return {"status": "eval started", "strategies": strategies, "results_dir": "eval/results/"}
