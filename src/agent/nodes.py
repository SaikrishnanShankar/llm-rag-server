"""
LangGraph agent nodes — each function takes State and returns a State update.

Node responsibilities:
  1. classify_query   — decide if retrieval is needed
  2. retrieve_chunks  — pgvector similarity search
  3. rerank_chunks    — cross-encoder or score-based reranking
  4. generate_answer  — vLLM inference
  5. format_response  — assemble final response with citations

Design decision: cross-encoder reranking uses sentence-transformers
cross-encoder/ms-marco-MiniLM-L-6-v2 when available, falling back to
simple cosine-score sorting. This keeps the node functional without
requiring the heavy cross-encoder download in local dev.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List

from src.agent.state import AgentState
from src.config import settings
from src.inference.vllm_client import (
    build_direct_messages,
    build_rag_messages,
    generate,
)
from src.retrieval.embeddings import embed_query
from src.retrieval.vectorstore import similarity_search

logger = logging.getLogger(__name__)

# ─── Retrieval keywords — queries matching these almost always need docs ──────
_RETRIEVAL_KEYWORDS = {
    "what", "how", "explain", "describe", "define", "difference",
    "compare", "why", "when", "where", "which", "who", "list",
    "summarize", "summary", "tell me about", "examples of",
}

_DIRECT_PATTERNS = {
    "hello", "hi", "thanks", "thank you", "help", "who are you",
    "what can you do", "calculate", "compute", "translate",
}


# ─── Node 1: classify_query ───────────────────────────────────────────────────

async def classify_query(state: AgentState) -> Dict[str, Any]:
    """
    Heuristic + LLM-backed classifier.

    First tries a fast keyword heuristic. If ambiguous, asks the LLM.
    This avoids an extra LLM call for the majority of clear cases.
    """
    query = state["query"].lower().strip()

    # Fast path: obvious direct queries
    for pat in _DIRECT_PATTERNS:
        if query.startswith(pat):
            logger.debug("classify_query → direct (keyword match: %r)", pat)
            return {"needs_retrieval": False, "routing_reason": f"direct pattern: {pat}"}

    # Fast path: obvious retrieval queries
    first_word = query.split()[0] if query.split() else ""
    if first_word in _RETRIEVAL_KEYWORDS or any(kw in query for kw in {"what is", "how does", "explain", "describe"}):
        logger.debug("classify_query → retrieval (keyword: %r)", first_word)
        return {"needs_retrieval": True, "routing_reason": f"retrieval keyword: {first_word}"}

    # LLM fallback for ambiguous queries
    messages = [
        {
            "role": "system",
            "content": (
                "Classify whether this query requires searching a knowledge base. "
                "Reply with exactly one word: RETRIEVAL or DIRECT."
            ),
        },
        {"role": "user", "content": state["query"]},
    ]
    answer, _ = await generate(messages, temperature=0.0, max_tokens=10)
    needs = "RETRIEVAL" in answer.upper()
    logger.debug("classify_query → %s (LLM decision)", "retrieval" if needs else "direct")
    return {
        "needs_retrieval": needs,
        "routing_reason": f"LLM classified as {'RETRIEVAL' if needs else 'DIRECT'}",
    }


# ─── Node 2: retrieve_chunks ──────────────────────────────────────────────────

async def retrieve_chunks(state: AgentState) -> Dict[str, Any]:
    """Embed the query and run pgvector cosine similarity search."""
    t0 = time.perf_counter()

    query_vec = embed_query(state["query"])
    chunks = similarity_search(
        query_embedding=query_vec,
        top_k=state.get("top_k", settings.default_top_k),
        chunk_strategy=state.get("chunk_strategy") or None,
    )

    retrieval_latency = time.perf_counter() - t0
    logger.debug("retrieve_chunks: %d chunks in %.3fs", len(chunks), retrieval_latency)

    return {
        "retrieved_chunks": chunks,
        "retrieval_latency_seconds": retrieval_latency,
    }


# ─── Node 3: rerank_chunks ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Lazy-load cross-encoder; returns None if not installed."""
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        logger.info("Cross-encoder loaded for reranking")
        return model
    except Exception as e:
        logger.warning("Cross-encoder unavailable (%s); using score-based reranking", e)
        return None


async def rerank_chunks(state: AgentState) -> Dict[str, Any]:
    """
    Rerank retrieved chunks.

    Uses cross-encoder when available; falls back to original cosine
    similarity scores (already sorted by pgvector, so this is a no-op
    that still normalises the chunk list for downstream nodes).
    """
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"reranked_chunks": []}

    query = state["query"]
    cross_encoder = _get_cross_encoder()

    if cross_encoder is not None:
        pairs = [[query, c["content"]] for c in chunks]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(
            zip(chunks, scores), key=lambda x: x[1], reverse=True
        )
        reranked = []
        for chunk, score in ranked:
            chunk = dict(chunk)
            chunk["rerank_score"] = float(score)
            reranked.append(chunk)
    else:
        # Already sorted by cosine similarity from pgvector
        reranked = [dict(c) for c in chunks]

    return {"reranked_chunks": reranked}


# ─── Node 4: generate_answer ──────────────────────────────────────────────────

async def generate_answer(state: AgentState) -> Dict[str, Any]:
    """Call vLLM (or mock) to produce the final answer."""
    if state.get("needs_retrieval", True):
        chunks = state.get("reranked_chunks") or state.get("retrieved_chunks", [])
        messages = build_rag_messages(query=state["query"], context_chunks=chunks)
    else:
        messages = build_direct_messages(state["query"])

    answer, usage = await generate(messages)

    return {
        "answer": answer,
        "usage": usage,
    }


# ─── Node 5: format_response ──────────────────────────────────────────────────

async def format_response(state: AgentState) -> Dict[str, Any]:
    """
    Assemble the final response dict with sources and metadata.

    Sources are deduplicated by doc_id so the same document cited in
    multiple chunks appears only once.
    """
    chunks = state.get("reranked_chunks") or state.get("retrieved_chunks", [])

    # Deduplicate sources by doc_id
    seen_docs: set = set()
    sources: List[Dict] = []
    for c in chunks:
        doc_id = c.get("doc_id", "unknown")
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            sources.append({
                "doc_id": doc_id,
                "filename": c.get("metadata", {}).get("filename", doc_id),
                "chunk_index": c.get("chunk_index"),
                "similarity_score": c.get("similarity_score"),
                "content_preview": c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"],
            })

    response = {
        "query": state["query"],
        "answer": state.get("answer", ""),
        "sources": sources,
        "metadata": {
            "needs_retrieval": state.get("needs_retrieval", False),
            "routing_reason": state.get("routing_reason", ""),
            "chunks_retrieved": len(chunks),
            "chunk_strategy": state.get("chunk_strategy", settings.default_chunk_strategy),
            "top_k": state.get("top_k", settings.default_top_k),
            "retrieval_latency_seconds": state.get("retrieval_latency_seconds", 0.0),
            "usage": state.get("usage", {}),
        },
    }

    return {"final_response": response}
