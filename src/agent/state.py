"""LangGraph state schema for the RAG agent."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # Input
    query: str
    top_k: int
    chunk_strategy: Optional[str]   # None = search all strategies

    # Routing
    needs_retrieval: bool
    routing_reason: str

    # Retrieval
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_latency_seconds: float

    # Reranking
    reranked_chunks: List[Dict[str, Any]]

    # Generation
    answer: str
    usage: Dict[str, Any]

    # Output
    final_response: Dict[str, Any]
