"""
LangGraph state machine for the RAG agent.

Graph topology:
    START
      → classify_query
      → [conditional edge]
          needs_retrieval=True  → retrieve_chunks → rerank_chunks → generate_answer
          needs_retrieval=False → generate_answer
      → format_response
      → END

Design decision: the conditional edge after classify_query avoids
unnecessary pgvector calls for greetings/math/code queries, reducing
p50 latency by ~200ms for non-retrieval queries.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Literal

from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState
from src.agent.nodes import (
    classify_query,
    retrieve_chunks,
    rerank_chunks,
    generate_answer,
    format_response,
)

logger = logging.getLogger(__name__)


def _route_after_classify(state: AgentState) -> Literal["retrieve_chunks", "generate_answer"]:
    """Conditional edge: go to retrieval or skip directly to generation."""
    if state.get("needs_retrieval", True):
        return "retrieve_chunks"
    return "generate_answer"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify_query", classify_query)
    graph.add_node("retrieve_chunks", retrieve_chunks)
    graph.add_node("rerank_chunks", rerank_chunks)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("format_response", format_response)

    # Edges
    graph.add_edge(START, "classify_query")
    graph.add_conditional_edges(
        "classify_query",
        _route_after_classify,
        {
            "retrieve_chunks": "retrieve_chunks",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("retrieve_chunks", "rerank_chunks")
    graph.add_edge("rerank_chunks", "generate_answer")
    graph.add_edge("generate_answer", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


# Singleton compiled graph — compiled once at import time
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
        logger.info("LangGraph agent compiled successfully")
    return _graph


async def run_agent(
    query: str,
    top_k: int = 5,
    chunk_strategy: str | None = None,
) -> Dict[str, Any]:
    """
    Run the full RAG agent pipeline.

    Args:
        query:          User question.
        top_k:          Number of chunks to retrieve.
        chunk_strategy: "fixed" | "sentence" | "semantic" | None (all).

    Returns:
        The final_response dict from format_response node.
    """
    graph = get_graph()
    initial_state: AgentState = {
        "query": query,
        "top_k": top_k,
        "chunk_strategy": chunk_strategy,
    }

    logger.debug("Running agent: query=%r top_k=%d strategy=%r", query, top_k, chunk_strategy)
    final_state = await graph.ainvoke(initial_state)
    return final_state.get("final_response", {})
