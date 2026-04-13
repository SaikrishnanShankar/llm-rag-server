"""
Unit tests for the LangGraph agent — runs without a database or LLM.

All external I/O (vectorstore, vLLM) is patched so tests are fast and
hermetic. This validates routing logic and node contracts, not retrieval
quality.
"""
import pytest


# ─── classify_query ───────────────────────────────────────────────────────────

class TestClassifyQuery:
    """Test the routing heuristic before the LLM fallback fires."""

    @pytest.mark.asyncio
    async def test_hello_routes_direct(self):
        from src.agent.nodes import classify_query
        from src.agent.state import AgentState
        state: AgentState = {"query": "hello there", "top_k": 5}
        result = await classify_query(state)
        assert result["needs_retrieval"] is False

    @pytest.mark.asyncio
    async def test_what_routes_retrieval(self):
        from src.agent.nodes import classify_query
        from src.agent.state import AgentState
        state: AgentState = {"query": "what is RAG?", "top_k": 5}
        result = await classify_query(state)
        assert result["needs_retrieval"] is True

    @pytest.mark.asyncio
    async def test_explain_routes_retrieval(self):
        from src.agent.nodes import classify_query
        from src.agent.state import AgentState
        state: AgentState = {"query": "explain transformers", "top_k": 5}
        result = await classify_query(state)
        assert result["needs_retrieval"] is True

    @pytest.mark.asyncio
    async def test_how_does_routes_retrieval(self):
        from src.agent.nodes import classify_query
        from src.agent.state import AgentState
        state: AgentState = {"query": "how does vLLM work?", "top_k": 5}
        result = await classify_query(state)
        assert result["needs_retrieval"] is True

    @pytest.mark.asyncio
    async def test_thanks_routes_direct(self):
        from src.agent.nodes import classify_query
        from src.agent.state import AgentState
        state: AgentState = {"query": "thanks!", "top_k": 5}
        result = await classify_query(state)
        assert result["needs_retrieval"] is False

    @pytest.mark.asyncio
    async def test_routing_reason_always_present(self):
        from src.agent.nodes import classify_query
        from src.agent.state import AgentState
        for query in ["hello", "what is ML?", "describe deep learning"]:
            state: AgentState = {"query": query, "top_k": 5}
            result = await classify_query(state)
            assert "routing_reason" in result
            assert result["routing_reason"]


# ─── rerank_chunks ────────────────────────────────────────────────────────────

class TestRerankChunks:
    """Reranking should return chunks in some order, never drop them."""

    @pytest.mark.asyncio
    async def test_rerank_preserves_all_chunks(self):
        from src.agent.nodes import rerank_chunks
        from src.agent.state import AgentState
        chunks = [
            {"content": "RAG combines LLMs with retrieval.", "doc_id": "d1",
             "chunk_index": 0, "similarity_score": 0.9, "metadata": {}},
            {"content": "pgvector stores embeddings.", "doc_id": "d2",
             "chunk_index": 0, "similarity_score": 0.7, "metadata": {}},
            {"content": "vLLM serves LLMs at scale.", "doc_id": "d3",
             "chunk_index": 0, "similarity_score": 0.5, "metadata": {}},
        ]
        state: AgentState = {"query": "what is RAG", "retrieved_chunks": chunks, "top_k": 5}
        result = await rerank_chunks(state)
        assert len(result["reranked_chunks"]) == 3

    @pytest.mark.asyncio
    async def test_rerank_empty_chunks(self):
        from src.agent.nodes import rerank_chunks
        from src.agent.state import AgentState
        state: AgentState = {"query": "anything", "retrieved_chunks": [], "top_k": 5}
        result = await rerank_chunks(state)
        assert result["reranked_chunks"] == []


# ─── format_response ─────────────────────────────────────────────────────────

class TestFormatResponse:
    """format_response should deduplicate sources and include required keys."""

    @pytest.mark.asyncio
    async def test_format_deduplicates_by_doc_id(self):
        from src.agent.nodes import format_response
        from src.agent.state import AgentState
        # Two chunks from the same doc
        chunks = [
            {"content": "chunk A", "doc_id": "doc1", "chunk_index": 0,
             "similarity_score": 0.9, "metadata": {"filename": "f.txt"}},
            {"content": "chunk B", "doc_id": "doc1", "chunk_index": 1,
             "similarity_score": 0.8, "metadata": {"filename": "f.txt"}},
            {"content": "chunk C", "doc_id": "doc2", "chunk_index": 0,
             "similarity_score": 0.7, "metadata": {"filename": "g.txt"}},
        ]
        state: AgentState = {
            "query": "test query",
            "answer": "test answer",
            "reranked_chunks": chunks,
            "needs_retrieval": True,
            "routing_reason": "keyword",
            "retrieval_latency_seconds": 0.01,
            "usage": {"total_tokens": 100},
            "top_k": 5,
            "chunk_strategy": "sentence",
        }
        result = await format_response(state)
        response = result["final_response"]
        # doc1 appears in 2 chunks but only 1 source entry
        assert len(response["sources"]) == 2
        doc_ids = [s["doc_id"] for s in response["sources"]]
        assert len(set(doc_ids)) == len(doc_ids)

    @pytest.mark.asyncio
    async def test_format_response_schema(self):
        from src.agent.nodes import format_response
        from src.agent.state import AgentState
        state: AgentState = {
            "query": "my question",
            "answer": "my answer",
            "reranked_chunks": [],
            "needs_retrieval": False,
            "routing_reason": "direct",
            "retrieval_latency_seconds": 0.0,
            "usage": {},
            "top_k": 5,
            "chunk_strategy": "sentence",
        }
        result = await format_response(state)
        response = result["final_response"]
        for key in ("query", "answer", "sources", "metadata"):
            assert key in response
        for key in ("needs_retrieval", "routing_reason", "chunks_retrieved",
                    "chunk_strategy", "top_k", "retrieval_latency_seconds", "usage"):
            assert key in response["metadata"]


# ─── mock_client ─────────────────────────────────────────────────────────────

class TestMockClient:
    """Mock LLM client should produce non-empty, context-grounded answers."""

    @pytest.mark.asyncio
    async def test_mock_returns_answer(self):
        from src.inference.mock_client import MockLLMClient
        client = MockLLMClient()
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Context passages: RAG combines LLMs with retrieval."},
                {"role": "user", "content": "What is RAG?"},
            ],
            max_tokens=128,
        )
        assert response.choices[0].message.content
        assert len(response.choices[0].message.content) > 10

    @pytest.mark.asyncio
    async def test_mock_usage_stats(self):
        from src.inference.mock_client import MockLLMClient
        client = MockLLMClient()
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=64,
        )
        assert response.usage.total_tokens > 0
        assert response.usage.prompt_tokens > 0

    def test_mock_is_mock_property(self):
        from src.inference.mock_client import MockLLMClient
        assert MockLLMClient().is_mock is True
