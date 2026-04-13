"""
Integration tests for vectorstore insert + search round-trip.

Requires a running PostgreSQL with pgvector on DATABASE_URL.
Skipped automatically when the DB is unavailable (CI without a DB service).
"""
import json
import os
import pytest
import numpy as np

# Skip the whole module when DB is not available
pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_DB_TESTS", "false").lower() == "true",
    reason="DB tests skipped (SKIP_DB_TESTS=true)",
)


@pytest.fixture(scope="module")
def db_available():
    """Skip all tests in this module if the DB cannot be reached."""
    try:
        from src.retrieval.vectorstore import ensure_schema, get_engine
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        ensure_schema()
        return True
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


@pytest.fixture(scope="module")
def test_doc_id():
    return "pytest_test_doc_abc123"


@pytest.fixture(autouse=True, scope="module")
def cleanup(db_available, test_doc_id):
    """Remove test chunks before and after the test run."""
    from src.retrieval.vectorstore import delete_chunks_by_doc
    delete_chunks_by_doc(test_doc_id)
    yield
    delete_chunks_by_doc(test_doc_id)


def make_random_embeddings(n: int, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    vecs = rng.standard_normal((n, dim)).astype("float32")
    # Normalise so cosine similarity is well-defined
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


class TestInsertChunks:
    def test_insert_returns_count(self, db_available, test_doc_id):
        from src.retrieval.vectorstore import insert_chunks
        chunks = ["chunk one about ML", "chunk two about RAG", "chunk three about vLLM"]
        embeddings = make_random_embeddings(len(chunks))
        n = insert_chunks(chunks, embeddings, test_doc_id, "sentence", {"source": "pytest"})
        assert n == 3

    def test_insert_length_mismatch_raises(self, db_available, test_doc_id):
        from src.retrieval.vectorstore import insert_chunks
        with pytest.raises(ValueError, match="same length"):
            insert_chunks(["a", "b"], make_random_embeddings(3), test_doc_id, "sentence")


class TestSimilaritySearch:
    def test_search_returns_results(self, db_available, test_doc_id):
        from src.retrieval.vectorstore import similarity_search
        query_vec = make_random_embeddings(1)[0]
        results = similarity_search(query_vec, top_k=3, chunk_strategy="sentence")
        assert isinstance(results, list)
        # At minimum the 3 chunks inserted above should be searchable
        assert len(results) >= 1

    def test_search_result_schema(self, db_available, test_doc_id):
        from src.retrieval.vectorstore import similarity_search
        query_vec = make_random_embeddings(1)[0]
        results = similarity_search(query_vec, top_k=1, chunk_strategy="sentence")
        assert len(results) >= 1
        r = results[0]
        assert set(r.keys()) >= {"id", "doc_id", "content", "similarity_score", "chunk_strategy"}
        assert isinstance(r["similarity_score"], float)
        assert -1.0 <= r["similarity_score"] <= 1.0

    def test_search_strategy_filter(self, db_available, test_doc_id):
        from src.retrieval.vectorstore import similarity_search
        query_vec = make_random_embeddings(1)[0]
        results = similarity_search(query_vec, top_k=5, chunk_strategy="fixed")
        # All returned chunks should have the requested strategy
        for r in results:
            assert r["chunk_strategy"] == "fixed"

    def test_search_top_k_respected(self, db_available, test_doc_id):
        from src.retrieval.vectorstore import similarity_search
        query_vec = make_random_embeddings(1)[0]
        results = similarity_search(query_vec, top_k=2)
        assert len(results) <= 2

    def test_metadata_is_dict(self, db_available, test_doc_id):
        from src.retrieval.vectorstore import similarity_search
        query_vec = make_random_embeddings(1)[0]
        results = similarity_search(query_vec, top_k=3, chunk_strategy="sentence")
        for r in results:
            assert isinstance(r["metadata"], dict)
