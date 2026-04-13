"""
pgvector operations: insert chunks and similarity search.

Design decision: use raw SQLAlchemy Core (not ORM) for vector operations
because pgvector's custom operators (<=> for cosine, <-> for L2) are
easier to express as literal SQL than through ORM abstractions, and
the schema is simple enough that ORM overhead is not worth it.

Connection pooling is handled by SQLAlchemy's default QueuePool,
which keeps 5 persistent connections — fine for a single-node deployment.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import settings

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


# ─── Schema bootstrap ─────────────────────────────────────────────────────────

def ensure_schema() -> None:
    """Create tables and indexes if they do not exist yet."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id              SERIAL PRIMARY KEY,
                doc_id          TEXT NOT NULL,
                chunk_index     INTEGER NOT NULL,
                content         TEXT NOT NULL,
                metadata        JSONB DEFAULT '{}',
                chunk_strategy  TEXT NOT NULL DEFAULT 'sentence',
                embedding       vector(:dim),
                created_at      TIMESTAMPTZ DEFAULT NOW()
            )
        """), {"dim": settings.embedding_dimension})
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
                ON document_chunks (doc_id)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_strategy
                ON document_chunks (chunk_strategy)
        """))
        # HNSW index — works at any dataset size; IVFFlat needs ≥100*lists rows.
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
        """))
    logger.info("Schema ready.")


# ─── Insertion ────────────────────────────────────────────────────────────────

def insert_chunks(
    chunks: List[str],
    embeddings: np.ndarray,
    doc_id: str,
    chunk_strategy: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Bulk-insert chunks + embeddings into document_chunks.

    Returns the number of rows inserted.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    meta_json = json.dumps(metadata or {})
    engine = get_engine()

    # Use psycopg2 directly for bulk insert to avoid SQLAlchemy text() conflicting
    # with PostgreSQL's ::cast syntax during executemany (parameter style mismatch).
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            rows = [
                (
                    doc_id,
                    i,
                    chunk,
                    meta_json,
                    chunk_strategy,
                    "[" + ",".join(str(v) for v in embeddings[i].tolist()) + "]",
                )
                for i, chunk in enumerate(chunks)
            ]
            cur.executemany(
                """
                INSERT INTO document_chunks
                    (doc_id, chunk_index, content, metadata, chunk_strategy, embedding)
                VALUES
                    (%s, %s, %s, %s::jsonb, %s, %s::vector)
                """,
                rows,
            )
        raw_conn.commit()
    finally:
        raw_conn.close()

    logger.info("Inserted %d chunks for doc_id=%r strategy=%r", len(rows), doc_id, chunk_strategy)
    return len(rows)


# ─── Similarity search ────────────────────────────────────────────────────────

def similarity_search(
    query_embedding: np.ndarray,
    top_k: int = 5,
    chunk_strategy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return the top-k most similar chunks via cosine distance (<=>) operator.

    Args:
        query_embedding:  1-D float32 array of shape (embedding_dimension,).
        top_k:            Number of results to return.
        chunk_strategy:   Optional filter; if None, searches all strategies.

    Returns:
        List of dicts with keys: id, doc_id, chunk_index, content,
        metadata, chunk_strategy, similarity_score.
    """
    # Use raw psycopg2 to avoid SQLAlchemy text() conflicting with PostgreSQL
    # ::vector cast syntax (the ':' in '::' is misinterpreted as a bindparam).
    vector_literal = "[" + ",".join(str(v) for v in query_embedding.tolist()) + "]"
    strategy_filter = "AND chunk_strategy = %s" if chunk_strategy else ""

    sql = f"""
        SELECT
            id,
            doc_id,
            chunk_index,
            content,
            metadata,
            chunk_strategy,
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM document_chunks
        WHERE 1=1
        {strategy_filter}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    params: list = [vector_literal]
    if chunk_strategy:
        params.append(chunk_strategy)
    params.extend([vector_literal, top_k])

    engine = get_engine()
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        raw_conn.close()

    return [
        {
            "id": row["id"],
            "doc_id": row["doc_id"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}"),
            "chunk_strategy": row["chunk_strategy"],
            "similarity_score": float(row["similarity_score"]),
        }
        for row in rows
    ]


def delete_chunks_by_doc(doc_id: str, chunk_strategy: Optional[str] = None) -> int:
    """Remove all chunks for a given doc_id (optionally scoped by strategy)."""
    engine = get_engine()
    strategy_filter = "AND chunk_strategy = %s" if chunk_strategy else ""
    sql = f"DELETE FROM document_chunks WHERE doc_id = %s {strategy_filter}"
    params = [doc_id]
    if chunk_strategy:
        params.append(chunk_strategy)

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.execute(sql, params)
            rowcount = cur.rowcount
        raw_conn.commit()
    finally:
        raw_conn.close()
    return rowcount
