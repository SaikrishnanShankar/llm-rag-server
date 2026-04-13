-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Document chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id          SERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    chunk_strategy TEXT NOT NULL DEFAULT 'sentence',
    embedding   vector(384),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for ANN search — works at any dataset size, no minimum row count.
-- m=16 (graph connectivity), ef_construction=64 (build-time recall quality).
-- At query time set: SET hnsw.ef_search = 40; for a recall/speed tradeoff.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Index for filtering by strategy
CREATE INDEX IF NOT EXISTS idx_chunks_strategy
    ON document_chunks (chunk_strategy);

-- Index for doc lookup
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
    ON document_chunks (doc_id);
