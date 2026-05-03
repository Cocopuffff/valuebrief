-- ============================================================================
-- investment_memories: Vector index for Hybrid RAG "Hot Memory"
--
-- Stores LLM-distilled insight summaries as embeddings. No raw text — only
-- 1-2 sentence insight vectors with metadata pointers to local vault files.
--
-- Embedding model: OpenAI text-embedding-3-small (1536 dimensions)
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS investment_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- The embedding vector (1536 dims for text-embedding-3-small)
    embedding vector(1536),

    -- LLM-distilled 1-2 sentence insight (the only text stored in Supabase)
    summary TEXT NOT NULL,

    -- Flexible metadata: ticker, date, source_priority, local_path, source_type, etc.
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Denormalised fields for efficient filtering without JSONB unpacking
    ticker VARCHAR(10) NOT NULL,
    source_priority INT NOT NULL DEFAULT 0,   -- 0 = noise, 1 = secondary, 2 = first-party (10-K/Q)
    is_cited BOOLEAN NOT NULL DEFAULT false,   -- Set to true by the Curator when used in a report

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Indexes ──────────────────────────────────────────────────────────────────

-- Fast lookup by ticker
CREATE INDEX IF NOT EXISTS idx_memories_ticker
    ON investment_memories (ticker);

-- Fast lookup by ticker + recency (used heavily by the Curator for pruning)
CREATE INDEX IF NOT EXISTS idx_memories_ticker_date
    ON investment_memories (ticker, created_at DESC);

-- Fast lookup by memory type (used by Curator consolidation and pruning)
CREATE INDEX IF NOT EXISTS idx_memories_type
    ON investment_memories ((metadata->>'type'));

-- HNSW vector index for approximate nearest-neighbor search
-- HNSW offers better recall and query performance than IVFFlat at the cost of
-- slightly higher memory usage — acceptable for our sub-500MB constraint.
CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
    ON investment_memories
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
