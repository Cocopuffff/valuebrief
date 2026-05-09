-- ============================================================================
-- investment_pillars: Stable thesis-pillar identity and lifecycle table
--
-- Supabase stores queryable identity/lifecycle metadata.  Full pillar dossiers
-- remain in the local vault at detail_path.
-- ============================================================================

CREATE TABLE IF NOT EXISTS investment_pillars (
    pillar_id TEXT PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    pillar_type TEXT NOT NULL,
    canonical_statement TEXT NOT NULL,
    statement_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    version INT NOT NULL DEFAULT 1,
    current_memory_id UUID REFERENCES investment_memories(id) ON DELETE SET NULL,
    detail_path TEXT NOT NULL,
    detail_citation TEXT NOT NULL,
    merged_into_pillar_id TEXT REFERENCES investment_pillars(pillar_id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pillars_ticker_status
    ON investment_pillars (ticker, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_pillars_ticker_type
    ON investment_pillars (ticker, pillar_type);

CREATE INDEX IF NOT EXISTS idx_pillars_statement_hash
    ON investment_pillars (statement_hash);
