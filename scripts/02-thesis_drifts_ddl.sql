-- Append-only history of thesis verdict changes per ticker.
-- Rows are never updated — each run that changes the thesis adds a new row.

CREATE TABLE thesis_drifts (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker      VARCHAR(10)  NOT NULL,
    old_verdict TEXT,
    new_verdict TEXT,
    old_ev      DECIMAL(12, 2),
    new_ev      DECIMAL(12, 2),
    delta_pct   DECIMAL(8, 2),
    key_changes JSONB NOT NULL DEFAULT '[]',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_thesis_drifts_ticker      ON thesis_drifts (ticker);
CREATE INDEX idx_thesis_drifts_ticker_date ON thesis_drifts (ticker, created_at DESC);
