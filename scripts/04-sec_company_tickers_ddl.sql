-- ============================================================================
-- sec_company_tickers: Persistent SEC ticker-to-CIK map
--
-- Mirrors https://www.sec.gov/files/company_tickers.json so provider.py can
-- load SEC identifiers from Supabase before falling back to a fresh SEC
-- download. The provider also creates this table automatically if needed.
-- ============================================================================

CREATE TABLE IF NOT EXISTS sec_company_tickers (
    ticker TEXT PRIMARY KEY,
    cik CHAR(10) NOT NULL,
    company_name TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sec_company_tickers_cik
    ON sec_company_tickers (cik);
