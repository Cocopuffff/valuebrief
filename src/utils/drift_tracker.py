"""
drift_tracker.py
~~~~~~~~~~~~~~~~
Tracks the delta between consecutive investment theses for each ticker.

Stores entries in the Supabase ``thesis_drifts`` table (see
``scripts/02-thesis_drifts_ddl.sql``).  The Curator calls ``record_drift()``
after each run so you can query how the thesis has evolved over time.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from schemas import DriftEntry
from utils.db import get_pool
from utils.logger import get_logger

logger = get_logger(__name__)


async def record_drift(
    ticker: str,
    old_verdict: str,
    new_verdict: str,
    old_expected_value: Optional[float],
    new_expected_value: Optional[float],
    key_changes: Optional[list[str]] = None,
) -> DriftEntry:
    """Insert a drift entry for a ticker into Supabase.

    Args:
        ticker:             Stock ticker symbol.
        old_verdict:        Previous thesis verdict (e.g. "Buy on weakness").
        new_verdict:        New thesis verdict.
        old_expected_value: Previous expected intrinsic value.
        new_expected_value: New expected intrinsic value.
        key_changes:        List of human-readable strings describing what changed.

    Returns:
        The created DriftEntry.
    """
    ticker = ticker.upper()
    delta_pct: Optional[float] = None
    if old_expected_value and new_expected_value and old_expected_value != 0:
        delta_pct = round(
            ((new_expected_value - old_expected_value) / old_expected_value) * 100,
            2,
        )

    entry = DriftEntry(
        date=date.today().isoformat(),
        old_verdict=old_verdict,
        new_verdict=new_verdict,
        old_expected_value=old_expected_value,
        new_expected_value=new_expected_value,
        delta_pct=delta_pct,
        key_changes=key_changes or [],
    )

    try:
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO thesis_drifts
                        (ticker, old_verdict, new_verdict, old_ev, new_ev, delta_pct, key_changes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        ticker,
                        old_verdict,
                        new_verdict,
                        old_expected_value,
                        new_expected_value,
                        delta_pct,
                        Jsonb(key_changes or []),
                    ),
                )
    except Exception as e:
        logger.error(f"[DriftTracker] Failed to record drift for {ticker}: {e}")

    logger.info(
        f"[DriftTracker] 📊 {ticker} drift recorded: "
        f"{old_verdict} → {new_verdict} (Δ {delta_pct}%)"
    )
    return entry


async def get_drift_history(ticker: str, limit: int = 10) -> list[DriftEntry]:
    """Return the most recent drift entries for a ticker, newest first."""
    ticker = ticker.upper()

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT old_verdict, new_verdict, old_ev, new_ev, delta_pct,
                       key_changes, created_at::DATE::TEXT AS date
                FROM thesis_drifts
                WHERE ticker = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (ticker, limit),
            )
            rows = await cur.fetchall()

    return [
        DriftEntry(
            date=row["date"],
            old_verdict=row["old_verdict"] or "",
            new_verdict=row["new_verdict"] or "",
            old_expected_value=float(row["old_ev"]) if row["old_ev"] is not None else None,
            new_expected_value=float(row["new_ev"]) if row["new_ev"] is not None else None,
            delta_pct=float(row["delta_pct"]) if row["delta_pct"] is not None else None,
            key_changes=row["key_changes"] or [],
        )
        for row in rows
    ]
