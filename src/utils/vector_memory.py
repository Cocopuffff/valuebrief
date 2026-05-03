"""
vector_memory.py
~~~~~~~~~~~~~~~~
Supabase client for the ``investment_memories`` vector table.

Provides CRUD operations, similarity search, and pruning functions used by
the Curator Agent.  All operations go through the shared psycopg connection
pool in ``utils.db``.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from utils.db import get_pool
from utils.logger import get_logger
from schemas.rag import InsightRecord, MemoryRecord

logger = get_logger(__name__)


# ── Write operations ─────────────────────────────────────────────────────────

async def store_insight(insight: InsightRecord) -> str:
    """Insert a single insight vector. Returns the generated UUID."""
    pool = await get_pool()
    memory_id = str(uuid.uuid4())

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO investment_memories
                    (id, embedding, summary, metadata, ticker, source_priority, is_cited)
                VALUES (%s, %s::vector, %s, %s, %s, %s, %s)
                """,
                (
                    memory_id,
                    _vec_literal(insight.embedding),
                    insight.summary,
                    Jsonb(insight.metadata),
                    insight.ticker.upper(),
                    insight.source_priority,
                    insight.is_cited,
                ),
            )

    logger.debug(f"[VectorMemory] Stored insight {memory_id[:8]}… for {insight.ticker}")
    return memory_id


async def batch_store_insights(insights: list[InsightRecord]) -> list[str]:
    """Insert multiple insight vectors. Returns list of generated UUIDs."""
    if not insights:
        return []

    ids: list[str] = [str(uuid.uuid4()) for _ in insights]
    placeholders: list[str] = []
    params: list[Any] = []
    pool = await get_pool()

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            for insight, memory_id in zip(insights, ids):
                placeholders.append(
                    "(%s, %s::vector, %s, %s, %s, %s, %s)"
                )
                params.extend([
                    memory_id,
                    _vec_literal(insight.embedding),
                    insight.summary,
                    Jsonb(insight.metadata),
                    insight.ticker.upper(),
                    insight.source_priority,
                    insight.is_cited,
                ])
            query = f"""
                INSERT INTO investment_memories
                    (id, embedding, summary, metadata, ticker, source_priority, is_cited)
                VALUES {', '.join(placeholders)}
            """
            await cur.execute(query, params)

    logger.info(f"[VectorMemory] Batch-stored {len(ids)} insights")
    return ids


# ── Read operations ──────────────────────────────────────────────────────────

async def search_similar(
    query_embedding: list[float],
    ticker: Optional[str] = None,
    limit: int = 10,
) -> list[MemoryRecord]:
    """Cosine-similarity search against the vector index.

    Args:
        query_embedding: The 1536-dim query vector.
        ticker:          Optional ticker filter.
        limit:           Max results to return.

    Returns:
        List of MemoryRecord sorted by similarity (highest first).
    """
    pool = await get_pool()

    if ticker:
        query = """
            SELECT *, 1 - (embedding <=> %s::vector) AS similarity
            FROM investment_memories
            WHERE ticker = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = (_vec_literal(query_embedding), ticker.upper(),
                  _vec_literal(query_embedding), limit)
    else:
        query = """
            SELECT *, 1 - (embedding <=> %s::vector) AS similarity
            FROM investment_memories
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = (_vec_literal(query_embedding),
                  _vec_literal(query_embedding), limit)

    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)
            rows = await cur.fetchall()

    return [_row_to_memory(row) for row in rows]


async def get_memories_for_ticker(
    ticker: str,
    limit: int = 50,
) -> list[MemoryRecord]:
    """Return all memories for a ticker, newest first."""
    pool = await get_pool()

    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT * FROM investment_memories
                WHERE ticker = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (ticker.upper(), limit),
            )
            rows = await cur.fetchall()

    return [_row_to_memory(row) for row in rows]


# ── Pruning operations (used by the Curator) ────────────────────────────────

async def delete_uncited_memories(
    ticker: str,
    before_date: Optional[str] = None,
    created_after: Optional[str] = None,
) -> int:
    """Delete memories that were never cited in a report.

    Args:
        ticker:        Ticker to prune.
        before_date:   ISO date string. Only delete memories created before this date.
        created_after: ISO date string. Only delete memories created after this date.
                       Use this to scope deletion to the active window.
        If neither date is set, deletes all uncited memories for the ticker.

    Returns:
        Number of rows deleted.
    """
    pool = await get_pool()

    clauses = ["ticker = %s", "is_cited = false"]
    params: list = [ticker.upper()]

    if before_date:
        clauses.append("created_at < %s")
        params.append(before_date)
    if created_after:
        clauses.append("created_at >= %s")
        params.append(created_after)

    where = " AND ".join(clauses)

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"DELETE FROM investment_memories WHERE {where}",
                tuple(params),
            )
            deleted = cur.rowcount

    logger.info(f"[VectorMemory] 🗑️ Deleted {deleted} uncited memories for {ticker}")
    return deleted


async def delete_low_priority_memories(
    ticker: str,
    before_date: str,
    max_priority: int = 1,
) -> int:
    """Delete memories below a priority threshold that are older than a date.

    Args:
        ticker:       Ticker to prune.
        before_date:  ISO date string.
        max_priority: Delete memories with source_priority <= this value.

    Returns:
        Number of rows deleted.
    """
    pool = await get_pool()

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                DELETE FROM investment_memories
                WHERE ticker = %s
                  AND created_at < %s
                  AND source_priority <= %s
                """,
                (ticker.upper(), before_date, max_priority),
            )
            deleted = cur.rowcount

    logger.info(
        f"[VectorMemory] 🗑️ Deleted {deleted} low-priority (≤{max_priority}) "
        f"memories for {ticker} before {before_date}"
    )
    return deleted


async def mark_memories_cited(memory_ids: list[str]) -> int:
    """Set is_cited = true for a list of memory UUIDs.

    Returns:
        Number of rows updated.
    """
    if not memory_ids:
        return 0

    pool = await get_pool()

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # Use ANY with array parameter for batch update
            await cur.execute(
                """
                UPDATE investment_memories
                SET is_cited = true, updated_at = NOW()
                WHERE id = ANY(%s::uuid[])
                """,
                (memory_ids,),
            )
            updated = cur.rowcount

    logger.info(f"[VectorMemory] ✅ Marked {updated} memories as cited")
    return updated


async def merge_to_summary_vector(
    ticker: str,
    memory_ids: list[str],
    summary_text: str,
    embedding: list[float],
) -> str:
    """Replace N granular vectors with a single summary vector.

    Deletes the specified memories and inserts one consolidated summary.

    Args:
        ticker:       Ticker symbol.
        memory_ids:   UUIDs of memories to merge.
        summary_text: The consolidated summary text.
        embedding:    Embedding of the summary text.

    Returns:
        UUID of the new summary memory.
    """
    pool = await get_pool()
    new_id = str(uuid.uuid4())

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # Delete the granular memories
            if memory_ids:
                await cur.execute(
                    "DELETE FROM investment_memories WHERE id = ANY(%s::uuid[])",
                    (memory_ids,),
                )
                deleted = cur.rowcount
            else:
                deleted = 0

            # Insert the summary
            await cur.execute(
                """
                INSERT INTO investment_memories
                    (id, embedding, summary, metadata, ticker, source_priority, is_cited)
                VALUES (%s, %s::vector, %s, %s, %s, %s, %s)
                """,
                (
                    new_id,
                    _vec_literal(embedding),
                    summary_text,
                    Jsonb({"type": "historical_summary", "merged_count": deleted}),
                    ticker.upper(),
                    2,      # Summary vectors get first-party priority
                    True,   # Summaries are considered "cited" (they synthesise cited material)
                ),
            )

    logger.info(
        f"[VectorMemory] 🔄 Merged {deleted} memories into summary {new_id[:8]}… for {ticker}"
    )
    return new_id


# ── Storage health check ─────────────────────────────────────────────────────

async def get_table_size_bytes() -> int:
    """Return the total size of the investment_memories table in bytes.

    Uses pg_total_relation_size which includes indexes and TOAST.
    """
    pool = await get_pool()

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT pg_total_relation_size('investment_memories')"
            )
            row = await cur.fetchone()
            return row[0] if row else 0


async def get_memory_count(ticker: Optional[str] = None) -> int:
    """Return the total number of memories, optionally filtered by ticker."""
    pool = await get_pool()

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            if ticker:
                await cur.execute(
                    "SELECT COUNT(*) FROM investment_memories WHERE ticker = %s",
                    (ticker.upper(),),
                )
            else:
                await cur.execute("SELECT COUNT(*) FROM investment_memories")
            row = await cur.fetchone()
            return row[0] if row else 0


async def get_oldest_memories(
    ticker: str,
    limit: int = 50,
) -> list[MemoryRecord]:
    """Return the oldest memories for a ticker (for aggressive pruning)."""
    pool = await get_pool()

    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT * FROM investment_memories
                WHERE ticker = %s
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (ticker.upper(), limit),
            )
            rows = await cur.fetchall()

    return [_row_to_memory(row) for row in rows]

# ── Lifecycle operations (used by Curator consolidation) ─────────────────────


async def delete_memories_for_period(
    ticker: str,
    start_date: str,
    end_date: str,
) -> int:
    """Delete all granular (non-summary) memories for a ticker within a date range.

    Used during monthly consolidation to clean up individual vectors that are
    being replaced by a single summary vector.

    Args:
        ticker:     Ticker symbol.
        start_date: ISO date (inclusive).
        end_date:   ISO date (exclusive).

    Returns:
        Number of rows deleted.
    """
    pool = await get_pool()

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                DELETE FROM investment_memories
                WHERE ticker = %s
                  AND created_at >= %s
                  AND created_at < %s
                  AND (metadata->>'type') IS DISTINCT FROM 'monthly_summary'
                """,
                (ticker.upper(), start_date, end_date),
            )
            deleted = cur.rowcount

    logger.info(
        f"[VectorMemory] 🗑️ Deleted {deleted} granular memories "
        f"for {ticker} ({start_date} to {end_date})"
    )
    return deleted


async def upsert_summary_vector(
    ticker: str,
    month: str,
    summary_text: str,
    embedding: list[float],
    extra_metadata: Optional[dict] = None,
) -> str:
    """Create or replace the monthly summary vector for a ticker.

    Atomically deletes any existing summary for this ticker+month and
    inserts a fresh one.  Ensures exactly one summary vector per month.

    Args:
        ticker:         Ticker symbol.
        month:          Month string, e.g. ``"2026-01"``.
        summary_text:   The distilled summary text.
        embedding:      Embedding vector of the summary.
        extra_metadata: Optional additional metadata fields.

    Returns:
        UUID of the new summary vector.
    """
    pool = await get_pool()
    new_id = str(uuid.uuid4())
    metadata = {"type": "monthly_summary", "month": month}
    if extra_metadata:
        metadata.update(extra_metadata)

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # Remove any existing summary for this month
            await cur.execute(
                """
                DELETE FROM investment_memories
                WHERE ticker = %s
                  AND metadata->>'type' = 'monthly_summary'
                  AND metadata->>'month' = %s
                """,
                (ticker.upper(), month),
            )

            # Insert the fresh summary
            await cur.execute(
                """
                INSERT INTO investment_memories
                    (id, embedding, summary, metadata, ticker, source_priority, is_cited)
                VALUES (%s, %s::vector, %s, %s, %s, %s, %s)
                """,
                (
                    new_id,
                    _vec_literal(embedding),
                    summary_text,
                    Jsonb(metadata),
                    ticker.upper(),
                    2,     # Summary vectors get first-party priority
                    True,  # Summaries are always considered "cited"
                ),
            )

    logger.info(f"[VectorMemory] 📊 Upserted summary vector for {ticker}/{month}")
    return new_id


# ── Helpers ──────────────────────────────────────────────────────────────────

def _vec_literal(vec: list[float]) -> str:
    """Convert a Python list to a pgvector literal string: '[0.1,0.2,...]'."""
    return "[" + ",".join(str(v) for v in vec) + "]"


def _row_to_memory(row: dict) -> MemoryRecord:
    """Convert a database row dict to a MemoryRecord."""
    return MemoryRecord(
        id=str(row["id"]),
        summary=row["summary"],
        metadata=row.get("metadata", {}),
        ticker=row["ticker"],
        source_priority=row.get("source_priority", 0),
        is_cited=row.get("is_cited", False),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        similarity=row.get("similarity"),
    )
