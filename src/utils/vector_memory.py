"""
vector_memory.py
~~~~~~~~~~~~~~~~
Supabase client for the ``investment_memories`` vector table.

Provides CRUD operations, similarity search, and pruning functions used by
the Curator Agent.  All operations go through the shared psycopg connection
pool in ``utils.db``.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import pathlib
import re
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from utils.db import get_pool
from utils.logger import get_logger
from schemas.rag import InsightRecord, MemoryRecord

logger = get_logger(__name__)
_HEADING_ONLY_RE = re.compile(r"^#{1,6}\s+\S")

ACTIVE_PILLAR_STATUSES = ("supported", "weakened")
INACTIVE_PILLAR_STATUSES = ("contradicted", "superseded", "stale")
HIGH_PILLAR_SIMILARITY = float(os.getenv("PILLAR_SIMILARITY_HIGH", "0.86"))
MEDIUM_PILLAR_SIMILARITY = float(os.getenv("PILLAR_SIMILARITY_MEDIUM", "0.72"))
PILLAR_DUPLICATE_SIMILARITY = float(os.getenv("PILLAR_DUPLICATE_SIMILARITY", "0.90"))
SUMMARY_DUPLICATE_SIMILARITY = float(os.getenv("SUMMARY_DUPLICATE_SIMILARITY", "0.92"))


# ── Retrieval probes ─────────────────────────────────────────────────────────
# Fixed semantic probes for multi-dimensional prior-research retrieval.
# Each targets a distinct facet of equity analysis.

_PROBES: dict[str, str] = {
    "moat_growth": (
        "competitive advantage moat durable growth catalysts market share "
        "expansion revenue growth trajectory"
    ),
    "risks": (
        "key risks competitive threats regulatory headwinds margin compression "
        "earnings quality debt leverage"
    ),
    "valuation_assumptions": (
        "valuation assumptions DCF assumptions intrinsic value margin of safety "
        "WACC terminal growth revenue projections"
    ),
    "thesis_drift": (
        "thesis change drift prior valuation estimate revision broken assumptions "
        "confirmed weakened revised contradicted superseded stale verdict shift"
    ),
    "prior_valuation": (
        "prior valuation intrinsic value expected CAGR scenario analysis "
        "buy sell hold recommendation price target"
    ),
}


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
    *,
    min_source_priority: Optional[int] = None,
    cited_only: bool = False,
    source_types: Optional[list[str]] = None,
    min_similarity: Optional[float] = None,
    exclude_validity_statuses: Optional[list[str]] = None,
    validity_statuses: Optional[list[str]] = None,
) -> list[MemoryRecord]:
    """Descending order cosine-similarity search against the vector index.

    Args:
        query_embedding:           The 1536-dim query vector.
        ticker:                    Optional ticker filter.
        limit:                     Max results to return.
        min_source_priority:       Only memories with source_priority >= this.
        cited_only:                Only memories with is_cited = true.
        source_types:              Filter by metadata->>'source_type' or metadata->>'type'.
        min_similarity:            Minimum cosine similarity threshold (0.0-1.0).
        exclude_validity_statuses: Exclude memories with these validity_status values
                                   (e.g. ["contradicted", "stale"]).
        validity_statuses:         Only include memories with these validity_status values.

    Returns:
        List of MemoryRecord sorted by similarity (highest first).
    """
    pool = await get_pool()

    vec_literal = _vec_literal(query_embedding)
    clauses: list[str] = []
    params: list[Any] = [vec_literal]

    if ticker:
        clauses.append("ticker = %s")
        params.append(ticker.upper())

    if min_source_priority is not None:
        clauses.append("source_priority >= %s")
        params.append(min_source_priority)

    if cited_only:
        clauses.append("is_cited = true")

    if source_types:
        clauses.append(
            "(metadata->>'source_type' = ANY(%s::text[]) "
            "OR metadata->>'type' = ANY(%s::text[]))"
        )
        params.extend([source_types, source_types])

    if exclude_validity_statuses:
        clauses.append(
            "(metadata->>'validity_status' IS NULL "
            "OR metadata->>'validity_status' <> ALL(%s::text[]))"
        )
        params.append(exclude_validity_statuses)

    if validity_statuses:
        clauses.append("metadata->>'validity_status' = ANY(%s::text[])")
        params.append(validity_statuses)

    # Build WHERE clause
    where = ""
    if clauses:
        where = "WHERE " + " AND ".join(clauses)

    # min_similarity filters on the computed cosine similarity expression.
    similarity_filter = ""
    if min_similarity is not None:
        prefix = "AND" if clauses else "WHERE"
        similarity_filter = f" {prefix} 1 - (embedding <=> %s::vector) >= %s"
        params.extend([vec_literal, min_similarity])

    query = f"""
        SELECT *, 1 - (embedding <=> %s::vector) AS similarity
        FROM investment_memories
        {where}
        {similarity_filter}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    # Add ordering + limit params
    params.extend([vec_literal, limit])

    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, tuple(params))
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


# ── Vault-backed context formatting ─────────────────────────────────────────

@dataclass(frozen=True)
class ResolvedMemoryText:
    source_excerpt: str
    citation: str
    fallback_reason: str = ""


def _truncate_text(text: str, max_chars: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= max_chars:
        return clean
    clipped = clean[:max_chars].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].rstrip()
    return clipped + "..."


def _is_heading_only(text: str) -> bool:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return bool(lines) and all(_HEADING_ONLY_RE.match(line) for line in lines)


def _candidate_vault_paths(memory: MemoryRecord) -> list[pathlib.Path]:
    from utils.vault import VAULT_ROOT

    metadata = memory.metadata or {}
    candidates: list[pathlib.Path] = []
    local_path = metadata.get("local_path")
    filename = metadata.get("filename")

    if local_path:
        path = pathlib.Path(str(local_path))
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(pathlib.Path.cwd() / path)

    if filename:
        candidates.append(VAULT_ROOT / memory.ticker.upper() / str(filename))

    seen: set[str] = set()
    unique: list[pathlib.Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _resolve_legacy_heading_block(
    block_map: dict[str, str],
    block_id: str,
) -> Optional[str]:
    items = list(block_map.items())
    for idx, (candidate_id, text) in enumerate(items):
        if candidate_id != block_id:
            continue
        if not _is_heading_only(text):
            return text
        for _, next_text in items[idx + 1:]:
            if next_text.strip() and not _is_heading_only(next_text):
                return f"{text.strip()}\n\n{next_text.strip()}"
        return text
    return None


def resolve_memory_source(memory: MemoryRecord) -> ResolvedMemoryText:
    """Resolve a vector memory to the source vault text shown to analysts."""
    metadata = memory.metadata or {}
    block_id = metadata.get("block_id")
    citation = str(metadata.get("citation") or "")

    if block_id:
        from utils.vault import VaultReader

        reader = VaultReader()
        for path in _candidate_vault_paths(memory):
            if not path.exists():
                continue
            try:
                doc = reader.read_document(path)
                source = _resolve_legacy_heading_block(doc.block_map, str(block_id))
                if source:
                    if not citation:
                        citation = f"{path.name}#^{block_id}"
                    return ResolvedMemoryText(
                        source_excerpt=source,
                        citation=citation,
                        fallback_reason="",
                    )
            except Exception as e:
                logger.debug("[VectorMemory] Failed to resolve %s: %s", path, e)

    return ResolvedMemoryText(
        source_excerpt=memory.summary,
        citation=citation,
        fallback_reason="summary_fallback",
    )


def format_memory_for_context(
    memory: MemoryRecord,
    index: int,
    *,
    source_excerpt_chars: int = 1500,
    summary_chars: int = 320,
) -> str:
    """Format a retrieved memory with a vault source excerpt for prompts/tools."""
    sim_str = f" (similarity: {memory.similarity:.2f})" if memory.similarity else ""
    source = memory.metadata.get("source_type", memory.metadata.get("type", "unknown"))
    validity = memory.metadata.get("validity_status", "")
    badge = f" [previously {validity}]" if validity else ""
    resolved = resolve_memory_source(memory)
    summary = _truncate_text(memory.summary, summary_chars)
    source_excerpt = _truncate_text(resolved.source_excerpt, source_excerpt_chars)

    lines = [
        f"### Memory #{index}{badge}{sim_str}",
        f"**Source**: {source} | **Priority**: {memory.source_priority}",
    ]
    if resolved.citation:
        lines.append(f"**Citation**: (See: {resolved.citation})")
    if summary:
        lines.append(f"**Summary**: {summary}")
    if source_excerpt:
        lines.append(f"**Source excerpt**:\n{source_excerpt}")
    if resolved.fallback_reason:
        lines.append("**Source excerpt fallback**: vault block unavailable; used stored summary.")
    return "\n".join(lines)


# ── Multi-probe retrieval ────────────────────────────────────────────────────

async def retrieve_research_memories(
    ticker: str,
    *,
    limit_per_probe: int = 5,
    min_source_priority: Optional[int] = None,
    source_types: Optional[list[str]] = None,
    min_similarity: float = 0.65,
    include_contradicted: bool = False,
) -> dict:
    """Retrieve and rank prior research memories for a ticker using fixed probes.

    Embeds five semantic probes (moat/growth, risks, valuation assumptions,
    thesis drift, prior valuation) and merges results with a weighted scoring
    formula that rewards first-party, cited, and active supported/weakened memories.
    Deduplicates by memory ID; returns the top-ranked results.

    Args:
        ticker:               Ticker symbol.
        limit_per_probe:      Max results per probe query.
        min_source_priority:  Optional minimum source_priority.
        source_types:         Optional filter by metadata source_type/type.
        min_similarity:       Minimum cosine similarity (0.0-1.0).
        include_contradicted: If False (default), excludes memories with
                              validity_status in ("contradicted", "stale").
                              The thesis-drift probe always includes them.

    Returns:
        A dict with keys: memory_context (str), retrieved_memory_ids (list[str]),
        topics (list[dict]), error (str | None).
    """
    try:
        from utils.embeddings import get_embeddings

        probe_texts = list(_PROBES.values())
        probe_names = list(_PROBES.keys())

        # Single batch embedding call for all probes
        embeddings = await get_embeddings(probe_texts)
        if not embeddings or len(embeddings) != len(probe_texts):
            logger.warning("[VectorMemory] Embeddings returned empty/incomplete")
            return _empty_retrieval_result()

        # Default: exclude contradicted/stale. Thesis-drift probe overrides.
        default_exclude = None if include_contradicted else ["contradicted", "stale"]

        # Launch all probe searches in parallel
        tasks = []
        for i, emb in enumerate(embeddings):
            exclude = default_exclude
            if probe_names[i] == "thesis_drift":
                # Thesis-drift probe always includes contradicted so we can
                # explain how the thesis changed over time
                exclude = None
            tasks.append(
                search_similar(
                    emb,
                    ticker=ticker,
                    limit=limit_per_probe * 3,  # oversample for post-merge ranking
                    min_source_priority=min_source_priority,
                    source_types=source_types,
                    min_similarity=min_similarity,
                    exclude_validity_statuses=exclude,
                )
            )

        probe_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge: deduplicate by id, keep highest similarity
        seen: dict[str, tuple[MemoryRecord, float]] = {}
        for i, result in enumerate(probe_results):
            if isinstance(result, Exception):
                logger.warning(
                    "[VectorMemory] Probe '%s' failed: %s", probe_names[i], result
                )
                continue
            if not isinstance(result, list):
                continue
            for mem in result:
                sim = mem.similarity or 0.0
                # Weighted score: similarity * (1 + 0.3 * source_priority)
                score = sim * (1.0 + 0.3 * mem.source_priority)
                # Validity_status bonus/penalty
                vs = mem.metadata.get("validity_status", "")
                if vs in ("supported", "weakened"):
                    score += 0.2
                elif vs == "contradicted":
                    score -= 0.5
                if mem.id not in seen or score > seen[mem.id][1]:
                    seen[mem.id] = (mem, score)

        # Sort by weighted score desc, take top 8
        ranked = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        top_memories = [m for m, _ in ranked[:8]]

        if not top_memories:
            return _empty_retrieval_result()

        # Build memory_context prose
        context_lines = [
            f"## Prior Research Memories for {ticker.upper()}",
            (
                "The following insights were retrieved from past analyses. "
                "Treat them as HYPOTHESES to verify, not as accepted facts. "
                "You must confirm, contradict, or update each with fresh research.\n"
            ),
        ]
        for idx, m in enumerate(top_memories, 1):
            context_lines.append(format_memory_for_context(m, idx))
            context_lines.append("")

        memory_context = "\n".join(context_lines)

        # Rule-based topic generation
        topics = _build_retrieval_topics(top_memories)

        return {
            "memory_context": memory_context,
            "retrieved_memory_ids": [m.id for m in top_memories],
            "topics": [t.model_dump(mode="json") for t in topics],
            "error": None,
        }
    except Exception as e:
        logger.warning("[VectorMemory] retrieve_research_memories failed: %s", e)
        return _empty_retrieval_result()


def _empty_retrieval_result() -> dict:
    return {
        "memory_context": "",
        "retrieved_memory_ids": [],
        "topics": [],
        "error": None,
    }


def _build_retrieval_topics(memories: list[MemoryRecord]) -> list:
    """Derive structured ResearchTopics from retrieved memories (rule-based, not LLM)."""
    from schemas.rag import ResearchTopic

    topics: list[ResearchTopic] = []
    if not memories:
        return topics

    # Separate memories by source sentiment for side assignment
    bull_evidence = [
        m for m in memories
        if m.metadata.get("sentiment") in ("positive", "bullish")
        or m.metadata.get("source_type") in ("bull_thesis",)
    ]
    bear_evidence = [
        m for m in memories
        if m.metadata.get("sentiment") in ("negative", "bearish")
        or m.metadata.get("source_type") in ("bear_thesis",)
    ]
    valuation_drift = [
        m for m in memories
        if m.metadata.get("source_type") in ("judge_analysis", "monthly_summary", "historical_summary")
        or m.metadata.get("type") in ("monthly_summary", "historical_summary")
    ]

    # Bull topics
    if bull_evidence:
        topics.append(ResearchTopic(
            side="bull",
            question="Confirm or challenge prior bullish evidence",
            rationale=(
                "Prior analyses identified positive catalysts. Verify if these "
                "remain valid given current market conditions."
            ),
            evidence_memory_ids=[m.id for m in bull_evidence[:3]],
        ))

    # Bear topics
    if bear_evidence:
        topics.append(ResearchTopic(
            side="bear",
            question="Confirm or challenge prior bearish risks",
            rationale=(
                "Prior analyses identified risk factors. Verify if these have "
                "materialized or intensified."
            ),
            evidence_memory_ids=[m.id for m in bear_evidence[:3]],
        ))

    # Thesis drift topic
    if valuation_drift:
        topics.append(ResearchTopic(
            side="neutral",
            question="Detect thesis drift: what has changed since the last analysis?",
            rationale=(
                "Prior valuation, verdict, and assumptions may no longer hold. "
                "Identify material changes from the last analysis."
            ),
            evidence_memory_ids=[m.id for m in valuation_drift[:5]],
        ))
    elif len(memories) >= 1:
        # Fallback: generic drift question even without explicit valuation memories
        topics.append(ResearchTopic(
            side="neutral",
            question="Detect thesis drift: what has changed since the last analysis?",
            rationale="Prior analysis exists. Identify whether assumptions still hold.",
            evidence_memory_ids=[m.id for m in memories[:5]],
        ))

    return topics


# ── Outcome management ───────────────────────────────────────────────────────

async def update_validity_status(memory_ids: list[str], status: str) -> int:
    """Update metadata->>'validity_status' for a batch of memories.

    Args:
        memory_ids: UUIDs to update.
        status:     Stored pillar lifecycle status.

    Returns:
        Number of rows updated.
    """
    if not memory_ids:
        return 0

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE investment_memories
                SET metadata = jsonb_set(
                    COALESCE(metadata, '{}'::jsonb),
                    '{validity_status}',
                    %s::jsonb
                ),
                    updated_at = NOW()
                WHERE id = ANY(%s::uuid[])
                """,
                (f'"{status}"', memory_ids),
            )
            updated = cur.rowcount
    try:
        for memory_id in memory_ids:
            await update_pillar_lifecycle(memory_id=memory_id, status=status)
    except Exception as e:
        logger.debug("[VectorMemory] pillar lifecycle status side-update skipped: %s", e)
    return updated


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

    clauses = [
        "ticker = %s",
        "(is_cited = false OR metadata->>'validity_status' = 'stale')",
    ]
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

    for memory_id in memory_ids:
        await update_pillar_lifecycle(memory_id=memory_id, status="supported")

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
                  AND (metadata->>'source_type') IS DISTINCT FROM 'thesis_pillar'
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


# ── Pillar identity helpers ─────────────────────────────────────────────────

def normalize_pillar_statement(statement: str) -> str:
    """Normalize a pillar statement for deterministic identity hashing."""
    return re.sub(r"\s+", " ", str(statement or "").strip().lower())


def pillar_statement_hash(statement: str) -> str:
    """Stable SHA-1 hash for a normalized pillar statement."""
    return hashlib.sha1(normalize_pillar_statement(statement).encode("utf-8")).hexdigest()


def deterministic_pillar_id(ticker: str, pillar_type: str, statement: str) -> str:
    """Assign a deterministic ID for a new pillar candidate."""
    suffix = pillar_statement_hash(statement)[:10]
    safe_type = re.sub(r"[^a-z0-9_]+", "_", str(pillar_type or "pillar").lower()).strip("_")
    return f"{ticker.upper()}-{safe_type}-{suffix}"


def classify_pillar_similarity(
    similarity: float,
    *,
    high: float = HIGH_PILLAR_SIMILARITY,
    medium: float = MEDIUM_PILLAR_SIMILARITY,
) -> str:
    """Classify similarity into high/medium/low identity confidence."""
    if similarity >= high:
        return "high"
    if similarity >= medium:
        return "medium"
    return "low"


def resolve_pillar_id_from_candidates(
    *,
    ticker: str,
    pillar_type: str,
    statement: str,
    active_candidates: list[MemoryRecord],
    matched_pillar_id: str = "",
) -> tuple[str, str]:
    """Resolve high/medium/low new-vs-existing classification.

    High similarity reuses the best match.  Medium similarity only reuses an ID
    when the Judge selected a valid candidate.  Low or unresolved medium matches
    receive a deterministic new ID.
    """
    if not active_candidates:
        return deterministic_pillar_id(ticker, pillar_type, statement), "low"

    best = active_candidates[0]
    classification = classify_pillar_similarity(best.similarity or 0.0)
    candidate_ids = {
        str(candidate.metadata.get("pillar_id", ""))
        for candidate in active_candidates
        if candidate.metadata.get("pillar_id")
    }

    if classification == "high":
        return str(best.metadata.get("pillar_id")), "high"
    if classification == "medium" and matched_pillar_id in candidate_ids:
        return matched_pillar_id, "medium"
    return deterministic_pillar_id(ticker, pillar_type, statement), (
        "medium_unresolved" if classification == "medium" else "low"
    )


def _detail_citation_for_path(path: pathlib.Path) -> str:
    return f"pillars/{path.name}"


async def _get_pillar_row(pillar_id: str) -> Optional[dict]:
    """Return the identity row for a pillar, if the new table exists."""
    if not pillar_id:
        return None
    try:
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT * FROM investment_pillars WHERE pillar_id = %s",
                    (pillar_id,),
                )
                return await cur.fetchone()
    except Exception as e:
        logger.debug("[VectorMemory] investment_pillars lookup skipped: %s", e)
        return None


async def get_current_pillar_memory_id(pillar_id: str) -> str:
    """Return the current vector memory UUID for a pillar ID."""
    row = await _get_pillar_row(pillar_id)
    if row and row.get("current_memory_id"):
        return str(row["current_memory_id"])

    try:
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT id
                    FROM investment_memories
                    WHERE metadata->>'source_type' = 'thesis_pillar'
                      AND metadata->>'pillar_id' = %s
                    ORDER BY COALESCE((metadata->>'version')::int, 1) DESC, created_at DESC
                    LIMIT 1
                    """,
                    (pillar_id,),
                )
                mem = await cur.fetchone()
                return str(mem["id"]) if mem else ""
    except Exception as e:
        logger.debug("[VectorMemory] current pillar memory fallback failed: %s", e)
        return ""


async def _next_pillar_version(pillar_id: str) -> int:
    row = await _get_pillar_row(pillar_id)
    if row and row.get("version"):
        return int(row["version"]) + 1
    return 1


async def upsert_investment_pillar(
    *,
    ticker: str,
    pillar_id: str,
    pillar_type: str,
    canonical_statement: str,
    statement_hash: str,
    status: str,
    version: int,
    current_memory_id: str = "",
    detail_path: str,
    detail_citation: str,
    merged_into_pillar_id: str = "",
) -> int:
    """Upsert the queryable identity/lifecycle row for a thesis pillar."""
    try:
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO investment_pillars (
                        pillar_id, ticker, pillar_type, canonical_statement,
                        statement_hash, status, version, current_memory_id,
                        detail_path, detail_citation, merged_into_pillar_id,
                        last_seen_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NULLIF(%s, '')::uuid,
                            %s, %s, NULLIF(%s, ''), NOW())
                    ON CONFLICT (pillar_id) DO UPDATE SET
                        ticker = EXCLUDED.ticker,
                        pillar_type = EXCLUDED.pillar_type,
                        canonical_statement = EXCLUDED.canonical_statement,
                        statement_hash = EXCLUDED.statement_hash,
                        status = EXCLUDED.status,
                        version = GREATEST(investment_pillars.version, EXCLUDED.version),
                        current_memory_id = COALESCE(EXCLUDED.current_memory_id, investment_pillars.current_memory_id),
                        detail_path = EXCLUDED.detail_path,
                        detail_citation = EXCLUDED.detail_citation,
                        merged_into_pillar_id = EXCLUDED.merged_into_pillar_id,
                        updated_at = NOW(),
                        last_seen_at = NOW()
                    """,
                    (
                        pillar_id,
                        ticker.upper(),
                        pillar_type,
                        canonical_statement,
                        statement_hash,
                        status,
                        version,
                        current_memory_id,
                        detail_path,
                        detail_citation,
                        merged_into_pillar_id,
                    ),
                )
                return cur.rowcount
    except Exception as e:
        logger.warning("[VectorMemory] investment_pillars upsert skipped for %s: %s", pillar_id, e)
        return 0


async def update_pillar_lifecycle(
    *,
    pillar_id: str = "",
    memory_id: str = "",
    status: str,
    merged_into_pillar_id: str = "",
) -> int:
    """Update identity table lifecycle from a pillar ID or current memory ID."""
    if not pillar_id and not memory_id:
        return 0
    try:
        pool = await get_pool()
        clauses: list[str] = []
        where_params: list[Any] = []
        if pillar_id:
            clauses.append("pillar_id = %s")
            where_params.append(pillar_id)
        if memory_id:
            clauses.append("current_memory_id = %s::uuid")
            where_params.append(memory_id)

        where = " OR ".join(clauses)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE investment_pillars
                    SET status = %s,
                        merged_into_pillar_id = COALESCE(NULLIF(%s, ''), merged_into_pillar_id),
                        updated_at = NOW(),
                        last_seen_at = CASE
                            WHEN %s = ANY(%s::text[]) THEN NOW()
                            ELSE last_seen_at
                        END
                    WHERE {where}
                    """,
                    (
                        status,
                        merged_into_pillar_id,
                        status,
                        list(ACTIVE_PILLAR_STATUSES),
                        *where_params,
                    ),
                )
                return cur.rowcount
    except Exception as e:
        logger.debug("[VectorMemory] investment_pillars lifecycle update skipped: %s", e)
        return 0


async def find_similar_pillar_memories(
    embedding: list[float],
    *,
    ticker: str,
    pillar_type: str,
    statuses: tuple[str, ...],
    min_similarity: float,
    limit: int = 5,
) -> list[MemoryRecord]:
    """Find semantically similar thesis-pillar memories constrained by type/status."""
    results = await search_similar(
        embedding,
        ticker=ticker,
        limit=limit * 3,
        source_types=["thesis_pillar"],
        validity_statuses=list(statuses),
        min_similarity=min_similarity,
    )
    typed = [
        r for r in results
        if r.metadata.get("pillar_type") == pillar_type
        and r.metadata.get("pillar_id")
    ]
    return typed[:limit]


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


# ── Thesis pillar retrieval (replaces multi-probe paragraph retrieval) ──────

def _format_pillar_for_context(pillar: MemoryRecord, index: int) -> str:
    """Format a thesis pillar memory for the analyst RAG context."""
    ptype = pillar.metadata.get("pillar_type", "unknown")
    status = pillar.metadata.get("validity_status", "")
    badge = f" [{status}]" if status else ""
    pillar_id = pillar.metadata.get("pillar_id", "")
    version = pillar.metadata.get("version", "")
    detail_citation = pillar.metadata.get("detail_citation", "")
    source_urls = pillar.metadata.get("source_urls", [])
    evidence = pillar.metadata.get("evidence_citations", [])
    rationale = pillar.metadata.get("rationale", "")
    valuation_impact = pillar.metadata.get("valuation_impact", "")

    lines = [
        f"### Prior Ref P{index}: {ptype}{badge}",
        f"**Memory ID**: {pillar.id}",
        f"**Pillar ID**: {pillar_id or 'unassigned'}",
        f"**Version**: {version or 'unknown'}",
        f"**Claim**: {pillar.summary}",
    ]
    if detail_citation:
        lines.append(f"**Dossier**: {detail_citation}")
    if rationale:
        lines.append(f"**Rationale**: {rationale}")
    if valuation_impact:
        lines.append(f"**Valuation Impact**: {valuation_impact}")
    if source_urls:
        urls = ", ".join(source_urls[:3])
        lines.append(f"**Source URLs**: {urls}")
    if evidence:
        citations = ", ".join(evidence[:3])
        lines.append(f"**Vault Evidence**: {citations}")
    return "\n".join(lines)


async def retrieve_active_pillars(
    ticker: str,
    *,
    limit: int = 12,
) -> dict:
    """Retrieve active thesis pillar memories for a ticker.

    Only returns pillars with ``validity_status`` of ``"supported"`` or
    ``"weakened"``.  Inactive pillars are excluded from normal
    daily retrieval.

    Returns a dict with keys: memory_context, retrieved_memory_ids, topics, error.
    """
    pool = await get_pool()

    try:
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT m.*
                    FROM investment_pillars p
                    JOIN investment_memories m ON m.id = p.current_memory_id
                    WHERE p.ticker = %s
                      AND p.status = ANY(%s::text[])
                    ORDER BY p.last_seen_at DESC, p.updated_at DESC
                    LIMIT %s
                    """,
                    (ticker.upper(), list(ACTIVE_PILLAR_STATUSES), limit),
                )
                rows = await cur.fetchall()
    except Exception as e:
        logger.debug("[VectorMemory] Falling back to metadata pillar retrieval: %s", e)
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT * FROM investment_memories
                    WHERE ticker = %s
                      AND metadata->>'source_type' = 'thesis_pillar'
                      AND metadata->>'validity_status' = ANY(%s::text[])
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (ticker.upper(), list(ACTIVE_PILLAR_STATUSES), limit),
                )
                rows = await cur.fetchall()

    pillars = [_row_to_memory(row) for row in rows]

    if not pillars:
        return {
            "memory_context": "",
            "retrieved_memory_ids": [],
            "topics": [],
            "error": None,
        }

    # Build memory_context prose
    context_lines = [
        f"## Active Investment Thesis Pillars for {ticker.upper()}",
        (
            "The following are curated thesis pillars from prior analyses. "
            "Each pillar represents an enduring claim about the business. "
            "Your job: research each pillar against fresh sources and provide "
            "evidence that supports, weakens, contradicts, or revises it.\n"
        ),
    ]
    for idx, p in enumerate(pillars, 1):
        context_lines.append(_format_pillar_for_context(p, idx))
        context_lines.append("")

    memory_context = "\n".join(context_lines)

    # Build research topics from pillars
    topics = _build_pillar_topics(pillars)

    return {
        "memory_context": memory_context,
        "retrieved_memory_ids": [p.id for p in pillars],
        "topics": [t.model_dump(mode="json") for t in topics],
        "error": None,
    }


def _build_pillar_topics(pillars: list[MemoryRecord]) -> list:
    """Derive ResearchTopics from active pillars, assigning sides by pillar_type."""
    from schemas.rag import ResearchTopic

    topics: list[ResearchTopic] = []

    # Group pillars into bull-side, bear-side, and neutral
    bull_pillars = [
        p for p in pillars
        if p.metadata.get("pillar_type") in ("moat", "growth", "capital_allocation")
    ]
    bear_pillars = [
        p for p in pillars
        if p.metadata.get("pillar_type") in ("risk",)
    ]
    neutral_pillars = [
        p for p in pillars
        if p.metadata.get("pillar_type") in ("valuation_assumption", "thesis_change")
    ]

    if bull_pillars:
        statements = "; ".join(p.summary[:120] for p in bull_pillars[:3])
        topics.append(ResearchTopic(
            side="bull",
            question=f"Research and validate bull-side thesis pillars: {statements}",
            rationale=(
                "These pillars support the investment case. Verify each claim "
                "against current sources and provide confirming or challenging evidence."
            ),
            evidence_memory_ids=[p.id for p in bull_pillars[:3]],
        ))

    if bear_pillars:
        statements = "; ".join(p.summary[:120] for p in bear_pillars[:3])
        topics.append(ResearchTopic(
            side="bear",
            question=f"Research and validate risk-side thesis pillars: {statements}",
            rationale=(
                "These pillars identify risks to the investment case. Verify "
                "whether these risks have materialized or intensified."
            ),
            evidence_memory_ids=[p.id for p in bear_pillars[:3]],
        ))

    if neutral_pillars:
        statements = "; ".join(p.summary[:120] for p in neutral_pillars[:2])
        topics.append(ResearchTopic(
            side="neutral",
            question=f"Research and validate thesis assumptions: {statements}",
            rationale=(
                "These valuation assumptions and thesis-change signals need "
                "verification against current market data and fundamentals."
            ),
            evidence_memory_ids=[p.id for p in neutral_pillars[:5]],
        ))
    elif len(pillars) >= 1 and not bull_pillars and not bear_pillars:
        # Fallback: all pillars go to neutral if none matched side buckets
        topics.append(ResearchTopic(
            side="neutral",
            question="Research and validate all active thesis pillars",
            rationale="Verify prior thesis pillars against current evidence.",
            evidence_memory_ids=[p.id for p in pillars[:5]],
        ))

    return topics


# ── Thesis pillar persistence (curator-owned) ────────────────────────────────

async def persist_thesis_pillars(
    ticker: str,
    pillars: list,
    *,
    vault_citation_prefix: str = "",
) -> list[str]:
    """Persist thesis pillars as individual vector memories.

    Each persisted pillar updates the local dossier, the ``investment_pillars``
    identity table, and, when embeddings are available, one compact vector row in
    ``investment_memories`` with:
    - ``summary`` = pillar statement
    - ``metadata.source_type`` = ``"thesis_pillar"``
    - ``metadata.validity_status`` = pillar status
    - ``metadata.pillar_type``, ``pillar_id``, ``source_urls``,
      ``evidence_citations``, ``rationale``, ``valuation_impact``,
      ``detail_path``, ``detail_citation``

    Args:
        ticker:                 Ticker symbol.
        pillars:                List of ``ThesisPillar`` instances.
        vault_citation_prefix:  Prefix for evidence_citations (e.g. vault filename).

    Returns:
        List of generated memory UUIDs.
    """
    from utils.embeddings import get_embeddings
    from utils.vault import VaultWriter

    if not pillars:
        return []

    statements = [p.statement for p in pillars]
    embeddings: list[list[float] | None]
    try:
        raw_embeddings = await get_embeddings(statements)
        if not raw_embeddings or len(raw_embeddings) != len(pillars):
            raise RuntimeError("pillar embeddings returned incomplete results")
        embeddings = list(raw_embeddings)
    except Exception as e:
        logger.warning(
            "[VectorMemory] Pillar embeddings unavailable for %s; dossiers/table rows will still be written: %s",
            ticker,
            e,
        )
        embeddings = [None for _ in pillars]

    writer = VaultWriter()
    memory_ids: list[str] = []

    for pillar, embedding in zip(pillars, embeddings):
        status = pillar.status if pillar.status in ACTIVE_PILLAR_STATUSES else "supported"
        statement_hash = pillar_statement_hash(pillar.statement)
        pillar_id = str(getattr(pillar, "matched_pillar_id", "") or "").strip()

        active_candidates: list[MemoryRecord] = []
        inactive_candidates: list[MemoryRecord] = []
        if embedding:
            active_candidates = await find_similar_pillar_memories(
                embedding,
                ticker=ticker,
                pillar_type=pillar.pillar_type,
                statuses=ACTIVE_PILLAR_STATUSES,
                min_similarity=MEDIUM_PILLAR_SIMILARITY,
            )
            if active_candidates:
                matched = str(getattr(pillar, "matched_pillar_id", "") or getattr(pillar, "pillar_id", "") or "")
                pillar_id, classification = resolve_pillar_id_from_candidates(
                    ticker=ticker,
                    pillar_type=pillar.pillar_type,
                    statement=pillar.statement,
                    active_candidates=active_candidates,
                    matched_pillar_id=matched,
                )
                if classification == "medium_unresolved":
                    logger.info(
                        "[VectorMemory] Medium pillar match for %s was not selected by Judge; assigning new deterministic ID %s",
                        ticker,
                        pillar_id,
                    )

            inactive_candidates = await find_similar_pillar_memories(
                embedding,
                ticker=ticker,
                pillar_type=pillar.pillar_type,
                statuses=INACTIVE_PILLAR_STATUSES,
                min_similarity=HIGH_PILLAR_SIMILARITY,
            )
            if inactive_candidates and not (pillar.resurrection_reason and pillar.source_urls):
                logger.warning(
                    "[VectorMemory] Blocking pillar candidate similar to inactive pillar %s without resurrection evidence",
                    inactive_candidates[0].metadata.get("pillar_id", ""),
                )
                continue

        if not pillar_id:
            pillar_id = deterministic_pillar_id(ticker, pillar.pillar_type, pillar.statement)

        existing_row = await _get_pillar_row(pillar_id)
        if existing_row and normalize_pillar_statement(existing_row.get("canonical_statement", "")) == normalize_pillar_statement(pillar.statement):
            version = int(existing_row.get("version") or 1)
        elif existing_row:
            version = int(existing_row.get("version") or 1) + 1
        else:
            version = 1

        detail_path = writer.root / ticker.upper() / "pillars" / f"{pillar_id}.md"
        detail_citation = _detail_citation_for_path(detail_path)

        metadata = {
            "source_type": "thesis_pillar",
            "pillar_type": pillar.pillar_type,
            "pillar_id": pillar_id,
            "validity_status": status,
            "rationale": pillar.rationale,
            "valuation_impact": pillar.valuation_impact,
            "source_urls": pillar.source_urls,
            "evidence_citations": pillar.evidence_citations,
            "resurrection_reason": pillar.resurrection_reason,
            "statement_hash": statement_hash,
            "version": version,
            "detail_path": str(detail_path),
            "detail_citation": detail_citation,
        }
        if vault_citation_prefix:
            metadata["vault_citation"] = vault_citation_prefix

        memory_id = ""
        if embedding:
            try:
                memory_id = await store_insight(InsightRecord(
                    ticker=ticker,
                    summary=pillar.statement,
                    embedding=embedding,
                    metadata=metadata,
                    source_priority=2,
                    is_cited=True,
                ))
                memory_ids.append(memory_id)
            except Exception as e:
                logger.warning("[VectorMemory] Pillar vector insert failed for %s: %s", pillar_id, e)

        if memory_id:
            metadata["current_memory_id"] = memory_id

        detail_path = writer.write_pillar_dossier(
            ticker=ticker,
            pillar_id=pillar_id,
            pillar_type=pillar.pillar_type,
            status=status,
            version=version,
            statement=pillar.statement,
            rationale=pillar.rationale,
            valuation_impact=pillar.valuation_impact,
            source_urls=pillar.source_urls,
            evidence_citations=pillar.evidence_citations,
            current_memory_id=memory_id,
            statement_hash=statement_hash,
            lifecycle_event=status,
            lifecycle_reason=pillar.rationale,
            resurrection_reason=pillar.resurrection_reason,
        )

        await upsert_investment_pillar(
            ticker=ticker,
            pillar_id=pillar_id,
            pillar_type=pillar.pillar_type,
            canonical_statement=pillar.statement,
            statement_hash=statement_hash,
            status=status,
            version=version,
            current_memory_id=memory_id,
            detail_path=str(detail_path),
            detail_citation=detail_citation,
        )

    logger.info(
        "[VectorMemory] 🏛️ Persisted %d thesis pillar vector memories for %s",
        len(memory_ids), ticker,
    )
    return memory_ids


async def mark_pillar_transition(
    memory_id: str,
    new_status: str,
    *,
    superseded_by: str = "",
    version: Optional[int] = None,
) -> int:
    """Mark a pillar memory inactive and optionally set superseded_by.

    Used by the Curator to handle pillar lifecycle transitions.

    Args:
        memory_id:    UUID of the pillar memory to update.
        new_status:   New validity_status (e.g. ``"superseded"``, ``"stale"``, ``"contradicted"``).
        superseded_by: UUID of the replacement pillar memory.
        version:       Optional version number to set.

    Returns:
        Number of rows updated (0 or 1).
    """
    pool = await get_pool()

    updates = {"validity_status": new_status}
    if superseded_by:
        updates["superseded_by"] = superseded_by
    if version is not None:
        updates["version"] = version

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE investment_memories
                SET metadata = metadata || %s::jsonb,
                    updated_at = NOW(),
                    is_cited = false
                WHERE id = %s::uuid
                """,
                (Jsonb(updates), memory_id),
            )
            updated = cur.rowcount
    await update_pillar_lifecycle(memory_id=memory_id, status=new_status)
    return updated


async def consolidate_near_duplicate_pillars(
    ticker: str,
    *,
    threshold: float = PILLAR_DUPLICATE_SIMILARITY,
) -> int:
    """Merge near-duplicate active pillar identities using vector self-search."""
    pool = await get_pool()
    try:
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT
                        a.id AS keeper_memory_id,
                        b.id AS duplicate_memory_id,
                        a.metadata->>'pillar_id' AS keeper_pillar_id,
                        b.metadata->>'pillar_id' AS duplicate_pillar_id,
                        b.summary AS duplicate_statement,
                        b.metadata AS duplicate_metadata,
                        1 - (a.embedding <=> b.embedding) AS similarity
                    FROM investment_memories a
                    JOIN investment_memories b ON a.id <> b.id
                    WHERE a.ticker = %s
                      AND b.ticker = %s
                      AND a.created_at <= b.created_at
                      AND a.metadata->>'source_type' = 'thesis_pillar'
                      AND b.metadata->>'source_type' = 'thesis_pillar'
                      AND a.metadata->>'pillar_type' = b.metadata->>'pillar_type'
                      AND a.metadata->>'validity_status' = ANY(%s::text[])
                      AND b.metadata->>'validity_status' = ANY(%s::text[])
                      AND COALESCE(a.metadata->>'pillar_id', '') <> ''
                      AND COALESCE(b.metadata->>'pillar_id', '') <> ''
                      AND a.metadata->>'pillar_id' <> b.metadata->>'pillar_id'
                      AND 1 - (a.embedding <=> b.embedding) >= %s
                    ORDER BY similarity DESC
                    """,
                    (
                        ticker.upper(),
                        ticker.upper(),
                        list(ACTIVE_PILLAR_STATUSES),
                        list(ACTIVE_PILLAR_STATUSES),
                        threshold,
                    ),
                )
                pairs = await cur.fetchall()
    except Exception as e:
        logger.debug("[VectorMemory] duplicate pillar consolidation skipped: %s", e)
        return 0

    merged = 0
    seen_duplicates: set[str] = set()
    from utils.vault import VaultWriter

    writer = VaultWriter()
    for row in pairs:
        duplicate_pillar_id = str(row.get("duplicate_pillar_id") or "")
        keeper_pillar_id = str(row.get("keeper_pillar_id") or "")
        duplicate_memory_id = str(row.get("duplicate_memory_id") or "")
        if not duplicate_pillar_id or not keeper_pillar_id or duplicate_pillar_id in seen_duplicates:
            continue
        seen_duplicates.add(duplicate_pillar_id)

        await mark_pillar_transition(duplicate_memory_id, "superseded", superseded_by=str(row.get("keeper_memory_id") or ""))
        await update_pillar_lifecycle(
            pillar_id=duplicate_pillar_id,
            status="superseded",
            merged_into_pillar_id=keeper_pillar_id,
        )

        metadata = row.get("duplicate_metadata") or {}
        statement = str(row.get("duplicate_statement") or "")
        writer.write_pillar_dossier(
            ticker=ticker,
            pillar_id=duplicate_pillar_id,
            pillar_type=str(metadata.get("pillar_type", "unknown")),
            status="superseded",
            version=int(metadata.get("version") or 1),
            statement=statement,
            rationale=str(metadata.get("rationale", "")),
            valuation_impact=str(metadata.get("valuation_impact", "")),
            source_urls=list(metadata.get("source_urls", [])),
            evidence_citations=list(metadata.get("evidence_citations", [])),
            current_memory_id=duplicate_memory_id,
            statement_hash=str(metadata.get("statement_hash", pillar_statement_hash(statement))),
            lifecycle_event="merged",
            lifecycle_reason=f"Near-duplicate of {keeper_pillar_id} at similarity {float(row.get('similarity') or 0.0):.2f}.",
            merged_into_pillar_id=keeper_pillar_id,
        )
        merged += 1

    if merged:
        logger.info("[VectorMemory] 🔁 Merged %d near-duplicate pillars for %s", merged, ticker)
    return merged


async def append_pillar_dossier_event(
    memory_id: str,
    *,
    status: str,
    lifecycle_event: str,
    lifecycle_reason: str = "",
    source_urls: Optional[list[str]] = None,
) -> None:
    """Append a lifecycle event to the local dossier for an existing memory row."""
    if not memory_id:
        return
    try:
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT * FROM investment_memories WHERE id = %s::uuid",
                    (memory_id,),
                )
                row = await cur.fetchone()
        if not row:
            return
        memory = _row_to_memory(row)
        metadata = memory.metadata or {}
        pillar_id = str(metadata.get("pillar_id") or "")
        if not pillar_id:
            return
        from utils.vault import VaultWriter

        urls = source_urls if source_urls is not None else list(metadata.get("source_urls", []))
        writer = VaultWriter()
        detail_path = writer.write_pillar_dossier(
            ticker=memory.ticker,
            pillar_id=pillar_id,
            pillar_type=str(metadata.get("pillar_type", "unknown")),
            status=status,
            version=int(metadata.get("version") or 1),
            statement=memory.summary,
            rationale=str(metadata.get("rationale", "")),
            valuation_impact=str(metadata.get("valuation_impact", "")),
            source_urls=urls,
            evidence_citations=list(metadata.get("evidence_citations", [])),
            current_memory_id=memory.id,
            statement_hash=str(metadata.get("statement_hash", pillar_statement_hash(memory.summary))),
            lifecycle_event=lifecycle_event,
            lifecycle_reason=lifecycle_reason,
            resurrection_reason=str(metadata.get("resurrection_reason", "")),
        )
        await upsert_investment_pillar(
            ticker=memory.ticker,
            pillar_id=pillar_id,
            pillar_type=str(metadata.get("pillar_type", "unknown")),
            canonical_statement=memory.summary,
            statement_hash=str(metadata.get("statement_hash", pillar_statement_hash(memory.summary))),
            status=status,
            version=int(metadata.get("version") or 1),
            current_memory_id=memory.id,
            detail_path=str(detail_path),
            detail_citation=_detail_citation_for_path(detail_path),
        )
    except Exception as e:
        logger.debug("[VectorMemory] dossier event append skipped for %s: %s", memory_id, e)


async def find_near_duplicate_summary_pairs(
    ticker: str,
    *,
    threshold: float = SUMMARY_DUPLICATE_SIMILARITY,
) -> list[dict]:
    """Return near-duplicate monthly/historical summary vector pairs."""
    pool = await get_pool()
    try:
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT
                        a.id AS keeper_memory_id,
                        b.id AS duplicate_memory_id,
                        a.summary AS keeper_summary,
                        b.summary AS duplicate_summary,
                        a.metadata AS keeper_metadata,
                        b.metadata AS duplicate_metadata,
                        1 - (a.embedding <=> b.embedding) AS similarity
                    FROM investment_memories a
                    JOIN investment_memories b ON a.id <> b.id
                    WHERE a.ticker = %s
                      AND b.ticker = %s
                      AND a.created_at <= b.created_at
                      AND COALESCE(a.metadata->>'type', a.metadata->>'source_type') IN ('monthly_summary', 'historical_summary')
                      AND COALESCE(b.metadata->>'type', b.metadata->>'source_type') IN ('monthly_summary', 'historical_summary')
                      AND 1 - (a.embedding <=> b.embedding) >= %s
                    ORDER BY similarity DESC
                    """,
                    (ticker.upper(), ticker.upper(), threshold),
                )
                return [dict(row) for row in await cur.fetchall()]
    except Exception as e:
        logger.debug("[VectorMemory] duplicate summary scan skipped: %s", e)
        return []
