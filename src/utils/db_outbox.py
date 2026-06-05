"""
db_outbox.py
~~~~~~~~~~~~
Local JSONL outbox for retrying transient database create/update failures.

The outbox is intentionally small and append-only during normal execution.  A
startup drain replays entries FIFO and rewrites the file with any entries that
still could not be persisted.
"""

from __future__ import annotations

import json
import os
import pathlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from psycopg.types.json import Jsonb

from utils.db import get_pool, is_transient_db_error
from utils.logger import get_logger

logger = get_logger(__name__)

OUTBOX_PATH = pathlib.Path(os.environ.get("DB_OUTBOX_PATH", "data/outbox/db_cud_outbox.jsonl"))
BAD_OUTBOX_PATH = pathlib.Path(
    os.environ.get("DB_OUTBOX_BAD_PATH", "data/outbox/db_cud_outbox.bad.jsonl")
)


@dataclass(frozen=True)
class OutboxDrainResult:
    total: int
    succeeded: int
    retained: int
    corrupted: int


async def append_db_outbox_entry(
    *,
    operation: str,
    table: str,
    ticker: str,
    payload: dict[str, Any],
    error: str,
    attempts: int = 0,
    path: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Append a replayable create/update operation to the local JSONL outbox."""
    entry = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "table": table,
        "ticker": str(ticker or "").upper(),
        "payload": payload,
        "error": error,
        "attempts": attempts,
    }
    outbox_path = path or OUTBOX_PATH
    outbox_path.parent.mkdir(parents=True, exist_ok=True)
    with outbox_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, default=str))
        f.write("\n")
    return entry


async def drain_db_outbox(
    *,
    path: pathlib.Path | None = None,
    bad_path: pathlib.Path | None = None,
    replay_fn: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> OutboxDrainResult:
    """Replay pending outbox entries FIFO without blocking fresh agent runs."""
    outbox_path = path or OUTBOX_PATH
    bad_outbox_path = bad_path or BAD_OUTBOX_PATH
    entries, bad_lines = _read_outbox_entries(outbox_path)
    if bad_lines:
        _append_bad_lines(bad_outbox_path, bad_lines)

    if not entries:
        if bad_lines:
            _rewrite_outbox(outbox_path, [])
        return OutboxDrainResult(total=0, succeeded=0, retained=0, corrupted=len(bad_lines))

    replay = replay_fn or replay_outbox_entry
    retained: list[dict[str, Any]] = []
    succeeded = 0

    logger.info("[DBOutbox] Draining %d pending DB outbox entr%s", len(entries), "y" if len(entries) == 1 else "ies")
    for entry in entries:
        try:
            await replay(entry)
            succeeded += 1
        except Exception as e:
            updated = dict(entry)
            updated["attempts"] = int(updated.get("attempts") or 0) + 1
            updated["error"] = str(e)
            retained.append(updated)
            if is_transient_db_error(e):
                logger.warning(
                    "[DBOutbox] Replay still transient for %s/%s (%s); keeping entry %s",
                    entry.get("table"),
                    entry.get("operation"),
                    e,
                    entry.get("id"),
                )
            else:
                logger.warning(
                    "[DBOutbox] Replay failed with non-transient error for %s/%s (%s); keeping entry %s",
                    entry.get("table"),
                    entry.get("operation"),
                    e,
                    entry.get("id"),
                )

    _rewrite_outbox(outbox_path, retained)
    logger.info(
        "[DBOutbox] Drain complete: %d succeeded, %d retained, %d corrupt",
        succeeded,
        len(retained),
        len(bad_lines),
    )
    return OutboxDrainResult(
        total=len(entries),
        succeeded=succeeded,
        retained=len(retained),
        corrupted=len(bad_lines),
    )


async def replay_outbox_entry(entry: dict[str, Any]) -> None:
    table = str(entry.get("table", ""))
    operation = str(entry.get("operation", ""))
    payload = entry.get("payload") or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Outbox entry {entry.get('id')} has invalid payload")

    if table == "investment_memories" and operation == "insert":
        await _insert_replay_memory(payload)
    elif table == "investment_memories" and operation == "update_validity_status":
        await _replay_memory_update_validity_status(payload)
    elif table == "investment_memories" and operation == "mark_cited":
        await _replay_memory_mark_cited(payload)
    elif table == "investment_memories" and operation == "mark_pillar_transition":
        await _replay_memory_mark_pillar_transition(payload)
    elif table == "investment_pillars" and operation == "upsert":
        await _replay_pillar_upsert(payload)
    elif table == "investment_pillars" and operation == "update_lifecycle":
        await _replay_pillar_lifecycle(payload)
    elif table == "valuations" and operation == "upsert":
        from utils.db import _execute_valuation_upsert

        await _execute_valuation_upsert(payload)
    elif table == "thesis_drifts" and operation == "insert":
        await _replay_thesis_drift_insert(payload)
    else:
        raise ValueError(f"Unsupported outbox operation: {table}/{operation}")


def _read_outbox_entries(path: pathlib.Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], []

    entries: list[dict[str, Any]] = []
    bad_lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise ValueError("outbox line is not an object")
                entries.append(parsed)
            except Exception:
                bad_lines.append(line)
    return entries, bad_lines


def _append_bad_lines(path: pathlib.Path, lines: list[str]) -> None:
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def _rewrite_outbox(path: pathlib.Path, entries: list[dict[str, Any]]) -> None:
    if not entries:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True, default=str))
            f.write("\n")
    tmp_path.replace(path)


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(str(v) for v in vec) + "]"


async def _insert_replay_memory(payload: dict[str, Any]) -> None:
    records = list(payload.get("records") or [])
    if not records:
        return
    placeholders: list[str] = []
    params: list[Any] = []
    for record in records:
        placeholders.append("(%s, %s::vector, %s, %s, %s, %s, %s)")
        params.extend(
            [
                record["id"],
                _vec_literal(record["embedding"]),
                record["summary"],
                Jsonb(record.get("metadata") or {}),
                str(record["ticker"]).upper(),
                record.get("source_priority", 0),
                record.get("is_cited", False),
            ]
        )

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                INSERT INTO investment_memories
                    (id, embedding, summary, metadata, ticker, source_priority, is_cited)
                VALUES {', '.join(placeholders)}
                ON CONFLICT (id) DO NOTHING
                """,
                params,
            )


async def _replay_memory_update_validity_status(payload: dict[str, Any]) -> None:
    memory_ids = list(payload.get("memory_ids") or [])
    if not memory_ids:
        return
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
                (f'"{payload["status"]}"', memory_ids),
            )


async def _replay_memory_mark_cited(payload: dict[str, Any]) -> None:
    memory_ids = list(payload.get("memory_ids") or [])
    if not memory_ids:
        return
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE investment_memories
                SET is_cited = true, updated_at = NOW()
                WHERE id = ANY(%s::uuid[])
                """,
                (memory_ids,),
            )


async def _replay_memory_mark_pillar_transition(payload: dict[str, Any]) -> None:
    updates = dict(payload.get("updates") or {})
    pool = await get_pool()
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
                (Jsonb(updates), payload["memory_id"]),
            )


async def _replay_pillar_upsert(payload: dict[str, Any]) -> None:
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
                    payload["pillar_id"],
                    str(payload["ticker"]).upper(),
                    payload["pillar_type"],
                    payload["canonical_statement"],
                    payload["statement_hash"],
                    payload["status"],
                    payload["version"],
                    payload.get("current_memory_id", ""),
                    payload["detail_path"],
                    payload["detail_citation"],
                    payload.get("merged_into_pillar_id", ""),
                ),
            )


async def _replay_pillar_lifecycle(payload: dict[str, Any]) -> None:
    clauses: list[str] = []
    where_params: list[Any] = []
    if payload.get("pillar_id"):
        clauses.append("pillar_id = %s")
        where_params.append(payload["pillar_id"])
    if payload.get("memory_id"):
        clauses.append("current_memory_id = %s::uuid")
        where_params.append(payload["memory_id"])
    if not clauses:
        return

    pool = await get_pool()
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
                WHERE {' OR '.join(clauses)}
                """,
                (
                    payload["status"],
                    payload.get("merged_into_pillar_id", ""),
                    payload["status"],
                    ["supported", "weakened"],
                    *where_params,
                ),
            )


async def _replay_thesis_drift_insert(payload: dict[str, Any]) -> None:
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
                    str(payload["ticker"]).upper(),
                    payload.get("old_verdict"),
                    payload.get("new_verdict"),
                    payload.get("old_expected_value"),
                    payload.get("new_expected_value"),
                    payload.get("delta_pct"),
                    Jsonb(payload.get("key_changes") or []),
                ),
            )
