"""
curator.py
~~~~~~~~~~
The Curator Agent — the final "Knowledge Janitor" in the LangGraph workflow.

Runs after Report Generator. Responsibilities:
1. Extract citation manifest from the final report
2. Mark cited memories in Supabase (is_cited = true)
3. Delete uncited / contradicted memories for the ticker
4. Run local vault deduplication
5. Consolidate old daily vault files into monthly syntheses
6. Check Supabase storage health (DB_LIMIT_BYTES guardrail at AGGRESSIVE_THRESHOLD)
7. Log all maintenance actions
"""

from __future__ import annotations

import pathlib
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Literal

from langgraph.types import Command

from agents.states import WorkflowState
from schemas import AgentNode
from utils.citations import build_citation_manifest
from utils.deduplication import deduplicate
from utils.drift_tracker import record_drift
from utils.embeddings import get_embedding
from utils.vault import VaultReader, VaultWriter, VAULT_ROOT
from utils.logger import get_logger, log_node_execution
from utils.config import curator_model, config
from utils import vector_memory

logger = get_logger(__name__)

# Read thresholds from the shared config so they can be overridden via env vars.
DB_LIMIT_BYTES = config.curator.db_limit_mb * 1024 * 1024
AGGRESSIVE_THRESHOLD = config.curator.aggressive_threshold
CONSOLIDATION_CUTOFF_DAYS = config.curator.consolidation_cutoff_days


@log_node_execution
async def curator_agent(state: WorkflowState) -> Command[Literal["__end__"]]:
    """Post-report maintenance node.

    Manages thesis pillar lifecycle:
    1. Processes pillar_outcomes from the Judge — marks prior pillars as
       supported / weakened / revised / contradicted / stale.
    2. Persists new or revised thesis_pillars as active vector memories.
    3. Runs vault deduplication, monthly consolidation, and storage checks.
    """
    ticker = state["ticker"].upper()
    logs: list[str] = []

    def _log(msg: str) -> None:
        logger.info(f"[Curator] {msg}")
        logs.append(msg)

    _log(f"🧹 Starting maintenance for {ticker}")

    # ── 1. Persist new/revised thesis pillars ───────────────────────────
    pillar_outcomes = state.get("pillar_outcomes", [])
    thesis_pillars = state.get("thesis_pillars", [])
    _log(f"🏛️ Received {len(thesis_pillars)} thesis pillars + {len(pillar_outcomes)} outcomes")

    outcome_by_pillar = {
        str(o.get("pillar_id", "")): str(o.get("status", ""))
        for o in pillar_outcomes
        if o.get("pillar_id")
    }

    new_pillar_ids: list[str] = []
    if thesis_pillars:
        from schemas.rag import ThesisPillar
        pillars: list[ThesisPillar] = []
        for p_dict in thesis_pillars:
            try:
                pillar = ThesisPillar(**p_dict)
            except Exception as e:
                _log(f"⚠️ Skipping malformed pillar: {e}")
                continue

            matched_id = pillar.matched_pillar_id or pillar.pillar_id
            outcome_status = outcome_by_pillar.get(matched_id, "")
            should_persist = (
                not matched_id
                or outcome_status in ("revised", "weakened")
                or pillar.status == "weakened"
            )
            if should_persist:
                pillars.append(pillar)

        if pillars:
            new_pillar_ids = await vector_memory.persist_thesis_pillars(
                ticker=ticker,
                pillars=pillars,
            )
            _log(f"🏛️ Persisted {len(new_pillar_ids)} new/revised thesis pillar memories")

    # ── 2. Process pillar outcomes from the Judge ───────────────────────
    if pillar_outcomes:
        supported_count = 0
        weakened_count = 0
        revised_count = 0
        demoted_count = 0

        for outcome_dict in pillar_outcomes:
            memory_id = outcome_dict.get("memory_id", "")
            status = outcome_dict.get("status", "")
            if status == "updated":
                status = "revised"
            pillar_id = outcome_dict.get("pillar_id", "")
            reason = outcome_dict.get("reason", "")
            source_urls = outcome_dict.get("source_urls", [])

            if not memory_id:
                continue

            if status == "supported":
                await vector_memory.mark_memories_cited([memory_id])
                await vector_memory.append_pillar_dossier_event(
                    memory_id,
                    status="supported",
                    lifecycle_event="supported",
                    lifecycle_reason=reason,
                    source_urls=source_urls,
                )
                supported_count += 1

            elif status == "weakened":
                await vector_memory.update_validity_status([memory_id], "weakened")
                await vector_memory.append_pillar_dossier_event(
                    memory_id,
                    status="weakened",
                    lifecycle_event="weakened",
                    lifecycle_reason=reason,
                    source_urls=source_urls,
                )
                weakened_count += 1

            elif status == "revised":
                replacement_id = await vector_memory.get_current_pillar_memory_id(pillar_id)
                if replacement_id and replacement_id != memory_id:
                    await vector_memory.mark_pillar_transition(
                        memory_id,
                        "superseded",
                        superseded_by=replacement_id,
                    )
                    event_status = "superseded"
                else:
                    _log(f"⚠️ Revised pillar {pillar_id} has no replacement vector yet; keeping prior memory active")
                    event_status = "supported"
                await vector_memory.append_pillar_dossier_event(
                    memory_id,
                    status=event_status,
                    lifecycle_event="revised",
                    lifecycle_reason=reason,
                    source_urls=source_urls,
                )
                revised_count += 1

            elif status in ("contradicted", "stale"):
                await vector_memory.update_validity_status(
                    [memory_id], status
                )
                await vector_memory.append_pillar_dossier_event(
                    memory_id,
                    status=status,
                    lifecycle_event=status,
                    lifecycle_reason=reason,
                    source_urls=source_urls,
                )
                demoted_count += 1
            else:
                logger.warning(f"[Curator] Unknown pillar outcome status: {status}")
                _log(f"⚠️ Unknown pillar outcome status: {status}")

        _log(
            f"🏛️ Pillar outcomes: {supported_count} supported, "
            f"{weakened_count} weakened, {revised_count} revised, "
            f"{demoted_count} contradicted/stale"
        )

    try:
        merged_pillars = await vector_memory.consolidate_near_duplicate_pillars(ticker)
        if merged_pillars:
            _log(f"🔁 Consolidated {merged_pillars} near-duplicate pillar(s)")
    except Exception as e:
        _log(f"⚠️ Pillar duplicate consolidation skipped: {e}")

    # ── 3. Prune non-pillar memories (legacy paragraph memories) ────────
    #    Only pillar memories participate in normal retrieval. Legacy
    #    paragraph memories past the consolidation window are cleaned up.
    cutoff_date = (
        datetime.now(timezone.utc) - timedelta(days=CONSOLIDATION_CUTOFF_DAYS)
    ).isoformat()
    try:
        import psycopg.rows
        pool = await vector_memory.get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    DELETE FROM investment_memories
                    WHERE ticker = %s
                      AND created_at < %s
                      AND (metadata->>'source_type' IS NULL
                           OR metadata->>'source_type' != 'thesis_pillar')
                      AND is_cited = false
                    """,
                    (ticker, cutoff_date),
                )
                pruned = cur.rowcount
                if pruned:
                    _log(f"🗑️ Pruned {pruned} uncited non-pillar memories older than cutoff")
    except Exception as e:
        _log(f"⚠️ Non-pillar prune skipped: {e}")

    # ── 4. Thesis drift tracking ────────────────────────────────────────
    valuation = state.get("valuation")
    if valuation:
        if valuation.thesis_data:
            prior_decision = valuation.thesis_data.get("final_decision", "")
        else:
            prior_decision = ""

        new_decision = state.get("judge_decision", "")

        old_verdict = _extract_verdict(prior_decision)
        new_verdict = _extract_verdict(new_decision)

        if old_verdict or new_verdict:
            old_ev = valuation.expected_value if valuation else None
            new_ev = valuation.expected_value
            await record_drift(
                ticker=ticker,
                old_verdict=old_verdict or "N/A",
                new_verdict=new_verdict or "N/A",
                old_expected_value=old_ev,
                new_expected_value=new_ev,
                key_changes=_extract_key_changes(new_decision),
            )
            _log(f"📊 Drift recorded: {old_verdict} → {new_verdict}")

    # ── 5. Local vault deduplication ────────────────────────────────────
    dedup_report = deduplicate(ticker, dry_run=False)
    if dedup_report.files_removed > 0:
        _log(
            f"📁 Deduplication: removed {dedup_report.files_removed} "
            f"duplicate(s) from vault"
        )
    else:
        _log("📁 No duplicates found in vault")

    # ── 6. Monthly consolidation ────────────────────────────────────────
    consolidated = await _consolidate_monthly(ticker)
    if consolidated:
        _log(f"📦 Consolidated {consolidated} file(s) into monthly syntheses")

    summary_deduped = await _consolidate_duplicate_summaries(ticker)
    if summary_deduped:
        _log(f"📚 Consolidated {summary_deduped} near-duplicate summary vector(s)")

    # ── 7. Storage health check (500MB guardrail) ───────────────────────
    try:
        table_size = await vector_memory.get_table_size_bytes()
        size_mb = table_size / (1024 * 1024)
        usage_pct = table_size / DB_LIMIT_BYTES
        _log(f"💾 Storage: {size_mb:.1f}MB / {config.curator.db_limit_mb}MB ({usage_pct:.1%})")

        if usage_pct >= AGGRESSIVE_THRESHOLD:
            _log(f"⚠️ Storage above {AGGRESSIVE_THRESHOLD*100}% — triggering aggressive pruning")
            aggressive_deleted = await _aggressive_prune()
            _log(f"🔥 Aggressive pruning removed {aggressive_deleted} vectors")
    except Exception as e:
        _log(f"⚠️ Storage check failed: {e}")

    _log(f"✨ Maintenance complete for {ticker}")

    curator_log = "\n".join(logs)
    return Command(
        update={"curator_log": curator_log},
        goto="__end__",
    )


# ── Internal helpers ─────────────────────────────────────────────────────────

def _memory_ids_for_citations(manifest: list, vault_artifacts: list[dict]) -> list[str]:
    """Resolve report citation refs to vector memory IDs from vault artifacts."""
    by_file_and_block: dict[tuple[str, str], str] = {}

    for artifact in vault_artifacts:
        block_memory_ids = artifact.get("block_memory_ids") or {}
        filename = artifact.get("filename")
        path = artifact.get("path")
        if not filename and path:
            filename = pathlib.Path(str(path)).name

        for block_id, memory_id in block_memory_ids.items():
            if not memory_id:
                continue
            if filename:
                by_file_and_block[(str(filename), str(block_id))] = str(memory_id)
            if path:
                by_file_and_block[(str(path), str(block_id))] = str(memory_id)

    cited_ids: list[str] = []
    seen: set[str] = set()
    for ref in manifest:
        file_path = getattr(ref, "file_path", None)
        block_id = getattr(ref, "block_id", None)
        if isinstance(ref, dict):
            file_path = ref.get("file_path", file_path)
            block_id = ref.get("block_id", block_id)
        if not file_path or not block_id:
            continue

        filename = pathlib.Path(str(file_path)).name
        memory_id = (
            by_file_and_block.get((str(file_path), str(block_id)))
            or by_file_and_block.get((filename, str(block_id)))
        )
        if memory_id and memory_id not in seen:
            cited_ids.append(memory_id)
            seen.add(memory_id)

    return cited_ids


def _extract_verdict(decision_text: str) -> str:
    """Extract the verdict line from a judge decision.

    The judge output typically starts with:
    **Verdict** — <one sentence>
    or
    1. **Verdict** — <one sentence>
    """
    if not decision_text:
        return ""

    for line in decision_text.split("\n"):
        line = line.strip()
        if "verdict" in line.lower():
            # Strip markdown formatting
            clean = line.replace("**", "").replace("*", "")
            # Remove leading numbers/bullets
            clean = clean.lstrip("0123456789.-) ").strip()
            # Remove the "Verdict" label itself
            if "—" in clean:
                clean = clean.split("—", 1)[1].strip()
            elif ":" in clean:
                clean = clean.split(":", 1)[1].strip()
            return clean[:200]  # Cap length

    # Fallback: return first non-empty line
    for line in decision_text.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return ""


def _extract_key_changes(decision_text: str) -> list[str]:
    """Extract key risk/change bullet points from the judge decision."""
    changes: list[str] = []
    in_risks = False

    for line in decision_text.split("\n"):
        stripped = line.strip()

        # Detect "Key Risks" or similar section headers
        if any(kw in stripped.lower() for kw in ["key risk", "key change", "risk factor"]):
            in_risks = True
            continue

        # Detect next section header (exit)
        if in_risks and stripped.startswith("#"):
            break

        # Collect bullet points
        if in_risks and stripped.startswith(("-", "•", "*")):
            clean = stripped.lstrip("-•* ").strip()
            if clean:
                changes.append(clean[:200])

    return changes[:10]  # Cap at 10 items


async def _consolidate_monthly(ticker: str) -> int:
    """Synthesise daily vault files older than CUTOFF days into monthly digests.

    Operates on **both** storage layers atomically:
    - **Vault files**: Daily .md files are LLM-synthesised into a single
      ``{YYYY-MM}_synthesis.md`` file.  If one already exists, new files
      are incrementally incorporated.
    - **Vectors**: Granular vectors for the consolidated period are deleted
      and replaced with a single summary vector embedded from the synthesis.

    Original files are archived (``archived: true`` in frontmatter) but NOT
    deleted, so first-party data can be recovered if needed.

    Returns the number of source files consolidated.
    """
    reader = VaultReader()
    writer = VaultWriter()
    ticker = ticker.upper()
    ticker_dir = VAULT_ROOT / ticker

    if not ticker_dir.exists():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=CONSOLIDATION_CUTOFF_DAYS)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    # Group NON-ARCHIVED, non-synthesis files older than cutoff by month
    monthly_groups: dict[str, list[pathlib.Path]] = defaultdict(list)

    for md_file in sorted(ticker_dir.glob("*.md")):
        # Skip existing synthesis files and already-archived files
        if "_synthesis" in md_file.name:
            continue

        # Check if already archived
        try:
            doc = reader.read_document(md_file)
            if doc.archived:
                continue
        except Exception:
            continue

        file_date = md_file.stem[:10]  # YYYY-MM-DD
        if file_date >= cutoff_str:
            continue

        month_key = file_date[:7]  # YYYY-MM
        monthly_groups[month_key].append(md_file)

    consolidated = 0
    for month, new_files in monthly_groups.items():
        if not new_files:
            continue

        synthesis_path = ticker_dir / f"{month}_synthesis.md"

        # ── Gather LLM context ─────────────────────────────────────────
        raw_parts: list[str] = []
        is_incremental = False

        # If synthesis already exists, use it as the starting context
        if synthesis_path.exists():
            try:
                existing_doc = reader.read_document(synthesis_path)
                raw_parts.append(
                    f"### EXISTING MONTHLY SYNTHESIS\n{existing_doc.content}"
                )
                is_incremental = True
            except Exception:
                pass

        # Add the new daily files
        for f in new_files:
            try:
                doc = reader.read_document(f)
                header = f"### {doc.source_type} — {doc.date_scraped}"
                if doc.url:
                    header += f" ({doc.url})"
                raw_parts.append(f"{header}\n{doc.content}")
            except Exception as e:
                logger.warning(f"[Curator] Failed to read {f}: {e}")

        if not raw_parts:
            continue

        raw_dump = "\n\n---\n\n".join(raw_parts)

        # ── LLM Synthesis ───────────────────────────────────────────────
        if is_incremental:
            prompt_context = (
                f"You are updating an existing monthly synthesis for {ticker} ({month}). "
                f"The existing synthesis and {len(new_files)} new research document(s) "
                f"are provided below.  Produce an UPDATED synthesis that incorporates "
                f"the new insights."
            )
        else:
            prompt_context = (
                f"You are creating a monthly synthesis for {ticker} ({month}) "
                f"from {len(new_files)} research document(s)."
            )

        synthesis_prompt = (
            f"{prompt_context}\n\n"
            f"Distil them into a structured monthly synthesis using this format:\n\n"
            f"## {ticker} — {month} Monthly Synthesis\n\n"
            f"### Key Takeaways\n"
            f"- 3-5 bullet points of the most important insights\n\n"
            f"### Financial Trends\n"
            f"- Revenue, margins, cash flow, or valuation changes observed\n\n"
            f"### Risk Factors\n"
            f"- Material risks or red flags identified during this period\n\n"
            f"### Notable Events\n"
            f"- Earnings, filings, management changes, macro shifts\n\n"
            f"Rules:\n"
            f"- Be concise but preserve numerical specifics (exact figures, percentages, dates).\n"
            f"- Discard generic market commentary — keep only high-signal insights.\n"
            f"- Do NOT invent data that is not present in the source material.\n\n"
            f"--- RAW DATA ---\n\n{raw_dump}"
        )

        try:
            response = await curator_model.ainvoke(synthesis_prompt)
            synthesis_content = str(response.content)
        except Exception as e:
            logger.warning(
                f"[Curator] LLM synthesis failed for {ticker}/{month}, "
                f"falling back to concatenation: {e}"
            )
            synthesis_content = raw_dump

        # ── Write synthesis file (creates or overwrites) ───────────────────
        writer.write_synthesis(ticker=ticker, month=month, content=synthesis_content)

        # ── Vector lifecycle: clean up granular vectors, create summary ────
        month_start = f"{month}-01"
        year, mon = int(month[:4]), int(month[5:7])
        month_end = (
            f"{year + 1}-01-01" if mon == 12 else f"{year}-{mon + 1:02d}-01"
        )

        deleted_vectors = await vector_memory.delete_memories_for_period(
            ticker, month_start, month_end
        )

        try:
            embedding = await get_embedding(synthesis_content)

            # Read back synthesis to capture citation metadata
            citation_extra: dict = {}
            try:
                synth_doc = reader.read_document(synthesis_path)
                citation_extra["filename"] = synthesis_path.name
                citation_extra["local_path"] = str(synthesis_path)
                if synth_doc.block_map:
                    first_bid = next(iter(synth_doc.block_map))
                    citation_extra["block_id"] = first_bid
                    citation_extra["citation"] = (
                        f"{synthesis_path.name}#^{first_bid}"
                    )
            except Exception:
                pass  # Best-effort — citation metadata is optional

            await vector_memory.upsert_summary_vector(
                ticker=ticker,
                month=month,
                summary_text=synthesis_content,
                embedding=embedding,
                extra_metadata={
                    "source_files_count": len(new_files),
                    "vectors_consolidated": deleted_vectors,
                    "incremental": is_incremental,
                    **citation_extra,
                },
            )
        except Exception as e:
            logger.warning(
                f"[Curator] Failed to create summary vector for {ticker}/{month}: {e}"
            )

        # ── Archive source files ───────────────────────────────────────
        for f in new_files:
            try:
                reader.mark_archived(f)
            except Exception as e:
                logger.warning(f"[Curator] Failed to archive {f}: {e}")

        consolidated += len(new_files)
        mode = "Updated" if is_incremental else "Created"
        logger.info(
            f"[Curator] 📦 {mode} synthesis for {ticker}/{month} "
            f"from {len(new_files)} file(s), replaced {deleted_vectors} vector(s)"
        )

    return consolidated


async def _aggressive_prune() -> int:
    """Emergency pruning — merge monthly summary vectors into historical summaries.

    Only fires when storage exceeds the configured threshold.  By the time
    this runs, most data should already be in monthly-summary form thanks
    to ``_consolidate_monthly``.  This function provides a final compression
    layer by merging the oldest monthly summaries into a single historical
    summary per ticker, keeping only the 3 most recent months intact.
    """
    from psycopg.rows import dict_row
    from utils.db import get_pool

    pool = await get_pool()
    total_deleted = 0

    # Find tickers with > 3 monthly summary vectors
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT ticker, COUNT(*) as cnt
                FROM investment_memories
                WHERE metadata->>'type' = 'monthly_summary'
                GROUP BY ticker
                HAVING COUNT(*) > 3
                ORDER BY cnt DESC
                """
            )
            candidates = await cur.fetchall()

    for row in candidates:
        prune_ticker = row["ticker"]

        # Get all monthly summaries, oldest first
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT * FROM investment_memories
                    WHERE ticker = %s
                      AND metadata->>'type' = 'monthly_summary'
                    ORDER BY created_at ASC
                    """,
                    (prune_ticker,),
                )
                rows = await cur.fetchall()

        if len(rows) <= 3:
            continue

        from utils.vector_memory import _row_to_memory
        all_summaries = [_row_to_memory(r) for r in rows]

        # Keep the 3 most recent, merge the rest
        to_merge = all_summaries[:-3]
        months = [
            m.metadata.get("month", "?") for m in to_merge
        ]
        
        raw_dump = "\n\n---\n\n".join(
            f"### {m.metadata.get('month', 'Unknown Month')}\n{m.summary}"
            for m in to_merge
        )

        synthesis_prompt = (
            f"You are a senior equity research analyst compressing historical data for {prune_ticker}.\n\n"
            f"Below are {len(to_merge)} monthly summaries spanning from {months[0]} to {months[-1]}.\n"
            f"Since this is older historical data, your goal is to distil the most critical, enduring "
            f"themes, financial trends, and risk factors into a single cohesive historical summary.\n\n"
            f"Format:\n"
            f"## {prune_ticker} — Historical Summary ({months[0]} to {months[-1]})\n\n"
            f"### Long-Term Themes\n"
            f"- 3-5 bullet points of enduring narrative or structural shifts\n\n"
            f"### Financial Evolution\n"
            f"- Key trajectory of revenue, margins, or capital allocation\n\n"
            f"### Enduring Risks\n"
            f"- Systemic or long-tailed risks that persisted across these months\n\n"
            f"Rules:\n"
            f"- Be extremely concise: The entire summary must be under 500 words.\n"
            f"- Discard short-term noise or resolved events.\n"
            f"- Preserve critical numerical specifics (e.g., peak margins, major acquisition costs).\n"
            f"- Do NOT invent data.\n\n"
            f"--- RAW DATA ---\n\n{raw_dump}"
        )

        try:
            response = await curator_model.ainvoke(synthesis_prompt)
            summary_text = str(response.content)
        except Exception as e:
            logger.warning(
                f"[Curator] Historical LLM synthesis failed for {prune_ticker}, "
                f"falling back to concatenation: {e}"
            )
            # Fallback to concatenation if the model fails
            summaries_text = "; ".join(m.summary for m in to_merge)
            summary_text = (
                f"[Historical Summary — {months[0]} to {months[-1]}] {summaries_text}"
            )

        try:
            embedding = await get_embedding(summary_text)
            ids_to_merge = [m.id for m in to_merge]
            await vector_memory.merge_to_summary_vector(
                ticker=prune_ticker,
                memory_ids=ids_to_merge,
                summary_text=summary_text,
                embedding=embedding,
            )
            total_deleted += len(ids_to_merge)
        except Exception as e:
            logger.warning(
                f"[Curator] Aggressive prune failed for {prune_ticker}: {e}"
            )

    return total_deleted


async def _consolidate_duplicate_summaries(ticker: str) -> int:
    """Collapse near-duplicate monthly/historical summary vectors."""
    pairs = await vector_memory.find_near_duplicate_summary_pairs(ticker)
    if not pairs:
        return 0

    used: set[str] = set()
    consolidated = 0
    for pair in pairs:
        keeper_id = str(pair.get("keeper_memory_id") or "")
        duplicate_id = str(pair.get("duplicate_memory_id") or "")
        if not keeper_id or not duplicate_id or keeper_id in used or duplicate_id in used:
            continue

        keeper_summary = str(pair.get("keeper_summary") or "")
        duplicate_summary = str(pair.get("duplicate_summary") or "")
        raw_dump = f"### Existing Summary\n{keeper_summary}\n\n### Duplicate Summary\n{duplicate_summary}"
        prompt = (
            f"You are consolidating near-duplicate historical research summaries for {ticker}.\n"
            "Synthesize one concise replacement summary. Preserve durable thesis facts, "
            "important numeric specifics, and avoid inventing data.\n\n"
            f"{raw_dump}"
        )
        try:
            response = await curator_model.ainvoke(prompt)
            summary_text = str(response.content)
        except Exception:
            summary_text = f"{keeper_summary}\n\n{duplicate_summary}"

        try:
            embedding = await get_embedding(summary_text)
            await vector_memory.merge_to_summary_vector(
                ticker=ticker,
                memory_ids=[keeper_id, duplicate_id],
                summary_text=summary_text,
                embedding=embedding,
            )
            used.update({keeper_id, duplicate_id})
            consolidated += 1
        except Exception as e:
            logger.warning("[Curator] Duplicate summary consolidation failed for %s: %s", ticker, e)

    return consolidated
