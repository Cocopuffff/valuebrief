from langgraph.types import Command
from typing import Literal, cast
from datetime import date
from agents.states import WorkflowState
from schemas import AgentNode, ResearchTask, SourceInventoryRecord
from provider import DateTimeProvider, FinancialDataProvider
from utils.logger import get_logger, log_node_execution
from utils.report_writer import RunReportWriter
from utils import vector_memory
from utils.research_persistence import memory_ids_from_artifact, persist_research_artifact
from utils.config import supervisor_model
from utils.vault import VAULT_ROOT, _parse_frontmatter

logger = get_logger(__name__)

tools = [DateTimeProvider.get_current_date, FinancialDataProvider.get_asset_data, FinancialDataProvider.get_multiple_assets]
tools_by_name = {tool.name: tool for tool in tools}
supervisor_model_with_tools = supervisor_model.bind_tools(tools)


def _build_fundamentals_md(asset) -> str:
    """Build a Markdown summary of an Asset's fundamentals for vault storage."""
    f = asset.fundamentals
    lines = [
        f"# {asset.name or asset.ticker} — Fundamentals Snapshot",
        "",
        f"**Ticker:** {asset.ticker}",
        f"**Current Price:** ${asset.current_price:,.2f}" if asset.current_price else None,
        f"**Shares Outstanding:** {asset.shares_outstanding/1e6:,.1f}M" if asset.shares_outstanding else None,
        f"**Market Cap:** ${f.market_cap/1e6:,.1f}M" if f.market_cap else None,
        f"**Total Revenue (TTM):** ${f.total_revenue/1e6:,.1f}M" if f.total_revenue else None,
        f"**Revenue Growth (YoY):** {f.revenue_growth:.1%}" if f.revenue_growth is not None else None,
        f"**EBITDA Margin:** {f.ebitda_margin:.1%}" if f.ebitda_margin is not None else None,
        f"**Free Cash Flow:** ${f.free_cash_flow/1e6:,.1f}M" if f.free_cash_flow else None,
        f"**Trailing P/E:** {f.pe_ratio:.1f}" if f.pe_ratio else None,
        f"**Forward P/E:** {f.forward_pe_ratio:.1f}" if f.forward_pe_ratio else None,
        f"**PEG Ratio:** {f.peg_ratio:.2f}" if f.peg_ratio else None,
        f"**Debt/Equity:** {f.debt_to_equity:.1f}" if f.debt_to_equity else None,
        f"**ROE:** {f.return_on_equity:.1%}" if f.return_on_equity is not None else None,
        f"**Sector:** {asset.sector}" if asset.sector else None,
        f"**Industry:** {asset.industry}" if asset.industry else None,
    ]
    return "\n".join(line for line in lines if line is not None)


def _default_research_goal(state: WorkflowState) -> str:
    company = state.get("company") or state["ticker"]
    ticker = state["ticker"]
    return (
        f"Act as an expert stock research analyst to research and create an "
        f"investment thesis and DCF valuation inputs for {company} ({ticker}) "
        "from a value investor point of view. Use the latest information "
        "available. The research should not have buy or sell bias and should "
        "include hallmarks of value investing such as margin of safety. Do not "
        "make bad copy. Ensure all information is properly cited and annotated. "
        "Be objective."
    )


def _source_inventory_from_vault(ticker: str) -> list[dict]:
    """Return known filing/transcript inventory from local vault metadata."""
    ticker_dir = VAULT_ROOT / ticker.upper()
    if not ticker_dir.exists():
        return []

    inventory_source_types = {
        "source_inventory",
        "sec_filing",
        "earnings_report",
        "earnings_transcript",
    }
    records: list[dict] = []
    for path in sorted(ticker_dir.glob("*.md")):
        try:
            fm, _ = _parse_frontmatter(path.read_text(encoding="utf-8"))
        except Exception:
            fm = {}

        for raw_record in fm.get("source_inventory_records") or []:
            if not isinstance(raw_record, dict):
                continue
            record_dict = {**raw_record}
            record_dict.setdefault("ticker", ticker)
            record_dict.setdefault("local_path", str(path))
            try:
                records.append(
                    SourceInventoryRecord(**record_dict).model_dump(mode="json")
                )
            except Exception as e:
                logger.debug("[Supervisor] Skipping malformed source inventory record in %s: %s", path, e)

        source_type = str(fm.get("source_type") or "")
        url = str(fm.get("url") or "")
        if source_type not in inventory_source_types and "sec.gov" not in url:
            continue
        record = SourceInventoryRecord(
            ticker=ticker,
            source_type=source_type or "unknown",
            title=str(fm.get("title") or path.name),
            form_type=str(fm.get("form_type") or ""),
            period=str(fm.get("period") or fm.get("report_date") or ""),
            filed_at=str(fm.get("filed_at") or fm.get("filing_date") or ""),
            url=url,
            accession_number=str(fm.get("accession_number") or ""),
            local_path=str(path),
            freshness=str(fm.get("freshness") or "unknown"),
        )
        records.append(record.model_dump(mode="json"))
    return records


def _parse_iso_date(value: object) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _is_filing_inventory_record(record: dict) -> bool:
    source_type = str(record.get("source_type") or "")
    url = str(record.get("url") or "")
    return (
        source_type in ("source_inventory", "sec_filing")
        or bool(record.get("form_type"))
        or bool(record.get("accession_number"))
        or "sec.gov" in url
    )


def _has_current_source_inventory(
    records: list[dict],
    as_of_date: str,
    *,
    max_age_days: int = 80,
) -> bool:
    as_of = _parse_iso_date(as_of_date)
    if as_of is None:
        return False

    latest_filed_at: date | None = None
    for record in records:
        if not _is_filing_inventory_record(record):
            continue
        filed_at = _parse_iso_date(record.get("filed_at"))
        if filed_at is None:
            continue
        if latest_filed_at is None or filed_at > latest_filed_at:
            latest_filed_at = filed_at

    if latest_filed_at is None:
        return False

    age_days = (as_of - latest_filed_at).days
    return 0 <= age_days <= max_age_days


def _build_research_tasks(state: WorkflowState, source_inventory: list[dict]) -> list[dict]:
    """Create one bounded research task; Deep Agents manages the internal todo list."""
    company = state.get("company") or state["ticker"]
    ticker = state["ticker"]
    task = ResearchTask(
        task_id="deep_agent_research",
        kind="research_synthesis",
        title="Run bounded Deep Agent research",
        objective=(
            f"Research {company} ({ticker}) once using Deep Agents' internal todo "
            "list and return source inventory, business model, financial baseline, "
            "risks, DCF input evidence, and candidate investment pillars."
        ),
        acceptance_criteria=[
            "Use Deep Agents' write_todos planning internally.",
            "Return multiple cited ResearchFinding records in one ResearchFindingBundle.",
            "Include a Candidate Investment Pillars section in synthesis.",
        ],
        source_requirements=[
            "SEC filings",
            "company investor relations",
            "earnings transcript",
            "current reputable news or industry sources",
        ],
    )
    return [task.model_dump(mode="json")]


def _task_by_id(tasks: list[dict], task_id: str) -> dict | None:
    for task in tasks:
        if task.get("task_id") == task_id:
            return task
    return None


def _dependencies_complete(task: dict, tasks: list[dict]) -> bool:
    for dep_id in task.get("depends_on", []):
        dep = _task_by_id(tasks, dep_id)
        if dep and dep.get("status") not in ("completed", "skipped", "blocked"):
            return False
    return True


def _next_pending_task(tasks: list[dict]) -> dict | None:
    for task in tasks:
        if task.get("status") == "pending" and _dependencies_complete(task, tasks):
            return task
    return None


def _set_task_status(tasks: list[dict], task_id: str, status: str) -> list[dict]:
    updated = []
    for task in tasks:
        item = dict(task)
        if item.get("task_id") == task_id:
            item["status"] = status
        updated.append(item)
    return updated


@log_node_execution
async def supervisor(state: WorkflowState) -> Command[Literal[AgentNode.JUDGE, AgentNode.RESEARCH_ANALYST, "__end__"]]:
    """
    Supervisor coordinates the workflow and tracks progress.
    Fetches price data on first run, then logs analyst status.
    """
    updates = {}
    if not state.get('price_data', None):
        asset = FinancialDataProvider._get_asset_data(state["ticker"])
        
        if not asset or not asset.current_price or not asset.shares_outstanding or getattr(asset.fundamentals, "total_revenue", 0) in (None, 0):
            logger.error(f"[Supervisor] ⚠️ Insufficient financial data for {state['ticker']}. Canceling analysis.")
            return Command(goto="__end__")

        if asset:
            updates["price_data"] = asset
            if not state.get("company"):
                company = asset.name or state["ticker"]
                updates["company"] = company

                # First pass — company is now known; initialise the run artifact
                run_dt = state.get("run_datetime", "")
                if run_dt:
                    try:
                        writer = RunReportWriter(
                            ticker=state["ticker"],
                            run_datetime=run_dt,
                            company=company
                        )
                        writer.write_header()
                        logger.info(f"[Supervisor] 📝 Run artifact initialised: {writer.debug_path}")
                    except Exception as e:
                        logger.warning(f"[Supervisor] ⚠️ Failed to write run header: {e}")

            # Save fundamentals to the local vault
            try:
                fundamentals_md = _build_fundamentals_md(asset)
                artifact = await persist_research_artifact(
                    ticker=state["ticker"],
                    content=fundamentals_md,
                    source_type="fundamentals",
                    source_priority=2,
                    vectorize=False,
                    metadata={
                        "url": "alphavantage+yfinance",
                        "sentiment": "neutral",
                        "company": asset.name or state["ticker"],
                        "run_datetime": state.get("run_datetime", ""),
                    },
                )
                if artifact.path:
                    updates["vault_artifacts"] = [artifact.model_dump(mode="json")]
                    updates["active_memory_ids"] = memory_ids_from_artifact(artifact)
            except Exception as e:
                logger.warning(f"[Supervisor] ⚠️ Failed to write vault entry: {e}")

    # ── RAG retrieval: pull active thesis pillars for this ticker ──
    if not state.get("rag_context") and not state.get("retrieved_memory_ids"):
        try:
            result = await vector_memory.retrieve_active_pillars(
                ticker=state["ticker"],
            )
            updates["rag_context"] = result.get("memory_context", "")
            updates["retrieved_memory_ids"] = result.get("retrieved_memory_ids", [])
            updates["research_topics"] = result.get("topics", [])
            pillar_count = len(result.get("retrieved_memory_ids", []))
            if pillar_count:
                logger.info(
                    "[Supervisor] 🔍 Retrieved %d active thesis pillars for %s",
                    pillar_count, state["ticker"],
                )
            else:
                logger.info(
                    "[Supervisor] No active pillars for %s — pillar cold start",
                    state["ticker"],
                )
        except Exception as e:
            logger.warning("[Supervisor] ⚠️ Pillar retrieval failed, continuing: %s", e)

    if not state.get("research_goal") and not updates.get("research_goal"):
        updates["research_goal"] = _default_research_goal(
            cast(WorkflowState, {**state, **updates})
        )

    if not state.get("source_inventory") and not updates.get("source_inventory"):
        updates["source_inventory"] = _source_inventory_from_vault(state["ticker"])

    merged_state = cast(WorkflowState, {**state, **updates})
    research_complete = bool(
        merged_state.get("research_findings") or merged_state.get("research_synthesis")
    )
    if not research_complete and not merged_state.get("research_tasks"):
        updates["research_tasks"] = _build_research_tasks(
            merged_state,
            merged_state.get("source_inventory", []),
        )
        merged_state = cast(
            WorkflowState,
            {**merged_state, "research_tasks": updates["research_tasks"]},
        )

    if not research_complete:
        tasks = list(merged_state.get("research_tasks", []))
        next_task = _next_pending_task(tasks)
        if next_task:
            tasks = _set_task_status(tasks, str(next_task["task_id"]), "in_progress")
            updates["research_tasks"] = tasks
            updates["current_task_id"] = str(next_task["task_id"])
            logger.info(
                "[Supervisor] Routing to Deep Agent research task %s: %s",
                next_task.get("task_id"),
                next_task.get("title"),
            )
            return Command(update=updates, goto=AgentNode.RESEARCH_ANALYST)

    logger.info("[Supervisor] Research tasks complete. Routing to Judge.")
    goto = AgentNode.JUDGE

    return Command(update=updates, goto=goto)
