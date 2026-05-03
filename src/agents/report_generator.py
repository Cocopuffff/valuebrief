from typing import Literal
from langgraph.types import Command
from agents.states import WorkflowState
from agents.judge import _build_dcf_summary
from schemas import AgentNode

from utils.logger import log_node_execution, logging
from utils.db import upsert_valuation
from utils.report_writer import RunReportWriter
from utils.citations import build_citation_manifest
from utils.research_persistence import (
    format_artifact_citations,
    memory_ids_from_artifact,
    persist_research_artifact,
)
import json

logger = logging.getLogger(__name__)

@log_node_execution
async def report_generator(state: WorkflowState) -> Command[Literal[AgentNode.CURATOR]]:
    """Generate the final investment report including valuation results."""
    logger.info(f"Generating final report for {state['company']}...")

    # ── Valuation section ──
    valuation = state.get("valuation")
    if valuation:
        valuation_section = f"""
{'─'*50}
DCF VALUATION
{'─'*50}

{_build_dcf_summary(valuation)}
"""
    else:
        valuation_section = "\n(Valuation data unavailable)\n"

    # ── Deduplicate and format sources using a set ──
    unique_sources = sorted(list(set(state.get("sources", []))))
    sources_formatted = "\n".join(f"- {s}" for s in unique_sources) if unique_sources else "N/A"

    report = f"""
INVESTMENT THESIS:
{state.get('judge_decision', 'N/A')}
{valuation_section}

SOURCES:
{sources_formatted}
"""
    report_artifact: dict = {}
    report_artifacts = list(state.get("vault_artifacts", []))
    try:
        report_artifact = await persist_research_artifact(
            ticker=state["ticker"],
            content=report,
            source_type="final_report",
            source_priority=2,
            metadata={
                "agent": "report_generator",
                "company": state.get("company", ""),
                "run_datetime": state.get("run_datetime", ""),
                "source_urls": unique_sources,
            },
        )
        if report_artifact.path:
            report_artifacts.append(report_artifact)
    except Exception as e:
        logger.warning(f"[Report Generator] ⚠️ Failed to persist final report: {e}")

    vault_citations = format_artifact_citations(report_artifacts)
    if vault_citations:
        report = f"{report}\nVAULT CITATIONS:\n{vault_citations}\n"

    debug_report = f"""
DEBUG REPORT: {state['company']} ({state['ticker']})

BULL THESIS:
{state.get('bull_thesis', 'N/A')}

BEAR THESIS:
{state.get('bear_thesis', 'N/A')}

JUDGE DECISION:
{state.get('judge_decision', 'N/A')}
{valuation_section}

SOURCES:
{sources_formatted}

VAULT CITATIONS:
{vault_citations or 'N/A'}

WORKFLOW_STATE:
{json.dumps(state, indent=2, default=str)}
"""
    # Persist final report to run artifact
    run_dt = state.get("run_datetime", "")
    if run_dt:
        try:
            writer = RunReportWriter(
                ticker=state["ticker"], 
                run_datetime=run_dt, 
                company=state.get("company", "")
            )
            writer.write_final_report(debug_report, report, unique_sources)
            logger.info(f"[Report Generator] 📝 Written final report to {writer.final_path}")
            if valuation:
                await upsert_valuation(valuation)
                logger.info(f"[Report Generator] 📝 Upserted valuation for {state['ticker']} to database")
            else:
                logger.warning(f"[Report Generator] ⚠️ No valuation data available for {state['ticker']}")
        except TimeoutError:
            logger.warning(f"[Report Generator] ⚠️ Timed out writing final report for {state['ticker']}")
        except Exception as e:
            logger.warning(f"[Report Generator] ⚠️ Failed to write final report: {e}")

    # Build citation manifest for the Curator
    manifest = [
        {"file_path": c.file_path, "block_id": c.block_id, "resolved_text": c.resolved_text}
        for c in build_citation_manifest(report, state["ticker"])
    ]

    update = {"final_report": report, "citation_manifest": manifest}
    if getattr(report_artifact, "path", False):
        update["vault_artifacts"] = [report_artifact.model_dump(mode="json")]
        update["active_memory_ids"] = memory_ids_from_artifact(report_artifact)

    return Command(update=update, goto=AgentNode.CURATOR)
