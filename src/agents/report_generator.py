from typing import Literal
from langgraph.types import Command
from agents.states import WorkflowState
from agents.judge import _build_dcf_summary

from utils.logger import log_node_execution, logging
from utils.db import upsert_valuation
from utils.report_writer import RunReportWriter
import json

logger = logging.getLogger(__name__)

@log_node_execution
async def report_generator(state: WorkflowState) -> Command[Literal["__end__"]]:
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

WORKFLOW_STATE:
{json.dumps(state, indent=2, default=str)}
"""

    report = f"""
INVESTMENT THESIS:
{state.get('judge_decision', 'N/A')}
{valuation_section}

SOURCES:
{sources_formatted}
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

    return Command(update={"final_report": report}, goto="__end__")