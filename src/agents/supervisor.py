from langgraph.types import Command
from typing import Literal
from agents.states import WorkflowState
from schemas import AgentNode
from provider import DateTimeProvider, FinancialDataProvider
from utils.logger import get_logger, log_node_execution
from utils.report_writer import RunReportWriter
from utils.research_persistence import memory_ids_from_artifact, persist_research_artifact
from utils.config import supervisor_model

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


@log_node_execution
async def supervisor(state: WorkflowState) -> Command[Literal[AgentNode.JUDGE, "bull_research", "bear_research", "__end__"]]:
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
    
    bull_done = state.get('bull_thesis')
    bear_done = state.get('bear_thesis')

    logger.info(f"STATUS - Bull: {'COMPLETE' if bull_done else 'INCOMPLETE'}, Bear: {'COMPLETE' if bear_done else 'INCOMPLETE'}")

    goto = []
    if not bull_done:
        goto.append("bull_research")
    if not bear_done:
        goto.append("bear_research")
    
    if goto:
        logger.info(f"Routing to: {goto}") 
    else:
        goto = AgentNode.JUDGE
        logger.info("Theses complete. Routing to Judge.")

    return Command(update=updates, goto=goto)
