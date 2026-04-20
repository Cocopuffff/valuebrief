from langgraph.types import Command
from typing import Literal
from agents.states import WorkflowState
from models import AgentNode
from provider import *
from utils.logger import get_logger, log_node_execution
from utils.report_writer import RunReportWriter
from utils.config import supervisor_model

logger = get_logger(__name__)

tools = [DateTimeProvider.get_current_date, FinancialDataProvider.get_asset_data, FinancialDataProvider.get_multiple_assets]
tools_by_name = {tool.name: tool for tool in tools}
supervisor_model_with_tools = supervisor_model.bind_tools(tools)

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
                        writer = RunReportWriter(ticker=state["ticker"], run_datetime=run_dt)
                        writer.write_header(company=company)
                        logger.info(f"[Supervisor] 📝 Run artifact initialised: {writer.debug_path}")
                    except Exception as e:
                        logger.warning(f"[Supervisor] ⚠️ Failed to write run header: {e}")
    
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
