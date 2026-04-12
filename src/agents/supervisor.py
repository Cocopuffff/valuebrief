from langchain_openrouter import ChatOpenRouter
from langgraph.types import Command
from typing import Literal
from .states import WorkflowState
from models import AgentNode
from provider import *
from logger import get_logger
from report_writer import RunReportWriter

logger = get_logger(__name__)

supervisor_model = ChatOpenRouter(
    model="qwen/qwen3.6-plus",
    temperature=0.2
)

tools = [DateTimeProvider.get_current_date, FinancialDataProvider.get_asset_data, FinancialDataProvider.get_multiple_assets]
tools_by_name = {tool.name: tool for tool in tools}
supervisor_model_with_tools = supervisor_model.bind_tools(tools)

def supervisor(state: WorkflowState) -> Command[Literal[AgentNode.JUDGE, "bull_research", "bear_research"]]:
    """
    Supervisor coordinates the workflow and tracks progress.
    Fetches price data on first run, then logs analyst status.
    """
    updates = {}
    if not state.get('price_data', None):
        asset = FinancialDataProvider._get_asset_data(state["ticker"])
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

    goto = AgentNode.JUDGE
    incomplete = True
    if not bull_done:
        goto = "bull_research"
    elif not bear_done:
        goto = "bear_research"
    else:
        incomplete = False
    
    if incomplete:
        logger.info(f"Routing to: {goto}") 
    else:
        logger.info("Theses complete. Routing to Judge.")

    return Command(update=updates, goto=goto)
