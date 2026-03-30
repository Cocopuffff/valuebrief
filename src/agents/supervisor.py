from langchain.chat_models import init_chat_model
from typing import Literal
from .states import WorkflowState
from provider import *

supervisor_model = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    temperature=0
)

tools = [DateTimeProvider.get_current_date, FinancialDataProvider.get_asset_data, FinancialDataProvider.get_multiple_assets]
tools_by_name = {tool.name: tool for tool in tools}
supervisor_model_with_tools = supervisor_model.bind_tools(tools)

def supervisor(state: WorkflowState) -> dict:
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
                updates["company"] = asset.name or state["ticker"]
    
    bull_done = state.get('bull_thesis')
    bear_done = state.get('bear_thesis')

    print(f"\n{'='*50}")
    print(f"SUPERVISOR:")
    print(f"    Bull complete: {bool(bull_done)}")
    print(f"    Bear complete: {bool(bear_done)}")
    print(f"{'='*50}")

    return updates

def route_supervisor(state: WorkflowState) -> Literal["bull_research", "judge_analyst"]:
    """
    Conditional routing after supervisor:
    - If both theses exist → move to judge
    - Otherwise → run analyst research
    """
    bull_done = state.get('bull_thesis')
    bear_done = state.get('bear_thesis')

    if bull_done and bear_done:
        print("[Router] Both theses complete → Judge")
        return "judge_analyst"
    
    print("[Router] Theses incomplete → Research phase")
    return "bull_research"
