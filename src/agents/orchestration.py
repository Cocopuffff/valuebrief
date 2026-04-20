
from langchain.messages import AnyMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from typing import Literal
from agents.states import WorkflowState, ResearchState
from agents.supervisor import supervisor
from agents.analysts import bull_analyst, bear_analyst, research_tool_node
from agents.judge import judge_analyst
from agents.report_generator import report_generator
from models import AgentNode
from utils.logger import get_logger
from utils.report_writer import RunReportWriter
from utils.config import config

logger = get_logger(__name__)


# —— Conditional edge router for analyst subgraphs ——
def analyst_router(state: ResearchState) -> Literal[AgentNode.RESEARCH_TOOL_NODE, "__end__"]:
    """Route after an analyst runs: execute tools if pending, otherwise end."""
    if state.get("thesis"):
        return "__end__"
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return AgentNode.RESEARCH_TOOL_NODE
    return "__end__"


# —— Bull analyst sub graph ——
bull_subgraph = (
    StateGraph(ResearchState)
    .add_node(AgentNode.BULL, bull_analyst)
    .add_node(AgentNode.RESEARCH_TOOL_NODE, research_tool_node)
    .add_edge(START, AgentNode.BULL)
    .add_conditional_edges(
        AgentNode.BULL,
        analyst_router,
        {
            AgentNode.RESEARCH_TOOL_NODE: AgentNode.RESEARCH_TOOL_NODE,
            "__end__": END,
        },
    )
    .add_edge(AgentNode.RESEARCH_TOOL_NODE, AgentNode.BULL)
    .compile()
)

# —— Bear analyst sub graph ——
bear_subgraph = (
    StateGraph(ResearchState)
    .add_node(AgentNode.BEAR, bear_analyst)
    .add_node(AgentNode.RESEARCH_TOOL_NODE, research_tool_node)
    .add_edge(START, AgentNode.BEAR)
    .add_conditional_edges(
        AgentNode.BEAR,
        analyst_router,
        {
            AgentNode.RESEARCH_TOOL_NODE: AgentNode.RESEARCH_TOOL_NODE,
            "__end__": END,
        },
    )
    .add_edge(AgentNode.RESEARCH_TOOL_NODE, AgentNode.BEAR)
    .compile()
)

# —— Wrapper functions (bridge WorkflowState ↔ ResearchState) ——
async def run_bull_research(state: WorkflowState) -> Command[Literal[AgentNode.SUPERVISOR]]:
    research_input: ResearchState = {
        "date": state["date"],
        "company": state["company"],
        "ticker": state["ticker"],
        "price_data": state["price_data"],
        "existing_valuation": state.get("valuation"),
        "max_iterations": config.bull.max_iterations,
        "iteration_count": 0,
        "research_topics": [],
        "key_points": [],
        "thesis": "",
        "sources": [],
        "messages": [],
    }
    result = await bull_subgraph.ainvoke(research_input)
    update = {"bull_thesis": result["thesis"], "sources": state["sources"] + result["sources"]}

    # Persist bull thesis to run artifact
    run_dt = state.get("run_datetime", "")
    if run_dt:
        try:
            writer = RunReportWriter(ticker=state["ticker"], run_datetime=run_dt)
            writer.write_bull_thesis(result["thesis"])
            logger.info(f"[Bull Research] 📝 Written bull thesis to {writer.debug_path}")
        except Exception as e:
            logger.warning(f"[Bull Research] ⚠️ Failed to write bull thesis: {e}")

    return Command(update=update, goto=AgentNode.SUPERVISOR)

async def run_bear_research(state: WorkflowState) -> Command[Literal[AgentNode.SUPERVISOR]]:
    research_input: ResearchState = {
        "date": state["date"],
        "company": state["company"],
        "ticker": state["ticker"],
        "price_data": state["price_data"],
        "existing_valuation": state.get("valuation"),
        "max_iterations": config.bear.max_iterations,
        "iteration_count": 0,
        "research_topics": [],
        "key_points": [],
        "thesis": "",
        "sources": [],
        "messages": [],
    }
    result = await bear_subgraph.ainvoke(research_input)
    update = {"bear_thesis": result["thesis"], "sources": state["sources"] + result["sources"]}

    # Persist bear thesis to run artifact
    run_dt = state.get("run_datetime", "")
    if run_dt:
        try:
            writer = RunReportWriter(ticker=state["ticker"], run_datetime=run_dt)
            writer.write_bear_thesis(result["thesis"])
            logger.info(f"[Bear Research] 📝 Written bear thesis to {writer.debug_path}")
        except Exception as e:
            logger.warning(f"[Bear Research] ⚠️ Failed to write bear thesis: {e}")

    return Command(update=update, goto=AgentNode.SUPERVISOR)

def build_research_workflow(checkpointer: AsyncPostgresSaver):
    """Main research workflow.

    Flow: START → Supervisor → Bull/Bear Research → Supervisor → Judge
          (synthesise + valuate + reconcile) → Report Generator → END
    """
    store = InMemoryStore()
    return (
        StateGraph(WorkflowState)
        .add_node(AgentNode.SUPERVISOR, supervisor)
        .add_node("bull_research", run_bull_research)
        .add_node("bear_research", run_bear_research)
        .add_node(AgentNode.JUDGE, judge_analyst)
        .add_node(AgentNode.REPORT_GENERATOR, report_generator)
        .add_edge(START, AgentNode.SUPERVISOR)
        # Supervisor uses Command(goto=...) for dynamic routing — no conditional edges needed
        # Judge routes directly to report_generator via Command
        .add_edge(AgentNode.REPORT_GENERATOR, END)
        .compile(checkpointer=checkpointer, store=store)
    )