
from langchain.messages import AnyMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from typing import Literal
from agents.states import WorkflowState, ResearchState
from agents.supervisor import supervisor
from agents.analysts import bull_analyst, bear_analyst, research_tool_node
from agents.judge import judge_analyst, _build_dcf_summary
from models import AgentNode
from logger import get_logger
from report_writer import RunReportWriter
import json

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
bull_subgraph_builder = StateGraph(ResearchState)
bull_subgraph_builder.add_node(AgentNode.BULL, bull_analyst)
bull_subgraph_builder.add_node(AgentNode.RESEARCH_TOOL_NODE, research_tool_node)
bull_subgraph_builder.add_edge(START, AgentNode.BULL)
bull_subgraph_builder.add_conditional_edges(
    AgentNode.BULL,
    analyst_router,
    {
        AgentNode.RESEARCH_TOOL_NODE: AgentNode.RESEARCH_TOOL_NODE,
        "__end__": END,
    }
)
bull_subgraph_builder.add_edge(AgentNode.RESEARCH_TOOL_NODE, AgentNode.BULL)
bull_subgraph = bull_subgraph_builder.compile()

# —— Bear analyst sub graph ——
bear_subgraph_builder = StateGraph(ResearchState)
bear_subgraph_builder.add_node(AgentNode.BEAR, bear_analyst)
bear_subgraph_builder.add_node(AgentNode.RESEARCH_TOOL_NODE, research_tool_node)
bear_subgraph_builder.add_edge(START, AgentNode.BEAR)
bear_subgraph_builder.add_conditional_edges(
    AgentNode.BEAR,
    analyst_router,
    {
        AgentNode.RESEARCH_TOOL_NODE: AgentNode.RESEARCH_TOOL_NODE,
        "__end__": END,
    }
)
bear_subgraph_builder.add_edge(AgentNode.RESEARCH_TOOL_NODE, AgentNode.BEAR)
bear_subgraph = bear_subgraph_builder.compile()

# —— Wrapper functions (bridge WorkflowState ↔ ResearchState) ——
def run_bull_research(state: WorkflowState) -> Command[Literal[AgentNode.SUPERVISOR]]:
    research_input: ResearchState = {
        "date": state["date"],
        "company": state["company"],
        "ticker": state["ticker"],
        "price_data": state["price_data"],
        "max_iterations": 10,
        "iteration_count": 0,
        "research_topics": [],
        "key_points": [],
        "thesis": "",
        "sources": [],
        "messages": [],
    }
    result = bull_subgraph.invoke(research_input)
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

def run_bear_research(state: WorkflowState) -> Command[Literal[AgentNode.SUPERVISOR]]:
    research_input: ResearchState = {
        "date": state["date"],
        "company": state["company"],
        "ticker": state["ticker"],
        "price_data": state["price_data"],
        "max_iterations": 10,
        "iteration_count": 0,
        "research_topics": [],
        "key_points": [],
        "thesis": "",
        "sources": [],
        "messages": [],
    }
    result = bear_subgraph.invoke(research_input)
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

def report_generator(state: WorkflowState) -> Command[Literal["__end__"]]:
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
{json.dumps(state, indent=2)}
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
            writer = RunReportWriter(ticker=state["ticker"], run_datetime=run_dt)
            writer.write_final_report(debug_report, report, unique_sources)
            logger.info(f"[Report Generator] 📝 Written final report to {writer.final_path}")
        except Exception as e:
            logger.warning(f"[Report Generator] ⚠️ Failed to write final report: {e}")

    return Command(update={"final_report": report}, goto="__end__")



def build_research_workflow(checkpointer: AsyncPostgresSaver):
    """Main research workflow.

    Flow: START → Supervisor → Bull/Bear Research → Supervisor → Judge
          (synthesise + valuate + reconcile) → Report Generator → END
    """
    store = InMemoryStore()
    builder = StateGraph(WorkflowState)

    builder.add_node(AgentNode.SUPERVISOR, supervisor)
    builder.add_node("bull_research", run_bull_research)
    builder.add_node("bear_research", run_bear_research)
    builder.add_node(AgentNode.JUDGE, judge_analyst)
    builder.add_node(AgentNode.REPORT_GENERATOR, report_generator)

    builder.add_edge(START, AgentNode.SUPERVISOR)
    # Supervisor uses Command(goto=...) for dynamic routing — no conditional edges needed
    # Judge routes directly to report_generator via Command
    builder.add_edge(AgentNode.REPORT_GENERATOR, END)

    graph = builder.compile(checkpointer=checkpointer, store=store)
    return graph