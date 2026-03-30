
from langchain.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from .states import WorkflowState, ResearchState
from .supervisor import supervisor, route_supervisor
from .analysts import bull_analyst, bear_analyst, research_tool_node, should_continue
from .judge import judge_analyst

# —— Bull analyst sub graph ——
bull_subgraph_builder = StateGraph(ResearchState)
bull_subgraph_builder.add_node("bull_analyst", bull_analyst)
bull_subgraph_builder.add_node("research_tool_node", research_tool_node)
bull_subgraph_builder.add_edge(START, "bull_analyst")
bull_subgraph_builder.add_conditional_edges(
    "bull_analyst",
    should_continue,
    {"research_tool_node": "research_tool_node", "supervisor": END}
)
bull_subgraph_builder.add_edge("research_tool_node", "bull_analyst")
bull_subgraph = bull_subgraph_builder.compile()

# —— Bear analyst sub graph ——
bear_subgraph_builder = StateGraph(ResearchState)
bear_subgraph_builder.add_node("bear_analyst", bear_analyst)
bear_subgraph_builder.add_node("research_tool_node", research_tool_node)
bear_subgraph_builder.add_edge(START, "bear_analyst")
bear_subgraph_builder.add_conditional_edges(
    "bear_analyst",
    should_continue,
    {"research_tool_node": "research_tool_node", "supervisor": END}
)
bear_subgraph_builder.add_edge("research_tool_node", "bear_analyst")
bear_subgraph = bear_subgraph_builder.compile()

# —— Wrapper functions (bridge WorkflowState ↔ ResearchState) ——
def run_bull_research(state: WorkflowState) -> dict:
    research_input: ResearchState = {
        "date": state["date"],
        "company": state["company"],
        "ticker": state["ticker"],
        "price_data": state["price_data"],
        "max_iterations": 5,
        "iteration_count": 0,
        "research_topics": [],
        "key_points": [],
        "thesis": "",
        "sources": [],
        "messages": [],
    }
    result = bull_subgraph.invoke(research_input)
    return {"bull_thesis": result["thesis"], "bull_sources": result["sources"]}

def run_bear_research(state: WorkflowState) -> dict:
    research_input: ResearchState = {
        "date": state["date"],
        "company": state["company"],
        "ticker": state["ticker"],
        "price_data": state["price_data"],
        "max_iterations": 0,
        "iteration_count": 0,
        "research_topics": [],
        "key_points": [],
        "thesis": "",
        "sources": [],
        "messages": [],
    }
    result = bear_subgraph.invoke(research_input)
    return {"bear_thesis": result["thesis"], "bear_sources": result["sources"]}

def report_generator(state: WorkflowState) -> dict:
    """Generate the final investment report."""
    print(f"\n[Report] Generating final report for {state['company']}...")
    report = f"""
INVESTMENT REPORT: {state['company']} ({state['ticker']})
{'='*50}

BULL THESIS:
{state.get('bull_thesis', 'N/A')}

BEAR THESIS:
{state.get('bear_thesis', 'N/A')}

JUDGE DECISION:
{state.get('judge_decision', 'N/A')}
"""
    return {"final_report": report}

# —— Main workflow ——
workflow_builder = StateGraph(WorkflowState)
workflow_builder.add_node("supervisor", supervisor)
workflow_builder.add_node("bull_research", run_bull_research)
workflow_builder.add_node("bear_research", run_bear_research)
workflow_builder.add_node("judge_analyst", judge_analyst)
workflow_builder.add_node("report_generator", report_generator)

workflow_builder.add_edge(START, "supervisor")
# Supervisor conditionally routes: research phase or judge
workflow_builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {"bull_research": "bull_research", "judge_analyst": "judge_analyst"}
)
workflow_builder.add_edge("bull_research", "bear_research")
workflow_builder.add_edge("bear_research", "supervisor")
# Final phase
workflow_builder.add_edge("judge_analyst", "report_generator")
workflow_builder.add_edge("report_generator", END)

workflow = workflow_builder.compile()