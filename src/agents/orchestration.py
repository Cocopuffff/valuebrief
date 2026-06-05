
from langchain.messages import AnyMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from typing import Literal
from agents.states import WorkflowState, ResearchState, ResearchTaskState
from agents.supervisor import supervisor
from agents.analysts import bull_analyst, bear_analyst, research_tool_node, run_research_task
from agents.judge import judge_analyst
from agents.report_generator import report_generator
from agents.curator import curator_agent
from schemas import AgentNode
from utils.logger import get_logger
from utils.report_writer import RunReportWriter
from utils.research_persistence import memory_ids_from_artifact, persist_research_artifact
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
        "research_topics": [
            t for t in state.get("research_topics", [])
            if t.get("side") in ("bull", "neutral")
        ],
        "key_points": [],
        "thesis": "",
        "sources": [],
        "messages": [],
        "rag_context": state.get("rag_context", ""),
    }
    result = await bull_subgraph.ainvoke(research_input)
    update = {"bull_thesis": result["thesis"], "sources": result["sources"]}

    try:
        artifact = await persist_research_artifact(
            ticker=state["ticker"],
            content=result["thesis"],
            source_type="bull_thesis",
            source_priority=1,
            vectorize=False,
            metadata={
                "agent": "bull",
                "company": state.get("company", ""),
                "run_datetime": state.get("run_datetime", ""),
                "source_urls": sorted(set(result.get("sources", []))),
            },
        )
        if artifact.path:
            update["vault_artifacts"] = [artifact.model_dump(mode="json")]
    except Exception as e:
        logger.warning(f"[Bull Research] ⚠️ Failed to persist bull thesis: {e}")

    # Persist bull thesis to run artifact
    run_dt = state.get("run_datetime", "")
    if run_dt:
        try:
            writer = RunReportWriter(
                ticker=state["ticker"],
                run_datetime=run_dt,
                company=state.get("company", ""),
            )
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
        "research_topics": [
            t for t in state.get("research_topics", [])
            if t.get("side") in ("bear", "neutral")
        ],
        "key_points": [],
        "thesis": "",
        "sources": [],
        "messages": [],
        "rag_context": state.get("rag_context", ""),
    }
    result = await bear_subgraph.ainvoke(research_input)
    update = {"bear_thesis": result["thesis"], "sources": result["sources"]}

    try:
        artifact = await persist_research_artifact(
            ticker=state["ticker"],
            content=result["thesis"],
            source_type="bear_thesis",
            source_priority=1,
            vectorize=False,
            metadata={
                "agent": "bear",
                "company": state.get("company", ""),
                "run_datetime": state.get("run_datetime", ""),
                "source_urls": sorted(set(result.get("sources", []))),
            },
        )
        if artifact.path:
            update["vault_artifacts"] = [artifact.model_dump(mode="json")]
    except Exception as e:
        logger.warning(f"[Bear Research] ⚠️ Failed to persist bear thesis: {e}")

    # Persist bear thesis to run artifact
    run_dt = state.get("run_datetime", "")
    if run_dt:
        try:
            writer = RunReportWriter(
                ticker=state["ticker"],
                run_datetime=run_dt,
                company=state.get("company", ""),
            )
            writer.write_bear_thesis(result["thesis"])
            logger.info(f"[Bear Research] 📝 Written bear thesis to {writer.debug_path}")
        except Exception as e:
            logger.warning(f"[Bear Research] ⚠️ Failed to write bear thesis: {e}")

    return Command(update=update, goto=AgentNode.SUPERVISOR)


def _task_by_id(tasks: list[dict], task_id: str) -> dict:
    for task in tasks:
        if task.get("task_id") == task_id:
            return task
    return {}


def _set_task_status(tasks: list[dict], task_id: str, status: str) -> list[dict]:
    updated: list[dict] = []
    for task in tasks:
        item = dict(task)
        if item.get("task_id") == task_id:
            item["status"] = status
        updated.append(item)
    return updated


def _source_inventory_records_from_finding(state: WorkflowState, finding: dict) -> list[dict]:
    records: list[dict] = []
    for raw_record in finding.get("source_inventory", []):
        record = dict(raw_record)
        record.setdefault("ticker", state["ticker"])
        record.setdefault("local_path", finding.get("artifact_path", ""))
        records.append(record)
    return records


async def run_neutral_research(state: WorkflowState) -> Command[Literal[AgentNode.SUPERVISOR]]:
    task_id = state.get("current_task_id", "")
    tasks = list(state.get("research_tasks", []))
    task = _task_by_id(tasks, task_id)
    if not task:
        logger.warning("[Research Analyst] No current task found; returning to supervisor")
        return Command(update={"current_task_id": ""}, goto=AgentNode.SUPERVISOR)

    research_input: ResearchTaskState = {
        "date": state["date"],
        "run_datetime": state.get("run_datetime", ""),
        "company": state["company"],
        "ticker": state["ticker"],
        "price_data": state["price_data"],
        "existing_valuation": state.get("valuation"),
        "research_goal": state.get("research_goal", ""),
        "task": task,
        "research_tasks": tasks,
        "source_inventory": state.get("source_inventory", []),
        "rag_context": state.get("rag_context", ""),
        "prior_findings": state.get("research_findings", []),
        "sources": [],
        "finding": {},
        "synthesis": "",
        "vault_artifacts": [],
        "active_memory_ids": [],
    }
    result = await run_research_task(research_input)
    findings = list(result.get("findings") or [])
    finding = result.get("finding", {})
    if not findings and finding:
        findings = [finding]
    status = "completed"
    if findings and all(
        item.get("needs_follow_up") and float(item.get("confidence") or 0) < 0.4
        for item in findings
    ):
        status = "blocked"

    update = {
        "research_tasks": _set_task_status(tasks, task_id, status),
        "current_task_id": "",
        "research_findings": findings,
    }
    if result.get("synthesis"):
        update["research_synthesis"] = result["synthesis"]
    if result.get("sources"):
        update["sources"] = result["sources"]
    if result.get("vault_artifacts"):
        update["vault_artifacts"] = result["vault_artifacts"]
    if result.get("active_memory_ids"):
        update["active_memory_ids"] = result["active_memory_ids"]
    inventory_records: list[dict] = []
    for item in findings:
        inventory_records.extend(_source_inventory_records_from_finding(state, item))
    if inventory_records:
        update["source_inventory"] = inventory_records

    try:
        run_dt = state.get("run_datetime", "")
        if run_dt and findings:
            writer = RunReportWriter(
                ticker=state["ticker"],
                run_datetime=run_dt,
                company=state.get("company", ""),
            )
            summaries = "\n\n".join(
                f"### {item.get('title') or item.get('task_id')}\n\n{item.get('summary', '')}"
                for item in findings
            )
            writer._append_file(
                str(writer.debug_path),
                (
                    f"\n## Research Task {task_id}: {task.get('title', '')}\n\n"
                    f"{summaries}\n\n"
                    "---\n"
                ),
            )
    except Exception as e:
        logger.warning("[Research Analyst] Failed to append task finding to run artifact: %s", e)

    return Command(update=update, goto=AgentNode.SUPERVISOR)

def build_research_workflow(checkpointer: AsyncPostgresSaver):
    """Main research workflow.

    Flow: START → Supervisor → Research Analyst → Judge (synthesise + valuate + reconcile)
          → Report Generator → Curator → END
    """
    store = InMemoryStore()
    return (
        StateGraph(WorkflowState)
        .add_node(AgentNode.SUPERVISOR, supervisor)
        .add_node(AgentNode.RESEARCH_ANALYST, run_neutral_research)
        .add_node(AgentNode.JUDGE, judge_analyst)
        .add_node(AgentNode.REPORT_GENERATOR, report_generator)
        .add_node(AgentNode.CURATOR, curator_agent)
        .add_edge(START, AgentNode.SUPERVISOR)
        # Supervisor uses Command(goto=...) for dynamic routing — no conditional edges needed
        # Judge routes directly to report_generator via Command
        .add_edge(AgentNode.REPORT_GENERATOR, AgentNode.CURATOR)
        .add_edge(AgentNode.CURATOR, END)
        .compile(checkpointer=checkpointer, store=store)
    )
