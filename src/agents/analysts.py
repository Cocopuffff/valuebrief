import asyncio
import json
import pathlib
import re
from dataclasses import dataclass
from langchain.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from schemas import AgentNode, ResearchTopic, ValuationModel, ResearchFinding, ResearchFindingBundle
from provider import FinancialDataProvider
from agents.states import ResearchState, ResearchTaskState
from utils.logger import get_logger, log_node_execution
from utils.config import (
    Provider,
    bear_model,
    bull_model,
    config,
    is_deepseek_reasoner_model,
    research_model,
)
from utils.research_persistence import memory_ids_from_artifact, persist_research_artifact

if TYPE_CHECKING:
    from langchain.agents.middleware.types import _InputAgentState

logger = get_logger(__name__)


@tool
async def search_investment_memory(ticker: str, query: str, limit: int = 5) -> str:
    """Search prior research memories for this ticker with a natural-language query.

    Use this to check if specific claims were made in previous analyses.
    Returns matching memory source excerpts with source metadata.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
        query: Natural-language search query
        limit: Max results (default 5)
    """
    try:
        from utils.embeddings import get_embedding
        from utils.vector_memory import format_memory_for_context, search_similar
        embedding = await get_embedding(query)
        results = await search_similar(
            embedding,
            ticker=ticker,
            limit=limit,
            exclude_validity_statuses=["contradicted", "superseded", "stale"],
        )
        if not results:
            return "No prior research memories found for this query."
        s = "y" if len(results) == 1 else "ies"
        lines = [f"Found {len(results)} prior research memor{s}:"]
        for i, r in enumerate(results, 1):
            lines.append(format_memory_for_context(r, i, source_excerpt_chars=1000))
        return "\n\n".join(lines)
    except Exception as e:
        return f"Error searching memories: {e}"


research_tools = [
    # FinancialDataProvider.get_asset_data,
    FinancialDataProvider.get_sec_filings,
    FinancialDataProvider.discover_earnings_call_transcripts,
    FinancialDataProvider.get_latest_news,
    FinancialDataProvider.search,
    FinancialDataProvider.scrape_website,
    search_investment_memory,
]

research_tools_by_name = {tool.name: tool for tool in research_tools}
bull_model_with_tools = bull_model.bind_tools(research_tools)
bear_model_with_tools = bear_model.bind_tools(research_tools)


URL_PATTERN = re.compile(r"https?://[^\s)\]>\"']+")
BUDGET_EXHAUSTED_MESSAGE = (
    "Research tool budget exhausted for {tool_name}. Do not call more tools. "
    "Synthesize the strongest answer possible from evidence already gathered."
)


@dataclass
class ResearchToolBudget:
    max_tool_calls: int
    max_search_calls: int
    max_scrape_calls: int
    tool_calls: int = 0
    search_calls: int = 0
    scrape_calls: int = 0

    def consume(self, tool_name: str, category: str = "tool") -> str | None:
        if (
            self.tool_calls >= self.max_tool_calls
            or category == "search" and self.search_calls >= self.max_search_calls
            or category == "scrape" and self.scrape_calls >= self.max_scrape_calls
        ):
            return BUDGET_EXHAUSTED_MESSAGE.format(tool_name=tool_name)

        self.tool_calls += 1
        if category == "search":
            self.search_calls += 1
        elif category == "scrape":
            self.scrape_calls += 1
        return None


def _research_budget_from_config() -> ResearchToolBudget:
    return ResearchToolBudget(
        max_tool_calls=config.research.max_tool_calls,
        max_search_calls=config.research.max_search_calls,
        max_scrape_calls=config.research.max_scrape_calls,
    )


def _extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    for match in URL_PATTERN.findall(text or ""):
        url = match.rstrip(".,;:")
        if url not in urls:
            urls.append(url)
    return urls


async def _ainvoke_base_tool(base_tool, args: dict[str, Any]) -> Any:
    return await base_tool.ainvoke(args)


def _build_research_tools(budget: ResearchToolBudget | None = None) -> list[Any]:
    if budget is None:
        return research_tools

    @tool("get_sec_filings")
    async def get_sec_filings(
        ticker: str,
        forms: list[str] | str | None = None,
        limit: int = 10,
    ) -> Any:
        """Fetch recent official SEC filings for a US-listed ticker."""
        exhausted = budget.consume("get_sec_filings")
        if exhausted:
            return exhausted
        return await _ainvoke_base_tool(
            FinancialDataProvider.get_sec_filings,
            {"ticker": ticker, "forms": forms, "limit": limit},
        )

    @tool("discover_earnings_call_transcripts")
    async def discover_earnings_call_transcripts(
        ticker: str,
        company: str = "",
        limit: int = 5,
    ) -> Any:
        """Discover earnings-call transcript sources using web search."""
        exhausted = budget.consume("discover_earnings_call_transcripts", "search")
        if exhausted:
            return exhausted
        return await _ainvoke_base_tool(
            FinancialDataProvider.discover_earnings_call_transcripts,
            {"ticker": ticker, "company": company, "limit": limit},
        )

    @tool("get_latest_news")
    async def get_latest_news(query: str) -> Any:
        """Gets latest news headlines with titles, URLs, and short snippets."""
        exhausted = budget.consume("get_latest_news", "search")
        if exhausted:
            return exhausted
        return await _ainvoke_base_tool(FinancialDataProvider.get_latest_news, {"query": query})

    @tool("search")
    async def search(query: str) -> Any:
        """Search the internet for information with titles, URLs, and snippets."""
        exhausted = budget.consume("search", "search")
        if exhausted:
            return exhausted
        return await _ainvoke_base_tool(FinancialDataProvider.search, {"query": query})

    @tool("scrape_website")
    async def scrape_website(url: str) -> Any:
        """Scrape a website for article content based on input URL."""
        exhausted = budget.consume("scrape_website", "scrape")
        if exhausted:
            return exhausted
        return await _ainvoke_base_tool(FinancialDataProvider.scrape_website, {"url": url})

    @tool("search_investment_memory")
    async def bounded_search_investment_memory(ticker: str, query: str, limit: int = 5) -> Any:
        """Search prior research memories for this ticker with a natural-language query."""
        exhausted = budget.consume("search_investment_memory")
        if exhausted:
            return exhausted
        return await _ainvoke_base_tool(
            search_investment_memory,
            {"ticker": ticker, "query": query, "limit": limit},
        )

    return [
        get_sec_filings,
        discover_earnings_call_transcripts,
        get_latest_news,
        search,
        scrape_website,
        bounded_search_investment_memory,
    ]


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEEPAGENT_SKILLS_ROOT = PROJECT_ROOT / ".deepagents" / "skills"


RESEARCH_ANALYST_SYSTEM = """You are a neutral expert stock research analyst.

Your job is to complete one assigned research task for a value-investing stock
research workflow. Be objective and do not adopt buy or sell bias. Use current
available information, primary sources when possible, and proper citations.

Use Deep Agents planning and scratch files to manage long source context. Use
skills when relevant. Do not write promotional copy. Do not calculate the final
DCF or final investment recommendation; the Judge owns valuation and synthesis.

Every major factual claim must cite a source URL or vault citation immediately
after the claim. If evidence is unavailable, say so directly.
"""

RESEARCH_FINDING_BUNDLE_SHAPE = """Expected ResearchFindingBundle shape:
{
  "findings": [
    {
      "task_id": "T1",
      "title": "Task title",
      "summary": "Concise cited finding for this task.",
      "key_points": ["Decision-relevant point with citation."],
      "source_urls": ["https://example.com/source"],
      "citations": ["optional-vault-file.md#^block-id"],
      "source_inventory": [
        {
          "ticker": "GNC",
          "source_type": "sec_filing",
          "title": "Form 10-Q",
          "form_type": "10-Q",
          "period": "2026-03-31",
          "filed_at": "2026-05-01",
          "url": "https://www.sec.gov/...",
          "accession_number": "0000000000-26-000001",
          "local_path": "",
          "freshness": "current"
        }
      ],
      "confidence": 0.8,
      "artifact_path": "",
      "memory_ids": [],
      "needs_follow_up": false,
      "follow_up_reason": ""
    }
  ],
  "synthesis": "Optional synthesis across findings."
}"""


# ── Context management helpers ────────────────────────────────────────────

def _trim_messages(messages: list, window_size: int = 20) -> list:
    """Sliding window to keep only the most recent messages during research.
    
    Prevents unbounded context growth across many iterations.
    Keeps the last `window_size` messages so the model has enough
    recent context to decide its next research step.
    """
    if len(messages) <= window_size:
        return messages
        
    trimmed = messages[-window_size:]
    
    # If the window sliced off an AIMessage but kept its ToolMessages, the 
    # API will reject the payload with a 400 error. We must drop any orphaned
    # ToolMessages at the start of our trimmed list to maintain a valid sequence.
    while trimmed and isinstance(trimmed[0], ToolMessage):
        trimmed.pop(0)
        
    return trimmed


def _prepare_messages_for_thesis(
    messages: list,
    max_tool_chars: int = 2000,
    max_total_chars: int = 50_000
) -> list:
    """Collapse the research message history into a valid sequence for thesis generation.

    Anthropic's API requires that every ToolMessage is preceded by an AIMessage
    that contains the matching tool_use block. Stripping tool_calls from AIMessages
    therefore creates orphaned ToolMessages that the API rejects as invalid.

    Instead, we walk the message list and group each (AIMessage w/ tool_calls) +
    its following ToolMessage(s) into a single HumanMessage that narrates the
    research round. This produces a clean, valid sequence the model can reason over.

    Plain AIMessages (the model's commentary without tool calls) are kept as-is.
    A total character budget is enforced, prioritising the most recent research.
    """
    # ── Pass 1: collapse tool-call rounds into HumanMessages ─────────────
    collapsed: list = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Gather this AI turn's preamble text + all immediately following ToolMessages
            parts: list[str] = []
            if msg.content:
                parts.append(f"Analyst reasoning: {msg.content}")

            tool_descriptions: list[str] = []
            for tc in msg.tool_calls:
                name = tc["name"]
                args = tc.get("args", {})
                if name == "scrape_website":
                    url = args.get("url", "") if isinstance(args, dict) else str(args)
                    tool_descriptions.append(f"scrape_website(url={url})")
                elif name == "search":
                    query = args.get("query", "") if isinstance(args, dict) else str(args)
                    tool_descriptions.append(f"search(query={query})")
                else:
                    tool_descriptions.append(name)
            parts.append(f"Tools called: {', '.join(tool_descriptions)}")

            i += 1
            while i < len(messages) and isinstance(messages[i], ToolMessage):
                tool_msg = messages[i]
                raw = tool_msg.content if isinstance(tool_msg.content, str) else str(tool_msg.content)
                if len(raw) > max_tool_chars:
                    raw = raw[:max_tool_chars] + "\n... [truncated]"
                parts.append(f"Research data:\n{raw}")
                i += 1

            collapsed.append(HumanMessage(content="\n\n".join(parts)))

        elif isinstance(msg, ToolMessage):
            # Orphaned ToolMessage (shouldn't happen, but handle gracefully)
            raw = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(raw) > max_tool_chars:
                raw = raw[:max_tool_chars] + "\n... [truncated]"
            collapsed.append(HumanMessage(content=f"Research data:\n{raw}"))
            i += 1

        else:
            # Plain AIMessage (model commentary) or other message types
            collapsed.append(msg)
            i += 1

    # ── Pass 2: enforce total character budget, keeping most recent first ─
    result: list = []
    total_chars = 0
    for msg in reversed(collapsed):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        msg_chars = len(content)
        if total_chars + msg_chars > max_total_chars:
            remaining = max_total_chars - total_chars
            if remaining > 500:
                truncated = content[:remaining] + "\n... [truncated for context limit]"
                result.append(type(msg)(content=truncated))
            break
        result.append(msg)
        total_chars += msg_chars

    result.reverse()
    logger.info(f"Context preparation: {len(result)}/{len(collapsed)} messages, ~{total_chars:,} chars")
    return result


# ── Content extraction ───────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Safely extract text from AIMessage.content.
    
    Anthropic thinking models via OpenRouter can return content as a list of 
    blocks like [{"type": "thinking", ...}, {"type": "text", "text": "..."}]
    instead of a plain string. This ensures we always get a string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Extract text from content blocks
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


# ── Deep Agents research wrapper ─────────────────────────────────────────

def _task_source_type(task: dict[str, Any]) -> str:
    kind = str(task.get("kind", "research")).strip() or "research"
    return {
        "source_discovery": "source_inventory",
        "business_model": "business_model_research",
        "financial_history": "financial_history_research",
        "risk_assessment": "risk_research",
        "pillar_validation": "pillar_validation_research",
        "research_synthesis": "research_synthesis",
    }.get(kind, f"{kind}_research")


def _finding_source_type(finding: ResearchFinding, default_task: dict[str, Any]) -> str:
    text = f"{finding.task_id} {finding.title}".lower()
    if any(token in text for token in ("source_inventory", "source discovery", "filing", "transcript")):
        return "source_inventory"
    if "business" in text or "moat" in text:
        return "business_model_research"
    if "financial" in text or "history" in text or "baseline" in text:
        return "financial_history_research"
    if "risk" in text:
        return "risk_research"
    if "dcf" in text or "valuation" in text:
        return "dcf_inputs_research"
    if "pillar" in text:
        return "pillar_candidate_research"
    return _task_source_type(default_task)


def _skill_markdown_paths(root: pathlib.Path = DEEPAGENT_SKILLS_ROOT) -> list[pathlib.Path]:
    """Return project Deep Agents skill files in deterministic order."""
    if not root.exists():
        return []
    return sorted(root.glob("*/SKILL.md"))


def _load_skill_files(create_file_data, root: pathlib.Path = DEEPAGENT_SKILLS_ROOT) -> dict[str, Any]:
    """Seed Deep Agents' StateBackend with project skills under /skills/."""
    files: dict[str, Any] = {}
    for path in _skill_markdown_paths(root):
        skill_name = path.parent.name
        files[f"/skills/{skill_name}/SKILL.md"] = create_file_data(
            path.read_text(encoding="utf-8")
        )
    return files


def _create_deep_research_agent(tools: list[Any] | None = None):
    """Create the Deep Agents harness lazily so tests can import without the package."""
    try:
        from deepagents import create_deep_agent
    except ImportError as e:
        raise RuntimeError(
            "deepagents is required for the neutral research analyst. "
            "Install project dependencies after pyproject.toml is updated."
        ) from e

    return create_deep_agent(
        model=research_model,
        tools=tools if tools is not None else research_tools,
        system_prompt=RESEARCH_ANALYST_SYSTEM,
        skills=["/skills/"],
        response_format=_deep_agent_response_format(),
        name="neutral-research-analyst",
    )


def _create_file_data_fn():
    try:
        from deepagents.backends.utils import create_file_data
    except ImportError as e:
        raise RuntimeError(
            "deepagents is required to seed research skills into the agent filesystem."
        ) from e
    return create_file_data


def _coerce_finding_bundle(raw: Any, task: dict[str, Any]) -> ResearchFindingBundle:
    """Validate Deep Agents structured output as a ResearchFindingBundle."""
    if isinstance(raw, ResearchFindingBundle):
        bundle = raw
    elif isinstance(raw, dict):
        bundle = ResearchFindingBundle.model_validate(raw)
    else:
        raise ValueError(
            "Deep Agent structured_response must be a ResearchFindingBundle "
            f"or dict, got {type(raw).__name__}.\n\n"
            f"{RESEARCH_FINDING_BUNDLE_SHAPE}"
        )

    if not bundle.findings:
        raise ValueError(
            "Deep Agent returned an empty findings list.\n\n"
            f"{RESEARCH_FINDING_BUNDLE_SHAPE}"
        )
    return bundle


def _is_source_inventory_finding(finding: ResearchFinding) -> bool:
    text = f"{finding.task_id} {finding.title}".lower()
    return any(token in text for token in ("source", "inventory", "filing", "transcript"))


def _normalize_finding_sources(finding: ResearchFinding) -> ResearchFinding:
    urls = set(finding.source_urls)
    for text in [finding.summary, *finding.key_points]:
        for url in _extract_urls(text):
            urls.add(url)
    for record in finding.source_inventory:
        if record.url:
            urls.add(record.url)
    finding.source_urls = sorted(list(urls))
    return finding


def _validate_finding_bundle(bundle: ResearchFindingBundle) -> ResearchFindingBundle:
    errors: list[str] = []
    for index, finding in enumerate(bundle.findings, 1):
        finding = _normalize_finding_sources(finding)
        label = finding.task_id or f"finding {index}"
        if not finding.summary.strip():
            errors.append(f"{label}: summary is empty")
        if not finding.key_points:
            errors.append(f"{label}: key_points must contain at least one item")
        if not finding.source_urls and not finding.citations:
            errors.append(f"{label}: source_urls or citations must contain at least one item")
        if _is_source_inventory_finding(finding) and not finding.source_inventory:
            errors.append(f"{label}: source_inventory must contain at least one structured record")

    if "Candidate Investment Pillars" not in bundle.synthesis:
        errors.append("synthesis must include a 'Candidate Investment Pillars' section")

    if errors:
        raise ValueError(
            "Deep Agent output failed semantic validation:\n- "
            + "\n- ".join(errors)
            + f"\n\n{RESEARCH_FINDING_BUNDLE_SHAPE}"
        )
    return bundle


def _deep_agent_response_format() -> type[ResearchFindingBundle] | None:
    """Return the Deep Agents response format supported by the configured model.

    LangChain implements structured output by adding a schema tool and forcing
    `tool_choice="required"`. DeepSeek's `deepseek-reasoner` rejects that
    parameter. DeepSeek thinking-mode tool calls also work best without forced
    `tool_choice`, so those models use prompt-shaped JSON plus local validation.
    """
    if (
        config.research.provider == Provider.DEEPSEEK
        and (is_deepseek_reasoner_model(config.research.model) or config.research.thinking)
    ):
        return None
    return ResearchFindingBundle


def _parse_json_object(text: str) -> Any:
    """Parse a JSON object from model text, allowing a single fenced block."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = stripped.find("{")
        if start == -1:
            raise
        parsed, _ = decoder.raw_decode(stripped[start:])
        return parsed


def _extract_deep_agent_output(result: dict[str, Any], task: dict[str, Any]) -> ResearchFindingBundle:
    if not isinstance(result, dict):
        raise ValueError(
            f"Deep Agent result must be a dict, got {type(result).__name__}.\n\n"
            f"{RESEARCH_FINDING_BUNDLE_SHAPE}"
        )

    structured = result.get("structured_response")
    if structured is not None:
        return _coerce_finding_bundle(structured, task)

    messages = result.get("messages", [])
    if messages:
        try:
            final_text = _extract_text(messages[-1].content)
            parsed = _parse_json_object(final_text)
        except (AttributeError, TypeError, json.JSONDecodeError):
            pass
        else:
            return _coerce_finding_bundle(parsed, task)

    raise ValueError(
        "Deep Agent did not return structured_response matching "
        f"ResearchFindingBundle.\n\n{RESEARCH_FINDING_BUNDLE_SHAPE}"
    )


def _summarize_deep_agent_output(result: Any, max_chars: int = 4000) -> str:
    """Build a compact malformed-output preview for retry feedback."""
    if not isinstance(result, dict):
        return str(result)[:max_chars]

    structured = result.get("structured_response", "<missing>")
    messages = result.get("messages", [])
    final_text = ""
    if messages:
        try:
            final_text = _extract_text(messages[-1].content)
        except Exception:
            final_text = str(messages[-1])

    preview = json.dumps(
        {
            "structured_response": structured,
            "final_message": final_text,
        },
        default=str,
        indent=2,
    )
    return preview[:max_chars]


def _format_retry_prompt(base_prompt: str, error: Exception, malformed_output: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "Your previous response was malformed and could not be parsed as a "
        "ResearchFindingBundle. Retry the same task and return valid structured "
        "output only.\n\n"
        f"Validation error:\n{error}\n\n"
        f"Malformed output preview:\n{malformed_output}"
    )


def _fallback_finding_from_error(
    task: dict[str, Any],
    error: Exception | None,
    diagnostic: str = "",
) -> ResearchFinding:
    if isinstance(error, TimeoutError):
        summary = "Research task timed out before producing valid structured output."
        reason = f"Deep Agent timed out after {config.research.timeout_seconds} seconds."
    else:
        summary = "Research task did not produce valid structured output after retries."
        reason = (
            "Deep Agent output parsing failed after retries"
            + (f": {error}" if error else ".")
        )
    if diagnostic:
        reason = f"{reason}\n\nDiagnostic preview:\n{diagnostic}"
    return ResearchFinding(
        task_id=str(task.get("task_id", "")),
        title=str(task.get("title", "")),
        summary=summary,
        confidence=0.1,
        needs_follow_up=True,
        follow_up_reason=reason,
    )


def _format_task_prompt(state: ResearchTaskState) -> str:
    task = state["task"]
    if task.get("task_id") == "deep_agent_research":
        return (
            f"Research goal:\n{state.get('research_goal') or 'Create objective value-investing stock research.'}\n\n"
            f"Company: {state.get('company')} ({state.get('ticker')})\n"
            f"Date: {state.get('date')}\n\n"
            f"Known source inventory:\n{json.dumps(state.get('source_inventory', []), indent=2, default=str)}\n\n"
            f"Prior pillar context:\n{state.get('rag_context') or '(No prior pillars retrieved.)'}\n\n"
            "Use Deep Agents' todo list to plan this single bounded research run. "
            "Do not wait for an external task queue. Complete all deliverables below "
            "within this one run and then synthesize.\n\n"
            "Required deliverables as ResearchFindingBundle findings:\n"
            "1. task_id='source_inventory': latest SEC filings, earnings releases, "
            "and transcripts. Include at least one structured source_inventory record.\n"
            "2. task_id='business_model': business model, moat, customers, market position.\n"
            "3. task_id='financial_history': revenue, margins, capex, debt, free cash flow baseline.\n"
            "4. task_id='risk_assessment': company-specific and industry risks.\n"
            "5. task_id='dcf_inputs': evidence for DCF assumptions; do not calculate final intrinsic value.\n"
            "6. task_id='candidate_pillars': candidate investment pillars, but do not assign stable pillar IDs.\n\n"
            "Every finding must include non-empty summary, key_points, and source_urls "
            "or citations. Every major factual claim must cite a URL or vault citation. "
            "Prefer primary sources and scrape only the highest-value pages.\n\n"
            "The synthesis string must include a Markdown section titled exactly "
            "'Candidate Investment Pillars'.\n\n"
            "Return only a JSON object matching this exact ResearchFindingBundle shape:\n"
            f"{RESEARCH_FINDING_BUNDLE_SHAPE}"
        )

    return (
        f"Research goal:\n{state.get('research_goal') or 'Create objective value-investing stock research.'}\n\n"
        f"Company: {state.get('company')} ({state.get('ticker')})\n"
        f"Date: {state.get('date')}\n\n"
        f"Current task JSON:\n{json.dumps(task, indent=2, default=str)}\n\n"
        f"All task statuses:\n{json.dumps(state.get('research_tasks', []), indent=2, default=str)}\n\n"
        f"Known source inventory:\n{json.dumps(state.get('source_inventory', []), indent=2, default=str)}\n\n"
        f"Prior pillar context:\n{state.get('rag_context') or '(No prior pillars retrieved.)'}\n\n"
        f"Prior completed findings:\n{json.dumps(state.get('prior_findings', []), indent=2, default=str)}\n\n"
        "Complete the current task only. Return structured output matching the "
        "ResearchFindingBundle schema. If the runtime does not provide a "
        "structured-output tool, return only a JSON object with the exact "
        "ResearchFindingBundle shape and no surrounding prose."
    )


def _finding_to_markdown(finding: ResearchFinding) -> str:
    key_points = "\n".join(f"- {point}" for point in finding.key_points) or "- None recorded."
    source_urls = "\n".join(f"- {url}" for url in finding.source_urls) or "- None recorded."
    citations = "\n".join(f"- {citation}" for citation in finding.citations) or "- None recorded."
    inventory = "\n".join(
        f"- {record.source_type} {record.form_type} {record.filed_at}: {record.url}".strip()
        for record in finding.source_inventory
    ) or "- None recorded."
    return (
        f"# Research Task {finding.task_id}: {finding.title or 'Finding'}\n\n"
        f"## Summary\n\n{finding.summary}\n\n"
        f"## Key Points\n\n{key_points}\n\n"
        f"## Source URLs\n\n{source_urls}\n\n"
        f"## Citations\n\n{citations}\n\n"
        f"## Source Inventory\n\n{inventory}\n\n"
        f"## Confidence\n\n{finding.confidence:.2f}\n"
    )


@log_node_execution
async def run_research_task(state: ResearchTaskState) -> dict[str, Any]:
    """Run one supervisor-assigned neutral research task with Deep Agents."""
    task = state["task"]
    logger.info(
        "[Research Analyst] Running task %s: %s",
        task.get("task_id"),
        task.get("title"),
    )

    budget = _research_budget_from_config()
    agent = _create_deep_research_agent(tools=_build_research_tools(budget))
    files = _load_skill_files(_create_file_data_fn())
    base_prompt = _format_task_prompt(state)
    max_retries = 1
    last_error: Exception | None = None
    bundle: ResearchFindingBundle | None = None
    last_result: Any = None
    run_config: RunnableConfig = {
        "configurable": {
            "thread_id": f"{state.get('ticker')}-{task.get('task_id')}-{state.get('date')}"
        },
        "recursion_limit": config.research.max_iterations,
    }

    for attempt in range(max_retries + 1):
        prompt = base_prompt
        if attempt > 0 and last_error is not None:
            prompt = _format_retry_prompt(
                base_prompt,
                last_error,
                _summarize_deep_agent_output(last_result),
            )

        try:
            agent_input = cast(
                "_InputAgentState",
                {
                    "messages": [HumanMessage(content=prompt)],
                    "files": files,
                },
            )
            last_result = await asyncio.wait_for(
                agent.ainvoke(
                    agent_input,
                    config=run_config,
                ),
                timeout=config.research.timeout_seconds,
            )
            bundle = _validate_finding_bundle(
                _extract_deep_agent_output(last_result, task)
            )
            break
        except TimeoutError as e:
            last_error = e
            logger.error(
                "[Research Analyst] Deep Agent timed out after %s seconds for task %s",
                config.research.timeout_seconds,
                task.get("task_id"),
            )
            break
        except (TypeError, ValueError) as e:
            last_error = e
            logger.warning(
                "[Research Analyst] Structured output attempt %d failed for task %s: %s",
                attempt + 1,
                task.get("task_id"),
                e,
            )

    if bundle is None:
        logger.error(
            "[Research Analyst] Structured output failed after %d attempts for task %s: %s",
            max_retries + 1,
            task.get("task_id"),
            last_error,
        )
        bundle = ResearchFindingBundle(
            findings=[
                _fallback_finding_from_error(
                    task,
                    last_error,
                    _summarize_deep_agent_output(last_result) if last_result is not None else "",
                )
            ],
            synthesis="",
        )

    update: dict[str, Any] = {
        "findings": [],
        "finding": {},
        "synthesis": bundle.synthesis,
        "sources": sorted({url for finding in bundle.findings for url in finding.source_urls}),
        "vault_artifacts": [],
        "active_memory_ids": [],
    }

    persisted_findings: list[dict[str, Any]] = []
    for finding in bundle.findings:
        if not finding.task_id:
            finding.task_id = str(task.get("task_id", ""))
        if not finding.title:
            finding.title = str(task.get("title", ""))

        try:
            artifact = await persist_research_artifact(
                ticker=state["ticker"],
                content=_finding_to_markdown(finding),
                source_type=_finding_source_type(finding, task),
                source_priority=1,
                vectorize=True,
                metadata={
                    "agent": "neutral_research_analyst",
                    "company": state.get("company", ""),
                    "run_datetime": state.get("run_datetime", ""),
                    "task_id": finding.task_id,
                    "task_kind": task.get("kind", ""),
                    "source_urls": sorted(set(finding.source_urls)),
                    "source_inventory_records": [
                        record.model_dump(mode="json")
                        for record in finding.source_inventory
                    ],
                },
            )
            if artifact.path:
                finding.artifact_path = artifact.path
                finding.memory_ids = memory_ids_from_artifact(artifact)
                update["vault_artifacts"].append(artifact.model_dump(mode="json"))
                update["active_memory_ids"].extend(memory_ids_from_artifact(artifact))
        except Exception as e:
            logger.warning("[Research Analyst] Failed to persist task finding: %s", e)

        persisted_findings.append(finding.model_dump(mode="json"))

    update["findings"] = persisted_findings
    update["finding"] = persisted_findings[0] if persisted_findings else {}

    return update


# ── Prior valuation context helper ───────────────────────────────────────

def _build_prior_valuation_context(valuation: ValuationModel | None) -> str:
    """Build a concise summary of a prior valuation for analyst prompts.
    
    Returns an empty string if no prior valuation exists, so the prompt
    reads cleanly without a placeholder block.
    """
    if not valuation:
        return ""

    from datetime import datetime, timezone
    age_str = "unknown age"
    if valuation.updated_at:
        delta = datetime.now(timezone.utc) - valuation.updated_at
        days = delta.days
        if days == 0:
            age_str = "updated today"
        elif days == 1:
            age_str = "1 day old"
        else:
            age_str = f"{days} days old"

    scenario_lines = []
    for label, s in valuation.scenarios.items():
        iv = f"${s.intrinsic_value:,.2f}" if s.intrinsic_value else "N/A"
        scenario_lines.append(f"  - {label} ({s.probability:.0%}): IV = {iv}")
    scenarios_block = "\n".join(scenario_lines)

    cagr = f"{valuation.expected_cagr:.1%}" if valuation.expected_cagr else "N/A"

    return (
        f"\n\nPRIOR VALUATION ({age_str}, recommendation: {valuation.recommendation}):\n"
        f"Expected Intrinsic Value: ${valuation.expected_value:,.2f} vs Current Price: ${valuation.current_price:,.2f}\n"
        f"Expected CAGR: {cagr}\n"
        f"Scenarios:\n{scenarios_block}\n"
        f"Use this as context — challenge or confirm these assumptions with your research."
    )


# ── Logging helpers ──────────────────────────────────────────────────────

def _log_response(label: str, response: AIMessage) -> None:
    """Log the model's thoughts and planned tool calls."""
    if response.content:
        logger.info(f"[{label}] 💭 Thinking:\n{response.content}")
    if response.tool_calls:
        tools_planned = [f"  → {tc['name']}({', '.join(f'{k}={v!r}' for k, v in tc['args'].items())})" for tc in response.tool_calls]
        logger.info(f"[{label}] 🔧 Tool calls planned:\n" + '\n'.join(tools_planned))


def _topic_value(topic: ResearchTopic | dict[str, Any], key: str, default: Any = "") -> Any:
    if isinstance(topic, dict):
        return topic.get(key, default)
    return getattr(topic, key, default)


def _format_topics(topics: Sequence[ResearchTopic | dict[str, Any]]) -> str:
    """Format ResearchTopic items into a Markdown prompt section."""
    if not topics:
        return "No prior research topics assigned."
    lines = [
        "## Prior Research Topics",
        "Prioritize investigating these in your research:",
        "",
    ]
    for i, t in enumerate(topics, 1):
        question = _topic_value(t, "question")
        rationale = _topic_value(t, "rationale")

        lines.append(f"{i}. {question}")
        if rationale:
            lines.append(f"   Rationale: {rationale}")
    return "\n".join(lines)


_SOURCE_URL_RULES = """
SOURCE CITATION RULES (MANDATORY):
- When you make a major factual claim in your thesis, you MUST attach the source URL
  immediately after the claim in parentheses.
  Example: "Salesforce grew revenue 12% YoY (https://example.com/earnings-report)"
- You are researching prior thesis pillars shown in the RAG context. Your job is to
  provide fresh evidence that supports, weakens, contradicts, or revises each pillar.
- Do NOT assign final memory outcomes (supported/weakened/revised/contradicted/stale).
  The Judge will evaluate all evidence and make those determinations.
- Focus on finding current, verifiable sources for every major claim.
"""


# ── Analyst nodes ─────────────────────────────────────────────────────────

BULL_SYSTEM = """You are a bullish equity analyst grounded in value investing principles. You are researching {company} ({ticker}) to build the strongest possible case for buying it. Today is {date}. You have {remaining} research iterations remaining.

RESEARCH METHODOLOGY — follow this workflow each iteration:
1. Call search or get_latest_news to find relevant articles, reports, and analysis.
2. Review the returned titles and snippets to identify the most relevant URLs.
3. Call scrape_website on 2-3 of the best URLs to get full article content.
4. Synthesize what you've learned so far.

IMPORTANT: search and get_latest_news only return headlines and short snippets. You MUST call scrape_website on promising URLs to get the actual article content needed for a serious thesis.
{prior_valuation}
{rag_context}
{topics}
{source_rules}
Your thesis should address: margin of safety, intrinsic value drivers, competitive moat, growth catalysts, and risk/reward asymmetry. When referencing prior thesis pillars, provide evidence that supports or challenges each one."""

BEAR_SYSTEM = """You are a bearish equity analyst grounded in value investing principles. You are researching {company} ({ticker}) to build the strongest possible case for selling or avoiding it. Today is {date}. You have {remaining} research iterations remaining.

RESEARCH METHODOLOGY — follow this workflow each iteration:
1. Call search or get_latest_news to find relevant articles, reports, and analysis.
2. Review the returned titles and snippets to identify the most relevant URLs.
3. Call scrape_website on 2-3 of the best URLs to get full article content.
4. Synthesize what you've learned so far.

IMPORTANT: search and get_latest_news only return headlines and short snippets. You MUST call scrape_website on promising URLs to get the actual article content needed for a serious thesis.
{prior_valuation}
{rag_context}
{topics}
{source_rules}
Your thesis should address: overvaluation risks, margin of safety concerns, competitive threats, earnings quality issues, and downside catalysts. When referencing prior thesis pillars, provide evidence that supports or challenges each one."""


@log_node_execution
async def bull_analyst(state: ResearchState) -> dict:
    """Bull analyst: researches with tools, then produces a final thesis."""
    iteration = state['iteration_count']
    remaining = state['max_iterations'] - iteration
    logger.info(f"[Bull Analyst] Iteration {iteration} ({remaining} remaining)")

    if iteration < state['max_iterations']:
        prior_valuation_ctx = _build_prior_valuation_context(state.get('existing_valuation'))
        rag_ctx = state.get('rag_context', '') or '(No prior research context — fresh analysis.)'
        topics_text = _format_topics(
            state.get('research_topics', [])  # already filtered by side in orchestration
        )
        system = BULL_SYSTEM.format(
            company=state['company'], ticker=state['ticker'],
            date=state['date'], remaining=remaining,
            prior_valuation=prior_valuation_ctx,
            rag_context=rag_ctx,
            topics=topics_text,
            source_rules=_SOURCE_URL_RULES if state.get('rag_context') else '',
        )
        # Trim context to avoid unbounded growth
        trimmed = _trim_messages(state['messages'])
        response = await bull_model_with_tools.ainvoke(
            [SystemMessage(content=system)] + trimmed
        )
        _log_response("Bull Analyst", response)
        result = {"messages": [response], "iteration_count": 1}
        # If the model chose to write analysis instead of calling tools,
        # capture it as the thesis now — analyst_router will route to END
        if not response.tool_calls and response.content:
            result["thesis"] = _extract_text(response.content)
        elif response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "scrape_website":
                    url_value = tool_call["args"]
                    url_str = None
                    if isinstance(url_value, str):
                        url_str = url_value
                    elif isinstance(url_value, dict) and "url" in url_value:
                        url_str = url_value["url"]
                    if url_str:
                        state["sources"].append(url_str)
        return result
    else:
        # Final iteration — produce thesis from all collected research
        logger.info(f"[Bull Analyst] Writing final thesis")
        thesis_messages = _prepare_messages_for_thesis(state['messages'])
        final_prompt = f"""You are a bullish equity analyst. You have completed all your research on {state['company']} ({state['ticker']}). Today is {state['date']}.

Based on ALL the research data in the conversation above, write your final bull thesis.

Your thesis MUST cover:
- Margin of safety analysis
- Intrinsic value drivers
- Competitive moat assessment
- Growth catalysts
- Risk/reward asymmetry

SOURCE CITATION: Attach the source URL in parentheses after every major factual claim.
Example: "The company grew revenue 12% YoY (https://example.com/report)."

Your thesis MUST adhere strictly to standard Markdown formatting rules:
- All headings MUST be surrounded by blank lines.
- All lists MUST be surrounded by blank lines.
- Always use exactly one space after a list marker (e.g., '- item', not '-  item').

Write a structured, substantive investment thesis. Do NOT attempt to call any tools."""
        response: AIMessage = await bull_model.ainvoke(
            [SystemMessage(content=final_prompt)] + thesis_messages
        )
        _log_response("Bull Analyst (Final)", response)
        return {
            "messages": [response],
            "thesis": _extract_text(response.content),
            "iteration_count": 1
        }


@log_node_execution
async def bear_analyst(state: ResearchState) -> dict:
    """Bear analyst: researches with tools, then produces a final thesis."""
    iteration = state['iteration_count']
    remaining = state['max_iterations'] - iteration
    logger.info(f"[Bear Analyst] Iteration {iteration} ({remaining} remaining)")

    if iteration < state['max_iterations']:
        prior_valuation_ctx = _build_prior_valuation_context(state.get('existing_valuation'))
        rag_ctx = state.get('rag_context', '') or '(No prior research context — fresh analysis.)'
        topics_text = _format_topics(
            state.get('research_topics', [])
        )
        system = BEAR_SYSTEM.format(
            company=state['company'], ticker=state['ticker'],
            date=state['date'], remaining=remaining,
            prior_valuation=prior_valuation_ctx,
            rag_context=rag_ctx,
            topics=topics_text,
            source_rules=_SOURCE_URL_RULES if state.get('rag_context') else '',
        )
        trimmed = _trim_messages(state['messages'])
        response = await bear_model_with_tools.ainvoke(
            [SystemMessage(content=system)] + trimmed
        )
        _log_response("Bear Analyst", response)
        result = {"messages": [response], "iteration_count": 1}
        if not response.tool_calls and response.content:
            result["thesis"] = _extract_text(response.content)
        elif response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "scrape_website":
                    url_value = tool_call["args"]
                    url_str = None
                    if isinstance(url_value, str):
                        url_str = url_value
                    elif isinstance(url_value, dict) and "url" in url_value:
                        url_str = url_value["url"]
                    if url_str:
                        state["sources"].append(url_str)
        return result
    else:
        logger.info(f"[Bear Analyst] Writing final thesis")
        thesis_messages = _prepare_messages_for_thesis(state['messages'])
        final_prompt = f"""You are a bearish equity analyst. You have completed all your research on {state['company']} ({state['ticker']}). Today is {state['date']}.

Based on ALL the research data in the conversation above, write your final bear thesis.

Your thesis MUST cover:
- Overvaluation risks
- Margin of safety concerns
- Competitive threats
- Earnings quality issues
- Downside catalysts

SOURCE CITATION: Attach the source URL in parentheses after every major factual claim.
Example: "The company faces increasing competition (https://example.com/analysis)."

Your thesis MUST adhere strictly to standard Markdown formatting rules:
- All headings MUST be surrounded by blank lines.
- All lists MUST be surrounded by blank lines.
- Always use exactly one space after a list marker (e.g., '- item', not '-  item').

Write a structured, substantive investment thesis. Do NOT attempt to call any tools."""
        response = await bear_model.ainvoke(
            [SystemMessage(content=final_prompt)] + thesis_messages
        )
        _log_response("Bear Analyst (Final)", response)
        return {
            "messages": [response],
            "thesis": _extract_text(response.content),
            "iteration_count": 1
        }


# ── Tool execution & routing ─────────────────────────────────────────────

async def research_tool_node(state: ResearchState) -> dict[str, Any]:
    """Executes tool calls from the last AI message."""
    result = []
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage):
        for tool_call in last_message.tool_calls:
            tool = research_tools_by_name[tool_call["name"]]
            observation = await tool.ainvoke(tool_call["args"])
            if not isinstance(observation, str):
                observation = json.dumps(observation, default=str)
            result.append(ToolMessage(content=observation, 
                                      tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: ResearchState) -> Literal[AgentNode.RESEARCH_TOOL_NODE, AgentNode.SUPERVISOR]:
    """Route: if the model made tool calls, execute them; otherwise return to supervisor."""
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            return AgentNode.RESEARCH_TOOL_NODE

    return AgentNode.SUPERVISOR
