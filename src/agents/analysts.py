from schemas import ResearchTopic
import json
from langchain.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain.tools import tool
from typing import Any, Literal

from schemas import AgentNode, ValuationModel
from provider import FinancialDataProvider
from agents.states import ResearchState
from utils.logger import get_logger, log_node_execution
from utils.config import bull_model, bear_model

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
    # FinancialDataProvider.get_sec_filings,
    FinancialDataProvider.get_latest_news,
    FinancialDataProvider.search,
    FinancialDataProvider.scrape_website,
    search_investment_memory,
]

research_tools_by_name = {tool.name: tool for tool in research_tools}
bull_model_with_tools = bull_model.bind_tools(research_tools)
bear_model_with_tools = bear_model.bind_tools(research_tools)


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


def _format_topics(topics: list[ResearchTopic | dict[str, Any]]) -> str:
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
