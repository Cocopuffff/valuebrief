import json
import re
from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import Command
from typing import Literal
from agents.states import WorkflowState
from schemas import ValuationModel, AgentNode
from utils.logger import get_logger, log_node_execution
from utils.report_writer import RunReportWriter
from utils.research_persistence import memory_ids_from_artifact, persist_research_artifact
from utils.config import judge_model, valuation_model

logger = get_logger(__name__)


# ── Prompts ──────────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM = """\
Act as an expert stock research analyst grounded in value investing principles. \
You are presented with bull and bear theses for {company} ({ticker}) \
and you are tasked with synthesizing both sides' research to create an investment thesis \
for {company}. This research should take an opinionated view based on the \
evidence presented by the bull and bear analysts and should include the hallmarks of \
value investing like discounted cash flow based on probability of outcomes and \
investability based on margin of safety. Present this information as your own analysis \
without naming bull or bear parties. Do NOT call any tools. Respond with plain text only.\
"""

VALUATION_SYSTEM = """\
You are a quantitative valuation analyst. Your job is to derive a three-scenario \
DCF (Discounted Cash Flow) valuation for {company} ({ticker}).

You will receive:
- Company fundamentals (revenue, margins, growth, FCF, etc.)
- A synthesis from the lead analyst summarising the investment thesis
- Optionally, a PRIOR VALUATION with previously derived DCF assumptions and results

If a prior valuation is provided, use it as a starting point — adjust assumptions \
based on new information from the synthesis rather than deriving from scratch. \
If circumstances have materially changed, you may deviate significantly.

Using this information, you must produce a JSON object that matches the schema below EXACTLY.

### Rules for your assumptions
- **Bear scenario**: assumes the downside risks materialise. Lower growth, margin compression, higher WACC. Probability should reflect your confidence (typically 0.20-0.35).
- **Base scenario**: a balanced "most likely" case. Moderate growth and margins near current levels. Probability typically 0.35-0.50.
- **Bull scenario**: assumes the upside catalysts materialise. Higher growth, margin expansion, lower WACC. Probability typically 0.20-0.35.
- Probabilities MUST sum to 1.0.
- `revenue_growth_stage_1` (years 1-5) and `revenue_growth_stage_2` (years 6-10) are annualized rates (e.g. 0.10 = 10%).
- `ebit_margin_target` is the target operating margin (e.g. 0.25 = 25%).
- `tax_rate` should be between 0.15 and 0.30 (default 0.21).
- `wacc` should be between 0.06 and 0.20.
- `terminal_growth` should be between 0.01 and 0.04 (never exceed WACC).
- For `intrinsic_value` in each scenario, set it to `null` — it will be computed programmatically.

### Required JSON schema

```json
{{
  "ticker": "{ticker}",
  "company": "{company}",
  "scenarios": {{
    "Bear": {{
      "label": "Bear",
      "probability": <float 0-1>,
      "assumptions": {{
        "revenue_growth_stage_1": <float>,
        "revenue_growth_stage_2": <float>,
        "ebit_margin_target": <float>,
        "tax_rate": <float>,
        "wacc": <float>,
        "terminal_growth": <float>
      }},
      "intrinsic_value": null
    }},
    "Base": {{ ... same structure ... }},
    "Bull": {{ ... same structure ... }}
  }}
}}
```

Respond ONLY with the JSON object. No commentary, no markdown fences, no explanation.\
"""

RECONCILE_SYSTEM = """\
You are the lead investment analyst for {company} ({ticker}). This is a DAILY \
investment update. You must accumulate, prune, and evolve the investment thesis over time.

You will receive:
1. **Prior Investment Thesis**: Your final decision from the previous update (if any). \
This is your FOUNDATION.
2. **Prior Thesis Pillars**: Active investment pillars from the previous run \
(if any). Each pillar has a prior_ref, pillar_id, memory_id, version, statement, current status, and dossier citation.
3. **New Qualitative Synthesis**: A summary of today's bull and bear arguments.
4. **New DCF Valuation Results**: The updated quantitative realities.

Your job is to produce an UPDATED investment thesis that reconciles any tension \
between the qualitative sentiment and the quantitative reality.

Structure your response in TWO sections separated by the delimiter ``----JSON----``:

**Section 1 — Investment Thesis (plain text):**
1. **Verdict** — one sentence.
2. **Rationale** — 2-3 paragraphs integrating prior thesis with new insights.
3. **Key Risks** — bullet list.

**Section 2 — Thesis Pillars & Outcomes (JSON after the delimiter):**
Produce a JSON object with two keys:

- ``thesis_pillars``: Array of current thesis pillar candidates derived from the reconciled \
investment thesis. Each pillar must have:
  * ``candidate_ref``: stable local reference you create in this response (e.g. "C1")
  * ``matched_prior_ref``: prior_ref if this candidate is the same/revised prior pillar, empty string if new
  * ``matched_pillar_id``: prior pillar_id if this candidate is the same/revised prior pillar, empty string if new
  * ``pillar_type``: one of "moat", "growth", "risk", "valuation_assumption", "capital_allocation", "thesis_change"
  * ``statement``: the core claim (1-2 sentences)
  * ``rationale``: supporting evidence (1-3 sentences)
  * ``valuation_impact``: how this affects intrinsic value (1 sentence)
  * ``source_urls``: list of URLs from analyst research that support this pillar
  * ``evidence_citations``: list of vault block citations (e.g. "file.md#^blockid")
  * ``resurrection_reason``: required only if reviving a previously contradicted/stale/superseded idea
  * ``status``: "supported" or "weakened" for current active pillars

Do not create stable pillar IDs for new pillars. The system assigns them deterministically.

- ``pillar_outcomes``: Array evaluating each prior pillar you received. Each outcome must have:
  * ``memory_id``: the UUID of the prior pillar memory
  * ``pillar_id``: matches the prior pillar's pillar_id
  * ``status``: "supported" | "weakened" | "revised" | "contradicted" | "stale"
  * ``reason``: why this status was assigned (1-2 sentences)
  * ``replacement_statement``: if "revised", the new statement (empty string otherwise)
  * ``source_urls``: list of URLs that support the evaluation

- ``valuation_impact`` (optional): brief summary of how the DCF results inform the thesis.

CRITICAL: The JSON must be valid. Do NOT wrap it in markdown fences after the delimiter.\
"""


# ── Helpers (moved from valuation.py) ────────────────────────────────────

def _build_fundamentals_summary(state: WorkflowState) -> str:
    """Build a human-readable summary of the Asset fundamentals for the valuation prompt."""
    asset = state.get("price_data")
    if not asset:
        return "No fundamental data available."

    f = asset.fundamentals
    lines = [
        f"Current Price: ${asset.current_price:,.2f}",
        f"Shares Outstanding: {asset.shares_outstanding/1000000:,.1f} (in millions)" if asset.shares_outstanding else None,
        f"Market Cap: ${f.market_cap/1000000:,.1f} (in millions)" if f.market_cap else None,
        f"Total Revenue (TTM): ${f.total_revenue/1000000:,.1f} (in millions)" if f.total_revenue else None,
        f"Revenue Growth (YoY): {f.revenue_growth:.1%}" if f.revenue_growth is not None else None,
        f"EBITDA Margin: {f.ebitda_margin:.1%}" if f.ebitda_margin is not None else None,
        f"Free Cash Flow: ${f.free_cash_flow/1000000:,.1f} (in millions)" if f.free_cash_flow else None,
        f"Trailing P/E: {f.pe_ratio:.1f}" if f.pe_ratio else None,
        f"Forward P/E: {f.forward_pe_ratio:.1f}" if f.forward_pe_ratio else None,
        f"PEG Ratio: {f.peg_ratio:.2f}" if f.peg_ratio else None,
        f"Price/Book: {f.price_to_book:.2f}" if f.price_to_book else None,
        f"Debt/Equity: {f.debt_to_equity:.1f}" if f.debt_to_equity else None,
        f"Return on Equity: {f.return_on_equity:.1%}" if f.return_on_equity is not None else None,
        f"Current Ratio: {f.current_ratio:.2f}" if f.current_ratio else None,
        f"EPS (Trailing): ${f.earnings_per_share:.2f}" if f.earnings_per_share else None,
        f"Dividend Yield: {f.dividend_yield:.2%}" if f.dividend_yield else None,
        f"Sector: {asset.sector}" if asset.sector else None,
        f"Industry: {asset.industry}" if asset.industry else None,
    ]
    return "\n".join(line for line in lines if line)


def _parse_valuation_response(raw: str, state: WorkflowState) -> ValuationModel:
    """Parse the LLM's JSON response into a ValuationModel, with fallback cleanup."""
    # Strip markdown code fences if the model wrapped the JSON
    text = raw.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3].rstrip()

    data = json.loads(text)

    # Ensure required fields are present from state if model omitted them
    asset = state.get("price_data")
    if "current_price" not in data and asset:
        data["current_price"] = asset.current_price
    if "base_revenue" not in data:
        if asset and getattr(asset.fundamentals, "total_revenue", None):
            data["base_revenue"] = asset.fundamentals.total_revenue
        else:
            data["base_revenue"] = 1.0 # default to 1 to avoid zero division breakdown
    if "shares_outstanding" not in data and asset and asset.shares_outstanding:
        data["shares_outstanding"] = asset.shares_outstanding

    return ValuationModel(**data)


def _build_dcf_summary(valuation: ValuationModel) -> str:
    """Build a Markdown-formatted summary of DCF results.
    
    Returns a string with:
    - A scenario table (probability, intrinsic value, margin of safety)
    - A DCF year-by-year projections table per scenario
    - Summary metrics (expected value, CAGR, recommendation)
    """
    cagr = valuation.expected_cagr
    cagr_str = f"{cagr:.1%}" if cagr is not None else "N/A"

    # ── Scenario summary table ───────────────────────────────────────────
    scenario_rows = []
    for label, s in valuation.scenarios.items():
        margin = valuation.scenario_margins.get(label)
        margin_str = f"{margin:.1%}" if margin is not None else "N/A"
        iv_str = f"${s.intrinsic_value:,.2f}" if s.intrinsic_value is not None else "N/A"
        scenario_rows.append(
            f"| {label} | {s.probability:.0%} | {iv_str} | {margin_str} |"
        )

    scenario_table = (
        "| Scenario | Probability | Intrinsic Value | Margin of Safety |\n"
        "| ---------- | ------------- | ----------------- | ------------------ |\n"
        + "\n".join(scenario_rows)
    )

    # ── Per-scenario DCF projection tables ──────────────────────────────
    projection_blocks: list[str] = []
    for label, s in valuation.scenarios.items():
        if not s.dcf_table:
            continue
        
        rows = [
            f"| {row['year']:.0f} | ${row['revenue']/1000000:,.1f} | ${row['nopat']/1000000:,.1f} | ${row['pv_fcf']/1000000:,.1f} |"
            for row in s.dcf_table
        ]
        
        assumptions_block = (
            f"\n*Assumptions:*\n"
            f"| Revenue Growth (Year 1-5) | Revenue Growth (Year 6-10) | EBIT Margin Target | Tax Rate | WACC | Terminal Growth |\n"
            f"| --------------------------- | --------------------------- | ------------------ | -------- | ---- | ----------------- |\n"
            f"| {s.assumptions.revenue_growth_stage_1:.1%} | {s.assumptions.revenue_growth_stage_2:.1%} | {s.assumptions.ebit_margin_target:.1%} | {s.assumptions.tax_rate:.1%} | {s.assumptions.wacc:.1%} | {s.assumptions.terminal_growth:.1%} |"
        )
        
        tv = s.terminal_value_details
        if tv:
            sum_pv_fcf = sum(row['pv_fcf'] for row in s.dcf_table)
            ev = sum_pv_fcf + tv.pv_terminal
            shares_out_m = valuation.shares_outstanding / 1000000
            
            tv_block = (
                f"\n\n**Terminal Value & Intrinsic Value Calculation**\n\n"
                f"```text\n"
                f"         NOPAT₁₀ × (1 + g)        ${tv.terminal_nopat/1000000:,.1f}M\n"
                f"TV  =  ─────────────────────  =  ─────────────────────  =  ${tv.terminal_value/1000000:,.1f}M\n"
                f"             WACC − g              {s.assumptions.wacc:.1%} − {s.assumptions.terminal_growth:.1%}\n\n"
                f"           TV          ${tv.terminal_value/1000000:,.1f}M\n"
                f"PV  =  ──────────  =  ────────────────────  =  ${tv.pv_terminal/1000000:,.1f}M\n"
                f"        (1+WACC)¹⁰     (1 + {s.assumptions.wacc:.1%})¹⁰\n\n"
                f"Sum of PV(FCF) Years 1-10 :  ${sum_pv_fcf/1000000:,.1f}M\n"
                f"Plus PV of Terminal Value :  ${tv.pv_terminal/1000000:,.1f}M\n"
                f"────────────────────────────────────────────────────────\n"
                f"Enterprise Value          :  ${ev/1000000:,.1f}M\n"
                f"÷ Shares Outstanding      :   {shares_out_m:,.1f}M\n"
                f"────────────────────────────────────────────────────────\n"
                f"Intrinsic Value per Share :  ${s.intrinsic_value:,.2f}\n"
                f"```\n"
                f"*(Note: Intrinsic value assumes a static share count. The mechanical effect of "
                f"potential future share buybacks—which would naturally reduce shares outstanding "
                f"and increase per-share value—is not modeled, providing an additional margin of safety.)*"
            )
        else:
            tv_block = ""
        
        projection_blocks.append(
            assumptions_block + "\n\n" +
            f"### {label} Scenario Projections\n\n"
            f"| Year | Revenue | NOPAT | PV(FCF) |\n"
            f"| ------ | --------- | ------- | --------- |\n"
            + "\n".join(rows)
            + tv_block
        )

    projections_section = "*(All absolute financial figures in millions)*\n" + "\n".join(projection_blocks)

    # ── Summary metrics ──────────────────────────────────────────────────
    summary = (
        f"**Expected Intrinsic Value**: ${valuation.expected_value:,.2f}  \n"
        f"**Current Price**: ${valuation.current_price:,.2f}  \n"
        f"**Expected 5-Year CAGR**: {cagr_str}  \n"
        f"**Dispersion Ratio**: {valuation.dispersion_ratio:.2f}  \n"
        f"**Recommendation**: {valuation.recommendation}"
    )

    return f"{scenario_table}\n\n{projections_section}\n\n{summary}"


def _build_dcf_summary_for_judge(valuation: ValuationModel) -> str:
    """Build a truncated Markdown summary of DCF results exclusively for the Judge context.
    
    Excludes the year-by-year projections to prevent the LLM from focusing too heavily
    on mathematical line items rather than the overall thesis."""
    cagr = valuation.expected_cagr
    cagr_str = f"{cagr:.1%}" if cagr is not None else "N/A"

    scenario_rows = []
    for label, s in valuation.scenarios.items():
        margin = valuation.scenario_margins.get(label)
        margin_str = f"{margin:.1%}" if margin is not None else "N/A"
        iv_str = f"${s.intrinsic_value:,.2f}" if s.intrinsic_value is not None else "N/A"
        
        a = s.assumptions
        assumptions_str = (
            f"Growth(Y1-5): {a.revenue_growth_stage_1:.1%}, Growth(Y6-10): {a.revenue_growth_stage_2:.1%}, "
            f"EBIT Margin: {a.ebit_margin_target:.1%}, Term Growth: {a.terminal_growth:.1%}"
        )
        
        scenario_rows.append(
            f"**{label}** (Prob: {s.probability:.0%}) | IV: {iv_str} | Margin of Safety: {margin_str}\n"
            f"  *Assumptions*: {assumptions_str}"
        )

    scenarios_text = "\n\n".join(scenario_rows)

    summary = (
        f"**Expected Intrinsic Value**: ${valuation.expected_value:,.2f}  \n"
        f"**Current Price**: ${valuation.current_price:,.2f}  \n"
        f"**Expected 5-Year CAGR**: {cagr_str}  \n"
        f"**Dispersion Ratio**: {valuation.dispersion_ratio:.2f}  \n"
        f"**Recommendation**: {valuation.recommendation}"
    )

    return f"### DCF Scenarios\n{scenarios_text}\n\n### Valuation Summary\n{summary}"


# ── Main node ────────────────────────────────────────────────────────────

@log_node_execution
async def judge_analyst(state: WorkflowState) -> Command[Literal[AgentNode.REPORT_GENERATOR]]:
    """
    Three-step judge node: synthesise → valuate → reconcile.

    1. Synthesise: reads bull + bear theses, produces qualitative synthesis.
    2. Valuate:    reads fundamentals + synthesis, produces DCF assumptions as JSON,
                   then compute_dcf() fills in intrinsic values.
    3. Reconcile:  reviews DCF output vs qualitative synthesis, produces final decision.
    """
    logger.info(f"[Judge] Synthesising theses for {state['company']}...")

    # ── Step 1: Synthesise ──────────────────────────────────────────────
    synthesis_prompt = SYNTHESIS_SYSTEM.format(
        company=state["company"],
        ticker=state["ticker"],
    )
    synthesis_messages = [
        SystemMessage(content=synthesis_prompt),
        HumanMessage(
            content=(
                f"BULL THESIS:\n{state.get('bull_thesis', 'N/A')}\n\n"
                f"BEAR THESIS:\n{state.get('bear_thesis', 'N/A')}"
            )
        ),
    ]
    synthesis_response = await judge_model.ainvoke(synthesis_messages)
    initial_synthesis = synthesis_response.content
    logger.info("[Judge] ✅ Synthesis complete")
    logger.debug(f"[Judge] Synthesis snippet: {initial_synthesis[:200]}...")

    # ── Step 2: Valuate ─────────────────────────────────────────────────
    asset = state.get("price_data")
    if not asset or not getattr(asset.fundamentals, "total_revenue", None) or not asset.shares_outstanding:
        logger.warning("[Judge] Missing fundamental data — skipping valuation")
        return Command(
            update={"judge_decision": initial_synthesis},
            goto=AgentNode.REPORT_GENERATOR,
        )

    logger.info(f"[Judge] Deriving DCF valuation for {state['company']}...")
    valuation_prompt = VALUATION_SYSTEM.format(
        company=state["company"],
        ticker=state["ticker"],
    )
    fundamentals_block = _build_fundamentals_summary(state)

    # Build prior valuation context if available
    prior_valuation = state.get("valuation")
    prior_block = ""
    if prior_valuation and prior_valuation.updated_at:
        from datetime import datetime, timezone
        delta = datetime.now(timezone.utc) - prior_valuation.updated_at
        age_str = f"{delta.days} days old" if delta.days > 0 else "updated today"
        prior_scenarios = []
        for label, s in prior_valuation.scenarios.items():
            a = s.assumptions
            prior_scenarios.append(
                f"  {label} (p={s.probability}): "
                f"growth_s1={a.revenue_growth_stage_1:.1%}, growth_s2={a.revenue_growth_stage_2:.1%}, "
                f"ebit_margin={a.ebit_margin_target:.1%}, wacc={a.wacc:.1%}, "
                f"terminal_g={a.terminal_growth:.1%}"
            )
        prior_block = (
            f"\n\n## Prior Valuation ({age_str})\n"
            f"Previous recommendation: {prior_valuation.recommendation}\n"
            f"Previous expected IV: ${prior_valuation.expected_value:,.2f}\n"
            f"Previous assumptions:\n" + "\n".join(prior_scenarios)
        )
        logger.info(f"[Judge] Injecting prior valuation context ({age_str})")

    valuation_user_content = (
        f"## Company Fundamentals\n{fundamentals_block}\n\n"
        f"## Analyst Synthesis\n{initial_synthesis}"
        f"{prior_block}"
    )
    valuation_messages = [
        SystemMessage(content=valuation_prompt),
        HumanMessage(content=valuation_user_content),
    ]

    # Retry loop — structured output from free-tier models can be flaky
    max_retries = 2
    last_error = None
    valuation = None

    for attempt in range(max_retries + 1):
        try:
            response = await valuation_model.ainvoke(valuation_messages)
            raw_text = response.content if isinstance(response.content, str) else str(response.content)
            logger.debug(f"[Judge] Valuation raw (attempt {attempt + 1}):\n{raw_text[:500]}")

            valuation = _parse_valuation_response(raw_text, state)
            base_year = int(state["date"][:4])
            valuation.compute_dcf(base_year=base_year)

            logger.info(f"[Judge] ✅ Valuation complete — Expected Value: ${valuation.expected_value:.2f}")
            logger.info(f"[Judge]    Recommendation: {valuation.recommendation}")
            logger.info(f"[Judge]    Expected CAGR: {valuation.expected_cagr}")
            for label, s in valuation.scenarios.items():
                logger.info(f"[Judge]    {label}: IV=${s.intrinsic_value} (p={s.probability})")
            break

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            last_error = e
            logger.warning(f"[Judge] Valuation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries:
                valuation_messages.append(HumanMessage(
                    content=(
                        f"Your response was not valid JSON. Error: {e}\n"
                        "Please respond with ONLY a valid JSON object matching the schema. "
                        "No markdown, no commentary."
                    )
                ))

    if valuation is None:
        logger.error(f"[Judge] ❌ Valuation failed after {max_retries + 1} attempts: {last_error}")
        return Command(
            update={"judge_decision": initial_synthesis},
            goto=AgentNode.REPORT_GENERATOR,
        )

    # ── Step 3: Reconcile ───────────────────────────────────────────────
    logger.info("[Judge] Reconciling qualitative synthesis with DCF output...")
    reconcile_prompt = RECONCILE_SYSTEM.format(
        company=state["company"],
        ticker=state["ticker"],
    )

    prior_thesis = "No prior thesis available. This is the initial analysis."
    prior_valuation = state.get("valuation")
    if prior_valuation and prior_valuation.thesis_data:
        prior_thesis = prior_valuation.thesis_data.get("final_decision", prior_thesis)

    # Build prior pillars context from rag_context and retrieved_memory_ids
    prior_pillars_context = _build_prior_pillars_context(state)

    dcf_summary_for_judge = _build_dcf_summary_for_judge(valuation)

    reconcile_messages = [
        SystemMessage(content=reconcile_prompt),
        HumanMessage(
            content=(
                f"## Prior Investment Thesis\n{prior_thesis}\n\n"
                f"{prior_pillars_context}\n\n"
                f"## New Qualitative Synthesis\n{initial_synthesis}\n\n"
                f"## New DCF Valuation Results\n{dcf_summary_for_judge}"
            )
        ),
    ]
    reconcile_response = await judge_model.ainvoke(reconcile_messages)
    reconcile_raw = reconcile_response.content if isinstance(reconcile_response.content, str) else str(reconcile_response.content)
    logger.info("[Judge] ✅ Reconciliation complete")
    logger.debug(f"[Judge] Reconcile snippet: {reconcile_raw[:200]}...")

    # ── Parse thesis pillars and pillar outcomes from reconcile response ─
    final_decision, thesis_pillars, pillar_outcomes = _parse_reconcile_output(
        reconcile_raw, state
    )
    logger.info(
        "[Judge] Extracted %d thesis pillars, %d pillar outcomes",
        len(thesis_pillars), len(pillar_outcomes),
    )

    # Save our synthesized elements into the JSONB payloads for the database NOW
    if valuation is not None:
        valuation.thesis_data = {
            "synthesis": initial_synthesis,
            "bull": state.get("bull_thesis") or "",
            "bear": state.get("bear_thesis") or "",
            "final_decision": final_decision if isinstance(final_decision, str) else str(final_decision),
        }
        valuation.valuation_data = valuation.model_dump(mode="json")

    # ── Persist judge output to run artifact ────────────────────────────
    dcf_md = _build_dcf_summary(valuation)
    judge_content = (
        f"# {state['company']} ({state['ticker']}) Judge Analysis\n\n"
        f"## Qualitative Synthesis\n\n"
        f"{initial_synthesis if isinstance(initial_synthesis, str) else str(initial_synthesis)}\n\n"
        f"## DCF Valuation\n\n"
        f"{dcf_md}\n\n"
        f"## Final Decision\n\n"
        f"{final_decision if isinstance(final_decision, str) else str(final_decision)}"
    )
    artifact: dict = {}
    try:
        artifact = await persist_research_artifact(
            ticker=state["ticker"],
            content=judge_content,
            source_type="judge_analysis",
            source_priority=2,
            vectorize=False,
            metadata={
                "agent": "judge",
                "company": state.get("company", ""),
                "run_datetime": state.get("run_datetime", ""),
                "recommendation": valuation.recommendation if valuation else None,
            },
        )
    except Exception as e:
        logger.warning(f"[Judge] ⚠️ Failed to persist judge analysis: {e}")

    run_dt = state.get("run_datetime", "")
    if run_dt:
        try:
            writer = RunReportWriter(
                ticker=state["ticker"],
                run_datetime=run_dt,
                company=state.get("company", ""),
            )
            writer.write_judge_output(
                synthesis=initial_synthesis if isinstance(initial_synthesis, str) else str(initial_synthesis),
                valuation_md=dcf_md,
                decision=final_decision if isinstance(final_decision, str) else str(final_decision),
            )
            logger.info(f"[Judge] 📝 Written judge output to {writer.debug_path}")
        except Exception as e:
            logger.warning(f"[Judge] ⚠️ Failed to write judge output: {e}")

    update = {
        "judge_decision": final_decision,
        "valuation": valuation,
        "thesis_data": initial_synthesis,
        "thesis_pillars": [p.model_dump(mode="json") for p in thesis_pillars],
        "pillar_outcomes": [o.model_dump(mode="json") for o in pillar_outcomes],
    }
    if artifact.path:
        update["vault_artifacts"] = [artifact.model_dump(mode="json")]

    return Command(
        update=update,
        goto=AgentNode.REPORT_GENERATOR,
    )


# ── Prior pillars context builder ────────────────────────────────────────────


def _build_prior_pillars_context(state: WorkflowState) -> str:
    """Build a Markdown summary of prior active pillars for the reconcile prompt.

    Uses ``rag_context`` and ``retrieved_memory_ids`` from state to identify
    which pillars were retrieved and pass them with their UUIDs.
    """
    retrieved_ids = state.get("retrieved_memory_ids", [])
    if not retrieved_ids:
        return "## Prior Thesis Pillars\nNo prior pillars available."

    rag_context = state.get("rag_context", "")
    if rag_context:
        return f"## Prior Thesis Pillars\n{rag_context}"

    return f"## Prior Thesis Pillars\n{len(retrieved_ids)} prior pillar memory IDs retrieved."


# ── Reconcile output parsing ─────────────────────────────────────────────────

_JSON_DELIMITER = "----JSON----"


def _parse_reconcile_output(
    raw: str,
    state: WorkflowState,
) -> tuple[str, list, list]:
    """Parse the judge's reconcile output into decision text + structured pillars.

    Splits on ``----JSON----`` delimiter.  The text portion becomes
    ``final_decision``.  The JSON portion is parsed into ``ThesisPillar`` and
    ``PillarOutcome`` lists.

    Falls back gracefully: if no delimiter is found, the entire response becomes
    the decision with empty pillars/outcomes.
    """
    from schemas.rag import ThesisPillar, PillarOutcome

    retrieved_ids = state.get("retrieved_memory_ids", [])

    if _JSON_DELIMITER not in raw:
        logger.warning("[Judge] No pillar JSON delimiter found in reconcile output")
        return raw, [], []

    text_part, json_part = raw.split(_JSON_DELIMITER, 1)
    final_decision = text_part.strip()

    # ── Parse JSON ──────────────────────────────────────────────────────
    json_text = json_part.strip()
    # Strip markdown fences if present
    if json_text.startswith("```"):
        first_nl = json_text.index("\n") if "\n" in json_text else 3
        json_text = json_text[first_nl:].strip()
    if json_text.endswith("```"):
        json_text = json_text[:-3].strip()

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.warning("[Judge] Failed to parse pillar JSON: %s", e)
        return final_decision, [], []

    thesis_pillars: list[ThesisPillar] = []
    pillar_outcomes: list[PillarOutcome] = []

    # Parse thesis_pillars
    for raw_pillar in data.get("thesis_pillars", []):
        try:
            matched_pillar_id = str(raw_pillar.get("matched_pillar_id", "") or "")
            if not matched_pillar_id and raw_pillar.get("matched_prior_ref"):
                matched_pillar_id = str(raw_pillar.get("pillar_id", "") or "")
            raw_status = str(raw_pillar.get("status", "supported")).strip().lower()
            pillar_status = "weakened" if raw_status == "weakened" else "supported"
            thesis_pillars.append(ThesisPillar(
                pillar_id=matched_pillar_id,
                candidate_ref=str(raw_pillar.get("candidate_ref", "")),
                matched_prior_ref=str(raw_pillar.get("matched_prior_ref", "")),
                matched_pillar_id=matched_pillar_id,
                pillar_type=raw_pillar.get("pillar_type", "growth"),
                statement=str(raw_pillar.get("statement", "")),
                rationale=str(raw_pillar.get("rationale", "")),
                valuation_impact=str(raw_pillar.get("valuation_impact", "")),
                source_urls=list(raw_pillar.get("source_urls", [])),
                evidence_citations=list(raw_pillar.get("evidence_citations", [])),
                resurrection_reason=str(raw_pillar.get("resurrection_reason", "")),
                status=pillar_status,
            ))
        except Exception as e:
            logger.warning("[Judge] Skipping malformed thesis pillar: %s", e)

    # Parse pillar_outcomes
    for raw_outcome in data.get("pillar_outcomes", []):
        try:
            memory_id = str(raw_outcome.get("memory_id", ""))
            pillar_id = str(raw_outcome.get("pillar_id", ""))
            status = str(raw_outcome.get("status", "supported")).strip().lower()
            if status == "updated":
                status = "revised"

            # Validate memory_id is in retrieved_ids (security: don't trust LLM to invent UUIDs)
            if memory_id and retrieved_ids and memory_id not in retrieved_ids:
                logger.warning(
                    "[Judge] Pillar outcome memory_id %s not in retrieved_ids, skipping",
                    memory_id[:8],
                )
                continue

            pillar_outcomes.append(PillarOutcome(
                memory_id=memory_id,
                pillar_id=pillar_id,
                status=status,
                reason=str(raw_outcome.get("reason", "")),
                replacement_statement=str(raw_outcome.get("replacement_statement", "")),
                source_urls=list(raw_outcome.get("source_urls", [])),
            ))
        except Exception as e:
            logger.warning("[Judge] Skipping malformed pillar outcome: %s", e)

    logger.info(
        "[Judge] Parsed %d pillars + %d outcomes from reconcile JSON",
        len(thesis_pillars), len(pillar_outcomes),
    )
    return final_decision, thesis_pillars, pillar_outcomes
