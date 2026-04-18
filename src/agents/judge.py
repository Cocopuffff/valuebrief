import json
from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import Command
from typing import Literal
from .states import WorkflowState
from models import ValuationModel, AgentNode
from logger import get_logger
from report_writer import RunReportWriter
from config import judge_model, valuation_model

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
You are the lead investment analyst for {company} ({ticker}). You previously wrote a \
qualitative synthesis of the bull and bear cases. The quantitative DCF valuation has \
now been computed from assumptions you derived.

Review the qualitative synthesis alongside the DCF outputs below. Your job is to \
produce a FINAL investment decision that reconciles any tension between the qualitative \
sentiment and the quantitative reality.

For example:
- If your synthesis was bullish but the stock trades ABOVE even the bull-case intrinsic \
  value, flag the valuation risk and recommend patience.
- If your synthesis was cautious but the DCF shows significant upside with wide margin \
  of safety, acknowledge the quantitative case.

Structure your response as:
1. **Verdict** — one sentence (e.g. "Buy on weakness" / "Hold" / "Avoid").
2. **Rationale** — 2-3 paragraphs integrating qualitative and quantitative views.
3. **Key Risks** — bullet list of the most important risks to your thesis.

Do NOT call any tools. Respond with plain text only.\
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
        tv_block = (
            f"\n\n**Terminal Value — Perpetuity Growth Method**\n\n"
            f"```\n"
            f"         NOPAT₁₀ × (1 + g)        ${tv.terminal_nopat/1000000:,.1f}M\n"
            f"TV  =  ─────────────────────  =  ────────────────────────  =  ${tv.terminal_value/1000000:,.1f}M\n"
            f"             WACC − g              {s.assumptions.wacc:.1%} − {s.assumptions.terminal_growth:.1%}\n\n"
            f"           TV          ${tv.terminal_value/1000000:,.1f}M\n"
            f"PV  =  ──────────  =  ──────────────────────  =  ${tv.pv_terminal/1000000:,.1f}M\n"
            f"        (1+WACC)¹⁰     (1 + {s.assumptions.wacc:.1%})¹⁰\n"
            f"```"
            if tv else ""
        )
        
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


# ── Main node ────────────────────────────────────────────────────────────

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
            
            # Save our synthesized elements into the JSONB payloads for the database
            valuation.thesis_data = {
                "synthesis": initial_synthesis,
                "bull": state.get("bull_thesis"),
                "bear": state.get("bear_thesis")
            }
            valuation.valuation_data = valuation.model_dump(mode="json")

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
    dcf_summary = _build_dcf_summary(valuation)
    reconcile_messages = [
        SystemMessage(content=reconcile_prompt),
        HumanMessage(
            content=(
                f"## Your Previous Synthesis\n{initial_synthesis}\n\n"
                f"## DCF Valuation Results\n{dcf_summary}"
            )
        ),
    ]
    reconcile_response = await judge_model.ainvoke(reconcile_messages)
    final_decision = reconcile_response.content
    logger.info("[Judge] ✅ Reconciliation complete")
    logger.debug(f"[Judge] Final decision snippet: {final_decision[:200]}...")

    # ── Persist judge output to run artifact ────────────────────────────
    run_dt = state.get("run_datetime", "")
    if run_dt:
        try:
            writer = RunReportWriter(ticker=state["ticker"], run_datetime=run_dt)
            dcf_md = _build_dcf_summary(valuation)
            writer.write_judge_output(
                synthesis=initial_synthesis if isinstance(initial_synthesis, str) else str(initial_synthesis),
                valuation_md=dcf_md,
                decision=final_decision if isinstance(final_decision, str) else str(final_decision),
            )
            logger.info(f"[Judge] 📝 Written judge output to {writer.debug_path}")
        except Exception as e:
            logger.warning(f"[Judge] ⚠️ Failed to write judge output: {e}")

    return Command(
        update={"judge_decision": final_decision, "valuation": valuation, "thesis_data": initial_synthesis},
        goto=AgentNode.REPORT_GENERATOR,
    )
