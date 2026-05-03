"""
valuation.py
~~~~~~~~~~~~
DCF valuation models: assumptions, projection rows, scenarios, and the master model.
"""

from datetime import datetime
from typing import Optional, Dict, List, Literal

from pydantic import Field, computed_field

from schemas import StrictBaseModel, Ticker


# ── DCF Assumptions ──────────────────────────────────────────────────────

class DCFAssumptions(StrictBaseModel):
    """Growth, margin, and discount-rate drivers for a single DCF scenario."""
    revenue_growth_stage_1: float = Field(..., description="Annual revenue growth rate for years 1-5")
    revenue_growth_stage_2: float = Field(..., description="Annual revenue growth rate for years 6-10")
    ebit_margin_target: float = Field(..., description="Target EBIT (operating) margin")
    tax_rate: float = Field(0.21, description="Effective tax rate")
    wacc: float = Field(0.10, description="Weighted Average Cost of Capital")
    terminal_growth: float = Field(0.025, description="Perpetual growth rate for terminal value")


# ── DCF Projection Row ───────────────────────────────────────────────────

class DCFProjectionRow(StrictBaseModel):
    """A single year's row in a DCF projection table."""
    year: int
    revenue: float
    nopat: float
    pv_fcf: float


# ── Terminal Value ───────────────────────────────────────────────────────

class TerminalValueDetails(StrictBaseModel):
    """Terminal value breakdown computed during DCF."""
    terminal_nopat: float = Field(..., description="Year-10 NOPAT * (1 + terminal growth)")
    terminal_value: float = Field(..., description="Gordon Growth terminal value")
    pv_terminal: float = Field(..., description="Present value of terminal value")


# ── DCF Scenario ─────────────────────────────────────────────────────────

class DCFScenario(StrictBaseModel):
    """A single valuation outcome (Bear, Base, or Bull)."""
    label: Literal["Bear", "Base", "Bull"]
    probability: float = Field(..., ge=0, le=1)
    assumptions: DCFAssumptions
    intrinsic_value: Optional[float] = Field(
        default=None,
        description="Per-share intrinsic value. Computed via ValuationModel.compute_dcf() or set directly by an agent."
    )
    dcf_table: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Year-by-year projections: calendar_year, revenue, nopat, pv_fcf"
    )
    terminal_value_details: Optional[TerminalValueDetails] = Field(
        default=None,
        description="Terminal value breakdown"
    )


# ── Valuation Model ──────────────────────────────────────────────────────

class ValuationModel(StrictBaseModel):
    """
    Master valuation model tying together scenario-based DCF analysis.

    Usage:
        model = ValuationModel(ticker="AAPL", company="Apple Inc.", ...)
        model.compute_dcf(base_year=current_year)   # fills intrinsic_value, dcf_table, terminal_value_details
        print(model.expected_cagr)
    """
    ticker: Ticker = Field(..., description="Stock ticker symbol e.g. AAPL")
    exchange: Optional[str] = Field(default=None, description="Stock exchange e.g. NASDAQ, NYSE, AS")
    company: str = Field(..., description="Company name e.g. Apple Inc.")
    last_updated: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default=None, description="DB-managed timestamp of last persistence to Supabase")
    current_price: float = Field(..., description="Current market price per share")
    currency: str = "USD"
    base_revenue: float = Field(..., description="Trailing twelve months (TTM) revenue")
    shares_outstanding: float = Field(..., description="Total diluted shares outstanding")
    thesis_data: Optional[Dict] = Field(default=None, description="Qualitative thesis synthesis")
    valuation_data: Optional[Dict] = Field(default=None, description="JSON dictionary of the valuation result")

    scenarios: Dict[str, DCFScenario] = Field(
        ...,
        description="Map of scenario labels ('Bear', 'Base', 'Bull') to their models"
    )

    # ── Snapshot fields (hydration from DB / LLM) ────────────────────
    # These accept the known computed-property input keys and store them
    # internally. Public access goes through the @computed_field properties
    # below, which compute from scenario data when available and fall back
    # to these snapshots only when computation cannot produce a result.

    expected_value_snapshot: Optional[float] = Field(
        default=None,
        validation_alias="expected_value",
        exclude=True,
        description="Snapshot of expected_value hydrated from DB or LLM output. "
                    "The public .expected_value property computes from scenarios when possible.",
    )
    expected_cagr_snapshot: Optional[float] = Field(
        default=None,
        validation_alias="expected_cagr",
        exclude=True,
        description="Snapshot of expected_cagr hydrated from DB or LLM output.",
    )
    dispersion_ratio_snapshot: Optional[float] = Field(
        default=None,
        validation_alias="dispersion_ratio",
        exclude=True,
        description="Snapshot of dispersion_ratio hydrated from DB or LLM output.",
    )
    recommendation_snapshot: Optional[str] = Field(
        default=None,
        validation_alias="recommendation",
        exclude=True,
        description="Snapshot of recommendation hydrated from DB or LLM output.",
    )

    # ── DCF computation ──────────────────────────────────────────────

    def compute_dcf(self, base_year: int) -> None:
        """
        Two-stage DCF with calendar-year projections.

        Stage 1 (years 1-5):  revenue grows at stage_1 rate
        Stage 2 (years 6-10): revenue grows at stage_2 rate
        Terminal value:       Gordon Growth Model on year-10 NOPAT

        Args:
            base_year: the current calendar year.
                       Projections start at base_year + 1.
        """
        for scenario in self.scenarios.values():
            a = scenario.assumptions
            revenue = self.base_revenue
            pv_fcfs = 0.0
            dcf_table = []

            # Stage 1: years 1–5
            for year in range(1, 6):
                revenue *= (1 + a.revenue_growth_stage_1)
                nopat = revenue * a.ebit_margin_target * (1 - a.tax_rate)
                pv_fcf = nopat / (1 + a.wacc) ** year
                pv_fcfs += pv_fcf
                dcf_table.append({
                    "year": base_year + year,
                    "revenue": round(revenue, 2),
                    "nopat": round(nopat, 2),
                    "pv_fcf": round(pv_fcf, 2),
                })

            # Stage 2: years 6–10
            for year in range(6, 11):
                revenue *= (1 + a.revenue_growth_stage_2)
                nopat = revenue * a.ebit_margin_target * (1 - a.tax_rate)
                pv_fcf = nopat / (1 + a.wacc) ** year
                pv_fcfs += pv_fcf
                dcf_table.append({
                    "year": base_year + year,
                    "revenue": round(revenue, 2),
                    "nopat": round(nopat, 2),
                    "pv_fcf": round(pv_fcf, 2),
                })

            # Terminal value (Gordon Growth on year-10 NOPAT)
            terminal_nopat = revenue * a.ebit_margin_target * (1 - a.tax_rate) * (1 + a.terminal_growth)
            terminal_value = terminal_nopat / (a.wacc - a.terminal_growth)
            pv_terminal = terminal_value / (1 + a.wacc) ** 10

            # Per-share intrinsic value
            enterprise_value = pv_fcfs + pv_terminal
            scenario.intrinsic_value = round(enterprise_value / self.shares_outstanding, 2)
            scenario.dcf_table = dcf_table
            scenario.terminal_value_details = TerminalValueDetails(
                terminal_nopat=round(terminal_nopat, 2),
                terminal_value=round(terminal_value, 2),
                pv_terminal=round(pv_terminal, 2),
            )

    # ── Computed fields ──────────────────────────────────────────────

    @computed_field
    @property
    def expected_value(self) -> float:
        """Probability-weighted intrinsic value across all scenarios.

        Computes from scenario data when intrinsic values are available;
        falls back to the DB/LLM snapshot when scenarios lack computed values.
        """
        vals = [s.intrinsic_value for s in self.scenarios.values() if s.intrinsic_value is not None]
        if vals:
            return round(
                sum(s.probability * (s.intrinsic_value or 0) for s in self.scenarios.values()),
                2
            )
        if self.expected_value_snapshot is not None:
            return self.expected_value_snapshot
        return 0.0

    @computed_field
    @property
    def expected_cagr(self) -> Optional[float]:
        """
        Implied annual return (IRR) from the probability-weighted DCF cash flows
        at the current market price.

        Solves for r in:
            Market Cap = Sum(weighted_NOPAT_t / (1+r)^t) + weighted_TV / (1+r)^10

        Computes from DCF tables when available; falls back to the DB/LLM snapshot
        when DCF projections have not been filled in.
        """
        if self.current_price <= 0 or not self.scenarios:
            return self._cagr_fallback()

        market_cap = self.current_price * self.shares_outstanding

        # Build probability-weighted cash flows for years 1–10 and terminal value
        weighted_nopats: Dict[int, float] = {}  # year_index (1-10) -> weighted NOPAT
        weighted_tv = 0.0

        for s in self.scenarios.values():
            if not s.dcf_table or not s.terminal_value_details:
                return self._cagr_fallback()
            p = s.probability
            for i, row in enumerate(s.dcf_table):
                year_idx = i + 1
                weighted_nopats[year_idx] = weighted_nopats.get(year_idx, 0) + p * row["nopat"]
            weighted_tv += p * s.terminal_value_details.terminal_value

        # Bisection method to solve for IRR
        def npv(r: float) -> float:
            total = 0.0
            for t, nopat in weighted_nopats.items():
                total += nopat / (1 + r) ** t
            total += weighted_tv / (1 + r) ** 10
            return total - market_cap

        lo, hi = -0.50, 2.0  # search between -50% and +200%
        if npv(lo) * npv(hi) > 0:
            return self._cagr_fallback()

        for _ in range(100):  # bisection converges quickly
            mid = (lo + hi) / 2
            if npv(mid) > 0:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < 1e-4:
                break

        return round((lo + hi) / 2, 4)

    def _cagr_fallback(self) -> Optional[float]:
        """Fall back to the DB/LLM snapshot when DCF computation is not possible."""
        return self.expected_cagr_snapshot

    @computed_field
    @property
    def scenario_margins(self) -> Dict[str, Optional[float]]:
        """
        Margin of safety for each scenario.
        MoS = (Intrinsic Value - Current Price) / Intrinsic Value
        """
        result: Dict[str, Optional[float]] = {}
        for label, s in self.scenarios.items():
            if s.intrinsic_value and s.intrinsic_value > 0:
                result[label] = round(
                    (s.intrinsic_value - self.current_price) / s.intrinsic_value,
                    4
                )
            else:
                result[label] = None
        return result

    @computed_field
    @property
    def dispersion_ratio(self) -> float:
        """
        Spread of outcomes relative to expected value.
        (max IV - min IV) / Expected Value

        Computes from scenario data when intrinsic values are available;
        falls back to the DB/LLM snapshot otherwise.
        """
        vals = [s.intrinsic_value for s in self.scenarios.values() if s.intrinsic_value is not None]
        if not vals or self.expected_value == 0:
            if self.dispersion_ratio_snapshot is not None:
                return self.dispersion_ratio_snapshot
            return 0.0
        return round((max(vals) - min(vals)) / self.expected_value, 4)

    @computed_field
    @property
    def recommendation(self) -> str:
        """Investment recommendation based on expected upside to intrinsic value.

        Computes from scenario data when current_price and expected_value are
        meaningful; falls back to the DB/LLM snapshot otherwise.
        """
        if self.current_price <= 0:
            return self.recommendation_snapshot or "N/A"
        upside = (self.expected_value - self.current_price) / self.current_price
        if upside > 0.4:
            return "Strong Buy"
        if upside > 0.2:
            return "Buy"
        if upside < -0.4:
            return "Strong Sell"
        if upside < -0.2:
            return "Sell"
        return "Hold"
