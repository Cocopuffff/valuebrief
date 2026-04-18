from pydantic import BaseModel, ConfigDict, Field, computed_field
from typing import List, Optional, Literal, Dict
from datetime import datetime
from enum import Enum
import warnings


# ── Active Models ─────────────────────────────────────────────────────────

class FinancialMetrics(BaseModel):
    """Fundamental metrics of a company, populated from AlphaVantage and yfinance."""
    pe_ratio: Optional[float] = Field(default=None, description="Trailing Price to Earnings Ratio")
    forward_pe_ratio: Optional[float] = Field(default=None, description="Forward Price to Earnings Ratio")
    peg_ratio: Optional[float] = Field(default=None, description="Price/Earnings to Growth Ratio")
    price_to_book: Optional[float] = Field(default=None, description="Price to Book Ratio")
    debt_to_equity: Optional[float] = Field(default=None, description="Total Debt to Equity Ratio")
    dividend_yield: Optional[float] = Field(default=None, description="Forward Dividend Yield")
    free_cash_flow: Optional[float] = Field(default=None, description="Free Cash Flow")
    revenue_growth: Optional[float] = Field(default=None, description="Quarterly Revenue Growth (yoy)")
    ebitda_margin: Optional[float] = Field(default=None, description="EBITDA Margin")
    earnings_per_share: Optional[float] = Field(default=None, description="Trailing Earnings Per Share")
    market_cap: Optional[float] = Field(default=None, description="Market Capitalization")
    return_on_equity: Optional[float] = Field(default=None, description="Return on Equity")
    current_ratio: Optional[float] = Field(default=None, description="Current Ratio (current assets / current liabilities)")
    total_revenue: Optional[float] = Field(default=None, description="Total Revenue (TTM)")
    profit_margin: Optional[float] = Field(default=None, description="Profit Margin")
    operating_margin: Optional[float] = Field(default=None, description="Operating Margin (TTM)")
    return_on_assets: Optional[float] = Field(default=None, description="Return on Assets (TTM)")
    price_to_sales: Optional[float] = Field(default=None, description="Price to Sales Ratio (TTM)")
    ev_to_revenue: Optional[float] = Field(default=None, description="Enterprise Value to Revenue")
    ev_to_ebitda: Optional[float] = Field(default=None, description="Enterprise Value to EBITDA")
    beta: Optional[float] = Field(default=None, description="Beta")
    target_price: Optional[float] = Field(default=None, description="Analyst Target Price")
    fifty_two_week_high: Optional[float] = Field(default=None, description="52 Week High")
    fifty_two_week_low: Optional[float] = Field(default=None, description="52 Week Low")
    fifty_day_moving_average: Optional[float] = Field(default=None, description="50 Day Moving Average")
    two_hundred_day_moving_average: Optional[float] = Field(default=None, description="200 Day Moving Average")


class Asset(BaseModel):
    """Snapshot of a company's market data and fundamentals."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'GOOGL')")
    name: Optional[str] = Field(default=None, description="Company name")
    sector: Optional[str] = Field(default=None, description="Sector classification")
    industry: Optional[str] = Field(default=None, description="Industry classification")
    current_price: float = Field(..., description="Current market price per share")
    shares_outstanding: Optional[float] = Field(default=None, description="Total diluted shares outstanding")
    intrinsic_value: Optional[float] = Field(default=None, description="Estimated intrinsic value per share")
    margin_of_safety: Optional[float] = Field(default=None, description="Margin of safety vs current price")
    fundamentals: FinancialMetrics = Field(default_factory=lambda: FinancialMetrics())
    last_updated: datetime = Field(default_factory=datetime.now)

    @property
    def is_undervalued(self) -> bool:
        if self.intrinsic_value and self.current_price:
            return self.current_price < self.intrinsic_value
        return False


# ── Valuation Models ──────────────────────────────────────────────────────

class DCFAssumptions(BaseModel):
    """Growth, margin, and discount-rate drivers for a single DCF scenario."""
    revenue_growth_stage_1: float = Field(..., description="Annual revenue growth rate for years 1-5")
    revenue_growth_stage_2: float = Field(..., description="Annual revenue growth rate for years 6-10")
    ebit_margin_target: float = Field(..., description="Target EBIT (operating) margin")
    tax_rate: float = Field(0.21, description="Effective tax rate")
    wacc: float = Field(0.10, description="Weighted Average Cost of Capital")
    terminal_growth: float = Field(0.025, description="Perpetual growth rate for terminal value")


class TerminalValueDetails(BaseModel):
    """Terminal value breakdown computed during DCF."""
    terminal_nopat: float = Field(..., description="Year-10 NOPAT × (1 + terminal growth)")
    terminal_value: float = Field(..., description="Gordon Growth terminal value")
    pv_terminal: float = Field(..., description="Present value of terminal value")


class DCFScenario(BaseModel):
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


class ValuationModel(BaseModel):
    """
    Master valuation model tying together scenario-based DCF analysis.

    Usage:
        model = ValuationModel(ticker="AAPL", company="Apple Inc.", ...)
        model.compute_dcf(base_year=current_year)   # fills intrinsic_value, dcf_table, terminal_value_details
        print(model.expected_cagr)
    """
    ticker: str = Field(..., description="Stock ticker symbol e.g. AAPL")
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
        """Probability-weighted intrinsic value across all scenarios."""
        return round(
            sum(s.probability * (s.intrinsic_value or 0) for s in self.scenarios.values()),
            2
        )

    @computed_field
    @property
    def expected_cagr(self) -> Optional[float]:
        """
        Implied annual return (IRR) from the probability-weighted DCF cash flows
        at the current market price.

        Solves for r in:
            Market Cap = Σ(weighted_NOPAT_t / (1+r)^t) + weighted_TV / (1+r)^10

        This is the rate of return an investor earns if the projected cash flows
        materialise and the stock is purchased at today's market price —
        equivalent to the Stock Unlock DCF visualiser approach.
        """
        if self.current_price <= 0 or not self.scenarios:
            return None

        market_cap = self.current_price * self.shares_outstanding

        # Build probability-weighted cash flows for years 1–10 and terminal value
        weighted_nopats: Dict[int, float] = {}  # year_index (1-10) → weighted NOPAT
        weighted_tv = 0.0

        for s in self.scenarios.values():
            if not s.dcf_table or not s.terminal_value_details:
                return None  # DCF hasn't been computed yet
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
            return None  # no root in range

        for _ in range(100):  # bisection converges quickly
            mid = (lo + hi) / 2
            if npv(mid) > 0:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < 1e-4:
                break

        return round((lo + hi) / 2, 4)

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
        """
        vals = [s.intrinsic_value for s in self.scenarios.values() if s.intrinsic_value is not None]
        if not vals or self.expected_value == 0:
            return 0.0
        return round((max(vals) - min(vals)) / self.expected_value, 4)

    @computed_field
    @property
    def recommendation(self) -> str:
        """Investment recommendation based on expected upside to intrinsic value."""
        if self.current_price <= 0:
            return "N/A"
        upside = (self.expected_value - self.current_price) / self.current_price
        if upside > 0.4: return "Strong Buy"
        if upside > 0.2: return "Buy"
        if upside < -0.4: return "Strong Sell"
        if upside < -0.2: return "Sell"
        return "Hold"


# ── LangGraph Node Names ─────────────────────────────────────────────────

class AgentNode(str, Enum):
    SUPERVISOR = "supervisor"
    BEAR = "bear_analyst"
    BULL = "bull_analyst"
    JUDGE = "judge_analyst"
    REPORT_GENERATOR = "report_generator"

    RESEARCH_TOOL_NODE = "research_tool_node"


# ── Deprecated Models ─────────────────────────────────────────────────────
# The following models are no longer used by the agentic workflow.
# They are retained for backward compatibility and may be removed in a
# future version.

class Portfolio(BaseModel):
    """Deprecated: Portfolio tracking now uses portfolio.json + ticker list."""
    name: str
    assets: List[Asset] = []
    description: Optional[str] = None

    def get_undervalued_assets(self) -> List[Asset]:
        warnings.warn("Portfolio is deprecated", DeprecationWarning, stacklevel=2)
        return [asset for asset in self.assets if asset.is_undervalued]


class StockPrice(BaseModel):
    """Deprecated: Price data now flows through the Asset model."""
    ticker: str = Field(..., description="Stock ticker symbol", min_length=1, max_length=10)
    exchange: str = Field(..., description="Stock exchange", min_length=1, max_length=10)
    price: float = Field(..., description="Current stock price in local currency")
    currency: str = Field("USD", description="ISO 4217 currency code")
    previous_close: float = Field(..., description="Previous day's close")
    timestamp: datetime = Field(default_factory=datetime.now)

    @computed_field(description="Daily change percentage")
    @property
    def change_percent(self) -> float:
        if self.previous_close:
            return (self.price / self.previous_close) - 1
        return 0.0


class NewsArticle(BaseModel):
    """Deprecated: News data flows as raw dicts from DuckDuckGo search."""
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        str_strip_whitespace=True
    )

    url: str
    title: str
    source: Optional[str] = None
    published_date: Optional[datetime] = None
    summary: Optional[str] = None
    content: str
    relevance_score: Optional[float] = Field(None, ge=0, le=1)
    is_fundamental: Optional[bool] = None
    retrieval_timestamp: datetime = Field(default_factory=datetime.now)


class PortfolioSnapshot(BaseModel):
    """Deprecated: Depends on deprecated StockPrice and NewsArticle."""
    holdings: List[StockPrice]
    news: List[NewsArticle]
    generated_at: datetime = Field(default_factory=datetime.now)
    total_value: Optional[float] = None


class AgentAction(str, Enum):
    """Deprecated: Workflow routing now uses AgentNode enum + Command pattern."""
    FETCH_PRICES = "fetch_prices"
    FETCH_NEWS = "fetch_news"
    ANALYSE_FUNDAMENTALS = "analyse_fundamentals"
    VALUE_COMPANY = "value_company"
    GENERATE_REPORT = "generate_report"
    WAIT = "wait"