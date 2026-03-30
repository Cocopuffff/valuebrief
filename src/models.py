from pydantic import BaseModel, ConfigDict, HttpUrl, Field, computed_field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class FinancialMetrics(BaseModel):
    """Fundamental metrics of a company"""
    pe_ratio: Optional[float] = Field(default=None, description="Price to Earnings Ratio")
    forward_pe_ratio: Optional[float] = Field(default=None, description="Forward Price to Earnings Ratio")
    peg_ratio: Optional[float] = Field(default=None, description="Price/Earnings to Growth Ratio")
    price_to_book: Optional[float] = Field(default=None, description="Price to Book Ratio")
    debt_to_equity: Optional[float] = Field(default=None, description="Total Debt to Equity Ratio")
    dividend_yield: Optional[float] = Field(default=None, description="Forward Dividend Yield")
    free_cash_flow: Optional[float] = Field(default=None, description="Free Cash Flow")
    revenue_growth: Optional[float] = Field(default=None, description="Quarterly Revenue Growth (yoy)")
    ebitda_margin: Optional[float] = Field(default=None, description="EBITDA Margin")

class Asset(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol (eg. 'AAPL', 'MSFT', 'GOOGL')")
    name: Optional[str] = Field(default=None, description="The name of the asset")
    sector: Optional[str] = Field(default=None, description="The sector of the asset")
    industry: Optional[str] = Field(default=None, description="The industry of the asset")
    current_price: float = Field(..., description="The current price of the asset")
    intrinsic_value: Optional[float] = Field(default=None, description="The intrinsic value of the asset")
    margin_of_safety: Optional[float] = Field(default=None, description="The margin of safety of the asset")
    fundamentals: FinancialMetrics = Field(default_factory=lambda: FinancialMetrics())
    last_updated: datetime = Field(default_factory=datetime.now)

    @property
    def is_undervalued(self) -> bool:
        if self.intrinsic_value and self.current_price:
            return self.current_price < self.intrinsic_value
        return False

class Portfolio(BaseModel):
    name: str
    assets: List[Asset] = []
    description: Optional[str] = None

    def get_undervalued_assets(self) -> List[Asset]:
        return [asset for asset in self.assets if asset.is_undervalued]

class StockPrice(BaseModel):
    """Model for current stock price and daily change"""
    ticker: str = Field(..., description="Stock ticker symbol", min_length=1, max_length=10)
    exchange: str = Field(..., description="Stock exchange that the ticker is listed on", min_length=1, max_length=10)
    price: float = Field(..., description="Current stock price in local currency")
    currency: str = Field("USD", description="Three-letter identifiers of currencies (ISO 4217 codes)")
    previous_close: float = Field(..., description="Previous day's close")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @computed_field(description="Daily change percentage")
    @property
    def change_percent(self) -> float:
        if self.previous_close:
            return (self.price / self.previous_close) - 1
        return 0.0
    
class NewsArticle(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        str_strip_whitespace=True
    )

    url: HttpUrl
    title: str
    source: Optional[str] = None
    published_date: Optional[datetime] = None
    summary: Optional[str] = None
    content: str
    relevance_score: Optional[float] = Field(None, ge=0, le=1)
    is_fundamental: Optional[bool] = None
    retrieval_timestamp: datetime = Field(default_factory=datetime.now)

class PortfolioSnapshot(BaseModel):
    """Complete snapshot of portfolio at a point in time"""
    holdings: List[StockPrice]
    news: List[NewsArticle]
    generated_at: datetime = Field(default_factory=datetime.now)
    total_value: Optional[float] = None

# --- LangGraph-Specific Models ---

class AgentAction(str, Enum):
    """Possible actions agents can take."""
    FETCH_PRICES = "fetch_prices"
    FETCH_NEWS = "fetch_news"
    ANALYSE_FUNDAMENTALS = "analyse_fundamentals"
    VALUE_COMPANY = "value_company"
    GENERATE_REPORT = "generate_report"
    WAIT = "wait"

class AgentNodes(str, Enum):
    SUPERVISOR = "SUPERVISOR"
    BEAR = "BEAR ANALYST"
    BULL = "BULL ANALYST"
    JUDGE = "JUDGE ANALYST"
    REPORT_GENERATOR = "REPORT GENERATOR"
    