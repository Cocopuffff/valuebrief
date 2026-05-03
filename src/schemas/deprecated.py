"""
deprecated.py
~~~~~~~~~~~~~
Deprecated models retained for backward compatibility.
May be removed in a future version.
"""

import warnings
from datetime import datetime
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, ConfigDict, Field, computed_field

from schemas import Asset


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
