"""
finance.py
~~~~~~~~~~
Finance domain models: FinancialMetrics and Asset.
"""

from datetime import datetime
from typing import Optional

from pydantic import Field

from schemas import StrictBaseModel, Ticker


class FinancialMetrics(StrictBaseModel):
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


class Asset(StrictBaseModel):
    """Snapshot of a company's market data and fundamentals."""
    ticker: Ticker = Field(..., description="Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'GOOGL')")
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
