import os
import yfinance as yf
import pandas as pd
from typing import List, Optional, Any
from datetime import date, datetime
from models import Asset, FinancialMetrics
from langchain.tools import tool
from ddgs import DDGS
import httpx
from markdownify import markdownify
from logger import get_logger

logger = get_logger(__name__)

class FinancialDataProvider:
    @staticmethod
    def _get_asset_data(ticker: str) -> Optional[Asset]:
        """Fetch data from AlphaVantage and yfinance and return an Asset instance."""
        logger.info(f'Fetching data for {ticker}...')

        av_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        av_overview = {}
        av_quote = {}

        if av_api_key:
            try:
                # Fetch Overview
                url_overview = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={av_api_key}"
                resp_overview = httpx.get(url_overview, timeout=10)
                resp_overview.raise_for_status()
                data_overview = resp_overview.json()
                if "Symbol" in data_overview:
                    av_overview = data_overview
                else:
                    logger.warning(f"AlphaVantage OVERVIEW error for {ticker}: {data_overview}")

                # Fetch Global Quote
                url_quote = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={av_api_key}"
                resp_quote = httpx.get(url_quote, timeout=10)
                resp_quote.raise_for_status()
                data_quote = resp_quote.json()
                if "Global Quote" in data_quote and "05. price" in data_quote["Global Quote"]:
                    av_quote = data_quote["Global Quote"]
                else:
                    logger.warning(f"AlphaVantage GLOBAL_QUOTE error for {ticker}: {data_quote}")
            except Exception as e:
                logger.warning(f"Failed to fetch AlphaVantage data for {ticker}: {e}")

        stock = yf.Ticker(ticker)
        info = stock.info

        def _get_float_metric(
            av_key: str,
            yf_key: str,
            av_dict: Optional[dict] = None,
        ) -> Optional[float]:
            """Fetch a numeric metric from AlphaVantage (primary) with yfinance fallback.

            Args:
                av_key: AlphaVantage response key.
                yf_key: yfinance info dict key.
                av_dict: Override AV source dict (e.g. av_quote for price).
            """
            av_val = None
            if av_dict is None:
                av_dict = av_overview
            if av_dict and av_key in av_dict:
                val = av_dict[av_key]
                if val not in ("None", "-", ""):
                    try:
                        av_val = float(val)
                    except ValueError:
                        pass
            
            yf_val_raw = info.get(yf_key)
            try:
                yf_val = float(yf_val_raw) if yf_val_raw is not None else None
            except ValueError:
                yf_val = None

            if av_val is not None:
                if av_val == 0.0:
                    if yf_val is not None and yf_val != 0.0:
                        logger.debug(f"AV {av_key} is 0.0; falling back to yfinance {yf_key}={yf_val}")
                        return yf_val
                    else:
                        # yfinance is also 0.0 or None, so we stick with AV's 0.0
                        return av_val
                return av_val

            # AlphaVantage totally missing, use whatever yfinance has
            return yf_val

        # Extracting current price
        current_price = _get_float_metric("05. price", "currentPrice", av_quote)
        if current_price is None:
            current_price = info.get('regularMarketPrice') or info.get('previousClose')
        
        if current_price is None:
            logger.warning(f"Skipping {ticker} because no price data was found.")
            return None
        
        # Financial metrics
        fundamentals = FinancialMetrics(
            pe_ratio=_get_float_metric("TrailingPE", "trailingPE"),
            forward_pe_ratio=_get_float_metric("ForwardPE", "forwardPE"),
            peg_ratio=_get_float_metric("PEGRatio", "pegRatio"),
            price_to_book=_get_float_metric("PriceToBookRatio", "priceToBook"),
            debt_to_equity=info.get('debtToEquity'),
            dividend_yield=_get_float_metric("DividendYield", "dividendYield"),
            free_cash_flow=info.get('freeCashflow'),
            revenue_growth=_get_float_metric("QuarterlyRevenueGrowthYOY", "revenueGrowth"),
            ebitda_margin=_get_float_metric("ProfitMargin", "ebitdaMargins"),
            earnings_per_share=_get_float_metric("EPS", "trailingEps"),
            market_cap=_get_float_metric("MarketCapitalization", "marketCap"),
            return_on_equity=_get_float_metric("ReturnOnEquityTTM", "returnOnEquity"),
            current_ratio=info.get('currentRatio'),
            total_revenue=_get_float_metric("RevenueTTM", "totalRevenue"),
            profit_margin=_get_float_metric("ProfitMargin", "profitMargins"),
            operating_margin=_get_float_metric("OperatingMarginTTM", "operatingMargins"),
            return_on_assets=_get_float_metric("ReturnOnAssetsTTM", "returnOnAssets"),
            price_to_sales=_get_float_metric("PriceToSalesRatioTTM", "priceToSalesTrailing12Months"),
            ev_to_revenue=_get_float_metric("EVToRevenue", "enterpriseToRevenue"),
            ev_to_ebitda=_get_float_metric("EVToEBITDA", "enterpriseToEbitda"),
            beta=_get_float_metric("Beta", "beta"),
            target_price=_get_float_metric("AnalystTargetPrice", "targetMeanPrice"),
            fifty_two_week_high=_get_float_metric("52WeekHigh", "fiftyTwoWeekHigh"),
            fifty_two_week_low=_get_float_metric("52WeekLow", "fiftyTwoWeekLow"),
            fifty_day_moving_average=_get_float_metric("50DayMovingAverage", "fiftyDayAverage"),
            two_hundred_day_moving_average=_get_float_metric("200DayMovingAverage", "twoHundredDayAverage"),
        )

        name = av_overview.get("Name") if av_overview.get("Name") not in (None, "None", "-") else info.get('longName')
        sector = av_overview.get("Sector") if av_overview.get("Sector") not in (None, "None", "-") else info.get('sector')
        industry = av_overview.get("Industry") if av_overview.get("Industry") not in (None, "None", "-") else info.get('industry')

        asset = Asset(
            ticker=ticker,
            name=name,
            sector=sector,
            industry=industry,
            current_price=float(current_price),
            shares_outstanding=_get_float_metric("SharesOutstanding", "sharesOutstanding"),
            fundamentals=fundamentals,
            last_updated=datetime.now()
        )
        logger.debug(f'Financial data retrieved for {ticker}: {asset}')
        return asset

    @tool
    @staticmethod
    def get_asset_data(ticker: str) -> Optional[Asset]:
        """Fetch data from AlphaVantage and yfinance using the stock ticker and return an Asset instance."""
        return FinancialDataProvider._get_asset_data(ticker)

    @tool
    @staticmethod
    def get_multiple_assets(tickers: List[str]) -> List[Asset]:
        """Fetch multiple assets from AlphaVantage and yfinance using the stock tickers and return a list of Asset instances."""
        assets = []
        logger.info(f'Fetching data for {tickers}...')
        for ticker in tickers:
            asset = FinancialDataProvider._get_asset_data(ticker)
            if asset is not None:
                assets.append(asset)
        return assets
    
    @tool
    @staticmethod
    def get_sec_filings(ticker: str) -> str:
        """
        Returns SEC filings for ticker
        """
        return "filing"
    
    @tool
    @staticmethod
    def get_latest_news(query: str) -> list[dict[str, Any]]:
        """Gets latest news within the last 24 hours. Returns up to 10 headlines with titles, URLs, and short snippets. To read the full article content, use scrape_website with the article URL."""
        logger.info(f'Fetching news within last 24 hours for query "{query}"...')
        try:
            news = list(DDGS().news(
                query,
                region='us-en',
                max_results=10,
                timelimit="d"
            ))
            logger.info(f'{len(news)} news retrieved')
            for article in news:
                logger.debug(f"Title: {article.get('title')}, Link: {article.get('url', article.get('href'))}")
            return news
        except Exception as e:
            logger.error(f"Error fetching news for {query}: {e}")
            return []
    
    @tool
    @staticmethod
    def search(query: str) -> list[dict[str, Any]]:
        """Search the internet for information. Returns up to 10 results with titles, URLs, and short snippets. To read the full page content, use scrape_website with the page URL."""
        logger.info(f'Searching "{query}"...')
        try:
            searches = list(DDGS().text(
                query,
                region='us-en',
                max_results=10
            ))
            logger.info(f'{len(searches)} searches retrieved')
            for article in searches:
                logger.debug(f"Title: {article.get('title')}, Link: {article.get('href')}")
            return searches
        except Exception as e:
            logger.error(f"Error searching for {query}: {e}")
            return []

    @staticmethod
    def _scrape_url(url: str, max_chars: int = 5000) -> str:
        """Internal scraping method. Truncates to max_chars to keep LLM context manageable."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            response: httpx.Response = httpx.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            text = markdownify(response.text)
            if len(text) > max_chars:
                text = text[:max_chars] + "... [truncated]"
            return text
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return f"Error fetching {url}: {str(e)}"

    @tool
    @staticmethod
    def scrape_website(url: str) -> str:
        """Scrape a website for article content based on input URL"""
        return FinancialDataProvider._scrape_url(url)


class ValueAnalysis:
    @tool
    @staticmethod
    def graham_formula(earnings: float, growth: float) -> float:
        """Simplified Graham's intrinsic value formula: V = E * (8.5 + 2g)"""
        # V: Intrinsic value, E: Earnings per share, g: expected growth rate
        return earnings * (8.5 + 2 * growth)

    @tool
    @staticmethod
    def calculate_margin_of_safety(current_price: float, intrinsic_value: float) -> float:
        """Calculate the margin of safety given current price and intrinsic value."""
        if intrinsic_value <= 0:
            return 0.0
        return (intrinsic_value - current_price) / intrinsic_value

class DateTimeProvider:
    @tool
    @staticmethod
    def get_current_date() -> str:
        """returns current date"""
        return date.today().isoformat()