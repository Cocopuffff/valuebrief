import yfinance as yf
import pandas as pd
from typing import List, Optional, Any
from datetime import date, datetime
from models import Asset, FinancialMetrics
from langchain.tools import tool
from ddgs import DDGS
import httpx
from markdownify import markdownify

class FinancialDataProvider:
    @staticmethod
    def _get_asset_data(ticker: str) -> Optional[Asset]:
        """Fetch data from yfinance using the stock ticker and return an Asset instance."""
        print(f'Fetching data for {ticker}...')
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extracting current price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        
        if current_price is None:
            print(f"Skipping {ticker} because no price data was found.")
            return None
        
        # Financial metrics
        fundamentals = FinancialMetrics(
            pe_ratio=info.get('trailingPE'),
            peg_ratio=info.get('pegRatio'),
            price_to_book=info.get('priceToBook'),
            debt_to_equity=info.get('debtToEquity'),
            dividend_yield=info.get('dividendYield'),
            free_cash_flow=info.get('freeCashflow'),
            revenue_growth=info.get('revenueGrowth'),
            ebitda_margin=info.get('ebitdaMargins')
        )

        asset = Asset(
            ticker=ticker,
            name=info.get('longName'),
            sector=info.get('sector'),
            industry=info.get('industry'),
            current_price=current_price,
            fundamentals=fundamentals,
            last_updated=datetime.now()
        )
        print(f'Financial data retrieved for {ticker}:\n{asset}')
        return asset

    @tool
    @staticmethod
    def get_asset_data(ticker: str) -> Optional[Asset]:
        """Fetch data from yfinance using the stock ticker and return an Asset instance."""
        return FinancialDataProvider._get_asset_data(ticker)

    @tool
    @staticmethod
    def get_multiple_assets(tickers: List[str]) -> List[Asset]:
        """Fetch multiple assets from yfinance using the stock tickers and return a list of Asset instances."""
        assets = []
        print(f'Fetching data for {tickers}...')
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
        print(f'Fetching news within last 24 hours for query "{query}"...')
        news : list[dict[str, Any]] = DDGS().news(
            query,
            region='us-en',
            max_results=10,
            timelimit="1d"
        )
        print(f'{len(news)} news retrieved')
        for article in news:
            print(f"Title: {article.get('title')}")
            print(f"Link: {article.get('url', article.get('href'))}")
        
        return news
    
    @tool
    @staticmethod
    def search(query: str) -> list[dict[str, Any]]:
        """Search the internet for information. Returns up to 10 results with titles, URLs, and short snippets. To read the full page content, use scrape_website with the page URL."""
        print(f'Searching "{query}"...')
        searches = DDGS().text(
            query,
            region='us-en',
            max_results=10
        )
        print(f'{len(searches)} searches retrieved')
        for article in searches:
            print(f"Title: {article.get('title')}")
            print(f"Link: {article.get('href')}")
        
        return searches

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