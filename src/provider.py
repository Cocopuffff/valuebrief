import asyncio
import os
import re
import threading
import yfinance as yf
from typing import List, Optional, Any, Coroutine
from datetime import date, datetime
from schemas import Asset, FinancialMetrics, SearchResult, NewsResult
from langchain.tools import tool
from ddgs import DDGS
import httpx
from markdownify import markdownify
from utils.logger import get_logger
from utils.config import exchange_mappings
from utils.db import get_pool

logger = get_logger(__name__)

class FinancialDataProvider:
    _sec_ticker_cache: dict[str, str] | None = None
    _sec_ticker_cache_sec_refresh_attempted = False
    _SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    @staticmethod
    def translate_ticker_for_av(ticker: str) -> str:
        """Translate a Yahoo Finance ticker to AlphaVantage format using exchange mappings."""
        av_ticker = ticker
        if '.' in ticker:
            base, suffix = ticker.rsplit('.', 1)
            suffix = f".{suffix}"
            if suffix in exchange_mappings:
                av_ticker = f"{base}{exchange_mappings[suffix]}"
        return av_ticker

    @staticmethod
    def _get_asset_data(ticker: str) -> Optional[Asset]:
        """Fetch data from AlphaVantage and yfinance and return an Asset instance."""
        logger.info(f'Fetching data for {ticker}...')

        av_ticker = FinancialDataProvider.translate_ticker_for_av(ticker)
        if av_ticker != ticker:
            logger.debug(f"Translated ticker {ticker} to {av_ticker} for AlphaVantage")

        av_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        av_overview = {}
        av_quote = {}

        if av_api_key:
            try:
                # Fetch Overview
                url_overview = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={av_ticker}&apikey={av_api_key}"
                resp_overview = httpx.get(url_overview, timeout=10)
                resp_overview.raise_for_status()
                data_overview = resp_overview.json()
                if "Symbol" in data_overview:
                    av_overview = data_overview
                else:
                    logger.warning(f"AlphaVantage OVERVIEW error for {av_ticker}: {data_overview}")

                # Fetch Global Quote
                url_quote = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={av_ticker}&apikey={av_api_key}"
                resp_quote = httpx.get(url_quote, timeout=10)
                resp_quote.raise_for_status()
                data_quote = resp_quote.json()
                if "Global Quote" in data_quote and "05. price" in data_quote["Global Quote"]:
                    av_quote = data_quote["Global Quote"]
                else:
                    logger.warning(f"AlphaVantage GLOBAL_QUOTE error for {av_ticker}: {data_quote}")
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
    
    @staticmethod
    def _sec_headers() -> dict[str, str]:
        user_agent = os.getenv(
            "SEC_USER_AGENT",
            "ValueBrief research bot contact@example.com",
        )
        return {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}

    @staticmethod
    def _run_async_blocking(coro: Coroutine[Any, Any, Any]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: dict[str, Any] = {}

        def runner() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as e:
                result["exception"] = e

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()

        if "exception" in result:
            raise result["exception"]
        return result.get("value")

    @staticmethod
    def _parse_sec_company_tickers(raw: Any) -> list[tuple[str, str, str]]:
        records: list[tuple[str, str, str]] = []
        for item in raw.values():
            ticker = str(item.get("ticker", "")).strip().upper()
            cik_raw = item.get("cik_str")
            if not ticker or cik_raw in (None, ""):
                continue

            cik = str(cik_raw).strip()
            if not cik.isdigit():
                continue

            company_name = str(item.get("title") or "").strip()
            records.append((ticker, cik.zfill(10), company_name))
        return records

    @staticmethod
    def _records_to_sec_ticker_map(records: list[tuple[str, str, str]]) -> dict[str, str]:
        return {ticker: cik for ticker, cik, _ in records if ticker and cik}

    @staticmethod
    async def _load_sec_ticker_cache_from_supabase_async() -> dict[str, str]:
        try:
            pool = await get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT ticker, cik FROM sec_company_tickers")
                    rows = await cur.fetchall()
        except Exception as e:
            logger.warning("[Provider] Failed to load SEC ticker map from Supabase: %s", e)
            return {}

        mapping = {str(ticker).upper(): str(cik).zfill(10) for ticker, cik in rows}
        if mapping:
            logger.info("[Provider] Loaded %d SEC ticker mappings from Supabase", len(mapping))
        return mapping

    @staticmethod
    async def _save_sec_company_tickers_to_supabase_async(
        records: list[tuple[str, str, str]],
    ) -> None:
        if not records:
            return

        try:
            pool = await get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        """
                        INSERT INTO sec_company_tickers (ticker, cik, company_name, updated_at)
                        VALUES (%s, %s, %s, NOW())
                        ON CONFLICT (ticker) DO UPDATE SET
                            cik = EXCLUDED.cik,
                            company_name = EXCLUDED.company_name,
                            updated_at = NOW()
                        """,
                        records,
                    )
            logger.info("[Provider] Persisted %d SEC ticker mappings to Supabase", len(records))
        except Exception as e:
            logger.warning("[Provider] Failed to persist SEC ticker map to Supabase: %s", e)

    @staticmethod
    async def _download_sec_ticker_map_async() -> dict[str, str]:
        try:
            async with httpx.AsyncClient(
                headers=FinancialDataProvider._sec_headers(),
                timeout=10,
            ) as client:
                response = await client.get(FinancialDataProvider._SEC_COMPANY_TICKERS_URL)
            response.raise_for_status()
            records = FinancialDataProvider._parse_sec_company_tickers(response.json())
            mapping = FinancialDataProvider._records_to_sec_ticker_map(records)
            if mapping:
                await FinancialDataProvider._save_sec_company_tickers_to_supabase_async(records)
            return mapping
        except Exception as e:
            logger.warning("[Provider] Failed to load SEC ticker map: %s", e)
            return {}

    @staticmethod
    async def load_sec_ticker_cache_async() -> dict[str, str]:
        """Load SEC ticker-to-CIK mapping from Supabase, falling back to SEC."""
        if FinancialDataProvider._sec_ticker_cache is not None:
            return FinancialDataProvider._sec_ticker_cache

        mapping = await FinancialDataProvider._load_sec_ticker_cache_from_supabase_async()
        if not mapping:
            logger.info("[Provider] SEC ticker map not found in Supabase; downloading from SEC")
            FinancialDataProvider._sec_ticker_cache_sec_refresh_attempted = True
            mapping = await FinancialDataProvider._download_sec_ticker_map_async()

        FinancialDataProvider._sec_ticker_cache = mapping
        return mapping

    @staticmethod
    def _load_sec_ticker_cache() -> dict[str, str]:
        """Synchronous compatibility wrapper for LangChain tool callers."""
        return FinancialDataProvider._run_async_blocking(
            FinancialDataProvider.load_sec_ticker_cache_async()
        )

    @staticmethod
    async def _sec_cik_for_ticker_async(ticker: str) -> str:
        base_ticker = ticker.split(".", 1)[0].upper()
        mapping = await FinancialDataProvider.load_sec_ticker_cache_async()
        cik = mapping.get(base_ticker, "")
        if cik or FinancialDataProvider._sec_ticker_cache_sec_refresh_attempted:
            return cik

        logger.info(
            "[Provider] %s missing from persisted SEC ticker map; refreshing from SEC",
            base_ticker,
        )
        FinancialDataProvider._sec_ticker_cache_sec_refresh_attempted = True
        refreshed = await FinancialDataProvider._download_sec_ticker_map_async()
        if refreshed:
            FinancialDataProvider._sec_ticker_cache = refreshed
            return refreshed.get(base_ticker, "")
        return ""

    @staticmethod
    def _sec_cik_for_ticker(ticker: str) -> str:
        """Synchronous compatibility wrapper for LangChain tool callers."""
        return FinancialDataProvider._run_async_blocking(
            FinancialDataProvider._sec_cik_for_ticker_async(ticker)
        )

    @staticmethod
    async def _get_sec_filing_records_async(
        ticker: str,
        forms: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        cik = await FinancialDataProvider._sec_cik_for_ticker_async(ticker)
        if not cik:
            logger.info("[Provider] No SEC CIK found for %s", ticker)
            return []

        form_filter = {f.upper() for f in (forms or ["10-K", "10-Q", "8-K"])}
        try:
            async with httpx.AsyncClient(
                headers=FinancialDataProvider._sec_headers(),
                timeout=10,
            ) as client:
                response = await client.get(f"https://data.sec.gov/submissions/CIK{cik}.json")
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning("[Provider] SEC submissions lookup failed for %s: %s", ticker, e)
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms_raw = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_documents = recent.get("primaryDocument", [])

        results: list[dict[str, Any]] = []
        entity_name = data.get("name", "")
        cik_int = str(int(cik))
        for idx, form in enumerate(forms_raw):
            form = str(form).upper()
            if form not in form_filter:
                continue

            accession = str(accession_numbers[idx]) if idx < len(accession_numbers) else ""
            primary_doc = str(primary_documents[idx]) if idx < len(primary_documents) else ""
            accession_path = accession.replace("-", "")
            if primary_doc:
                url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_path}/{primary_doc}"
            else:
                url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_path}/"

            results.append({
                "ticker": ticker.upper(),
                "company": entity_name,
                "source_type": "sec_filing",
                "form_type": form,
                "filing_date": filing_dates[idx] if idx < len(filing_dates) else "",
                "report_date": report_dates[idx] if idx < len(report_dates) else "",
                "accession_number": accession,
                "primary_document": primary_doc,
                "url": url,
            })
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _get_sec_filing_records(
        ticker: str,
        forms: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Synchronous compatibility wrapper for LangChain tool callers."""
        return FinancialDataProvider._run_async_blocking(
            FinancialDataProvider._get_sec_filing_records_async(
                ticker=ticker,
                forms=forms,
                limit=limit,
            )
        )

    @tool
    @staticmethod
    def get_sec_filings(
        ticker: str,
        forms: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Fetch recent official SEC filings for a US-listed ticker.

        Args:
            ticker: Stock ticker, e.g. AAPL.
            forms: SEC form types to include, e.g. ["10-K", "10-Q", "8-K"].
            limit: Maximum number of filings to return.
        """
        if isinstance(forms, str):
            forms = [forms]
        return FinancialDataProvider._get_sec_filing_records(
            ticker=ticker,
            forms=forms,
            limit=limit,
        )

    @tool
    @staticmethod
    def discover_earnings_call_transcripts(
        ticker: str,
        company: str = "",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Discover earnings-call transcript sources using web search.

        Prefer company investor-relations pages when they appear in results.
        Returns titles, URLs, snippets, and source metadata; use scrape_website
        on promising URLs to read the transcript content.
        """
        company_part = f'"{company}" ' if company else ""
        query = (
            f'{company_part}{ticker.upper()} earnings call transcript '
            "investor relations 10-Q 10-K latest"
        )
        logger.info('[Provider] Discovering earnings transcripts with "%s"', query)
        try:
            raw_results = list(DDGS().text(query, region="us-en", max_results=max(10, limit)))
        except Exception as e:
            logger.error("[Provider] Error discovering transcripts for %s: %s", ticker, e)
            return []

        records: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        ir_pattern = re.compile(r"(investor|ir\.|investors|quarterly-results|events)", re.I)
        for item in raw_results:
            normalized = dict(item)
            url = str(normalized.get("href") or normalized.get("url") or "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = str(normalized.get("title", ""))
            snippet = str(normalized.get("body") or normalized.get("snippet") or "")
            records.append({
                "ticker": ticker.upper(),
                "company": company,
                "source_type": "earnings_transcript",
                "title": title,
                "url": url,
                "snippet": snippet,
                "source": str(normalized.get("source", "")),
                "is_company_ir": bool(ir_pattern.search(url) or ir_pattern.search(title)),
            })
            if len(records) >= limit:
                break
        return records
    
    @tool
    @staticmethod
    def get_latest_news(query: str) -> list[dict[str, Any]]:
        """Gets latest news within the last 24 hours. Returns up to 10 headlines with titles, URLs, and short snippets. To read the full article content, use scrape_website with the article URL."""
        logger.info(f'Fetching news within last 24 hours for query "{query}"...')
        try:
            raw_news = list(DDGS().news(
                query,
                region='us-en',
                max_results=10,
                timelimit="d"
            ))
        except Exception as e:
            logger.error(f"Error fetching news for {query}: {e}")
            return []

        validated: list[dict[str, Any]] = []
        for item in raw_news:
            try:
                normalized = dict(item)
                # DDGS news may put the URL in 'url' or 'link'
                if "url" not in normalized and "link" in normalized:
                    normalized["url"] = normalized.pop("link")
                validated.append(
                    NewsResult.model_validate(normalized).model_dump()
                )
            except Exception as e:
                logger.warning(
                    "[Provider] Skipping malformed news result for '%s': %s — %s",
                    query, item.get("title", "?"), e,
                )
        logger.info(f'{len(validated)} news retrieved (skipped {len(raw_news) - len(validated)})')
        return validated

    @tool
    @staticmethod
    def search(query: str) -> list[dict[str, Any]]:
        """Search the internet for information. Returns up to 10 results with titles, URLs, and short snippets. To read the full page content, use scrape_website with the page URL."""
        logger.info(f'Searching "{query}"...')
        try:
            raw_results = list(DDGS().text(
                query,
                region='us-en',
                max_results=10
            ))
        except Exception as e:
            logger.error(f"Error searching for {query}: {e}")
            return []

        validated: list[dict[str, Any]] = []
        for item in raw_results:
            try:
                normalized = dict(item)
                # DDGS text search uses 'href'; SearchResult aliases it to 'url'
                validated.append(
                    SearchResult.model_validate(normalized).model_dump()
                )
            except Exception as e:
                logger.warning(
                    "[Provider] Skipping malformed search result for '%s': %s — %s",
                    query, item.get("title", "?"), e,
                )
        logger.info(f'{len(validated)} searches retrieved (skipped {len(raw_results) - len(validated)})')
        return validated

    @staticmethod
    def _scrape_url(url: str, max_chars: int = 5000) -> str:
        """Internal scraping method. Truncates to max_chars to keep LLM context manageable."""
        logger.info(f"Scraping URL: {url}...")
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
