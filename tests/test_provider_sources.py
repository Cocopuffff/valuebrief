import asyncio
from pathlib import Path

import pytest

from provider import FinancialDataProvider


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


@pytest.fixture(autouse=True)
def reset_sec_ticker_cache():
    FinancialDataProvider._sec_ticker_cache = None
    FinancialDataProvider._sec_ticker_cache_sec_refresh_attempted = False
    yield
    FinancialDataProvider._sec_ticker_cache = None
    FinancialDataProvider._sec_ticker_cache_sec_refresh_attempted = False


def test_load_sec_ticker_cache_async_uses_supabase_before_sec_download(monkeypatch):
    async def fake_supabase_load():
        return {"GNC": "0000123456"}

    async def fail_download():
        raise AssertionError("Unexpected SEC download")

    monkeypatch.setattr(
        FinancialDataProvider,
        "_load_sec_ticker_cache_from_supabase_async",
        staticmethod(fake_supabase_load),
    )
    monkeypatch.setattr(
        FinancialDataProvider,
        "_download_sec_ticker_map_async",
        staticmethod(fail_download),
    )

    assert asyncio.run(FinancialDataProvider.load_sec_ticker_cache_async()) == {
        "GNC": "0000123456"
    }


def test_load_sec_ticker_cache_async_downloads_and_persists_when_supabase_empty(monkeypatch):
    persisted = []

    async def fake_supabase_load():
        return {}

    async def fake_save(records):
        persisted.extend(records)

    monkeypatch.setattr(
        FinancialDataProvider,
        "_load_sec_ticker_cache_from_supabase_async",
        staticmethod(fake_supabase_load),
    )
    monkeypatch.setattr(
        FinancialDataProvider,
        "_save_sec_company_tickers_to_supabase_async",
        staticmethod(fake_save),
    )

    def fake_get(url, **kwargs):
        assert url.endswith("company_tickers.json")
        return FakeResponse({
            "0": {"ticker": "GNC", "cik_str": 123456, "title": "Generic Corp Inc"},
        })

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            return fake_get(url)

    monkeypatch.setattr("provider.httpx.AsyncClient", FakeAsyncClient)

    assert asyncio.run(FinancialDataProvider.load_sec_ticker_cache_async()) == {
        "GNC": "0000123456"
    }
    assert persisted == [("GNC", "0000123456", "Generic Corp Inc")]


def test_sec_cik_for_ticker_async_refreshes_when_supabase_cache_misses(monkeypatch):
    async def fake_supabase_load():
        return {"OLD": "0000000001"}

    async def fake_save(records):
        return None

    monkeypatch.setattr(
        FinancialDataProvider,
        "_load_sec_ticker_cache_from_supabase_async",
        staticmethod(fake_supabase_load),
    )
    monkeypatch.setattr(
        FinancialDataProvider,
        "_save_sec_company_tickers_to_supabase_async",
        staticmethod(fake_save),
    )

    def fake_get(url, **kwargs):
        assert url.endswith("company_tickers.json")
        return FakeResponse({
            "0": {"ticker": "GNC", "cik_str": 123456, "title": "Generic Corp Inc"},
        })

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            return fake_get(url)

    monkeypatch.setattr("provider.httpx.AsyncClient", FakeAsyncClient)

    assert asyncio.run(FinancialDataProvider._sec_cik_for_ticker_async("GNC")) == "0000123456"


def test_get_sec_filing_records_parses_recent_filings(monkeypatch):
    async def fake_supabase_load():
        return {}

    async def fake_save(records):
        return None

    monkeypatch.setattr(
        FinancialDataProvider,
        "_load_sec_ticker_cache_from_supabase_async",
        staticmethod(fake_supabase_load),
    )
    monkeypatch.setattr(
        FinancialDataProvider,
        "_save_sec_company_tickers_to_supabase_async",
        staticmethod(fake_save),
    )

    def fake_get(url, **kwargs):
        if url.endswith("company_tickers.json"):
            return FakeResponse({
                "0": {"ticker": "GNC", "cik_str": 123456},
            })
        assert "CIK0000123456.json" in url
        return FakeResponse({
            "name": "Generic Corp Inc",
            "filings": {
                "recent": {
                    "form": ["10-Q", "8-K", "10-K"],
                    "filingDate": ["2026-05-01", "2026-04-15", "2026-02-28"],
                    "reportDate": ["2026-03-31", "2026-04-15", "2025-12-31"],
                    "accessionNumber": [
                        "0000123456-26-000010",
                        "0000123456-26-000009",
                        "0000123456-26-000001",
                    ],
                    "primaryDocument": ["gnc-20260331.htm", "gnc-8k.htm", "gnc-20251231.htm"],
                }
            },
        })

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            return fake_get(url)

    monkeypatch.setattr("provider.httpx.AsyncClient", FakeAsyncClient)

    records = FinancialDataProvider._get_sec_filing_records(
        "GNC",
        forms=["10-K", "10-Q"],
        limit=5,
    )

    assert [record["form_type"] for record in records] == ["10-Q", "10-K"]
    assert records[0]["accession_number"] == "0000123456-26-000010"
    assert records[0]["url"].endswith("/000012345626000010/gnc-20260331.htm")


def test_get_sec_filing_records_sync_wrapper_works_inside_running_loop(monkeypatch):
    async def fake_async_records(ticker, forms=None, limit=10):
        return [{"ticker": ticker.upper(), "form_type": "10-K"}]

    monkeypatch.setattr(
        FinancialDataProvider,
        "_get_sec_filing_records_async",
        staticmethod(fake_async_records),
    )

    async def call_sync_wrapper():
        return FinancialDataProvider._get_sec_filing_records("gnc")

    assert asyncio.run(call_sync_wrapper()) == [{"ticker": "GNC", "form_type": "10-K"}]


def test_main_prewarm_uses_async_cache_loader():
    main_source = Path(__file__).resolve().parents[1] / "src" / "main.py"
    text = main_source.read_text()

    assert "await FinancialDataProvider.load_sec_ticker_cache_async()" in text
    assert "asyncio.to_thread(FinancialDataProvider._load_sec_ticker_cache)" not in text


def test_discover_earnings_call_transcripts_normalizes_results(monkeypatch):
    class FakeDDGS:
        def text(self, query, region, max_results):
            assert "GNC" in query
            return [
                {
                    "title": "Generic Corp Q1 2026 Earnings Call Transcript",
                    "href": "https://investors.example.com/events/q1-transcript",
                    "body": "Prepared remarks and Q&A.",
                }
            ]

    monkeypatch.setattr("provider.DDGS", FakeDDGS)

    records = FinancialDataProvider.discover_earnings_call_transcripts.invoke({
        "ticker": "GNC",
        "company": "Generic Corp Inc",
        "limit": 1,
    })

    assert records[0]["source_type"] == "earnings_transcript"
    assert records[0]["is_company_ir"] is True
    assert records[0]["url"] == "https://investors.example.com/events/q1-transcript"
