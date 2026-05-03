"""
provider.py
~~~~~~~~~~~
Schema models for third-party search/news/scraper results with tolerant validation.
"""

from typing import Optional

from pydantic import ConfigDict, Field

from schemas import StrictBaseModel


class SearchResult(StrictBaseModel):
    """Validated DDGS text search result. Tolerates unknown fields from DDGS."""
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    title: str
    url: str = Field(..., alias="href")
    snippet: str = ""
    source: str = ""


class NewsResult(StrictBaseModel):
    """Validated DDGS news result. Tolerates unknown fields from DDGS."""
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    published_at: Optional[str] = None


class ScrapeResult(StrictBaseModel):
    """Result of scraping a single URL."""
    url: str
    content: str
    truncated: bool = False
