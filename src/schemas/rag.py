"""
rag.py
~~~~~~
Hybrid RAG schema models: vault documents, artifacts, citations, and vector memory records.
"""

from datetime import datetime
from typing import Optional

from pydantic import Field

from schemas import StrictBaseModel, Ticker


class VaultFrontmatter(StrictBaseModel):
    """YAML frontmatter fields for a vault Markdown file."""
    model_config = {"extra": "ignore"}
    url: str = ""
    source_type: str = "analysis"
    date_scraped: str = ""
    sentiment: Optional[str] = None
    block_count: int = 0
    content_hash: str = ""
    archived: bool = False


class VaultDocument(StrictBaseModel):
    """In-memory representation of a vault Markdown file."""
    path: str
    ticker: Ticker
    url: Optional[str] = None
    source_type: str = "analysis"
    date_scraped: str = ""
    sentiment: Optional[str] = None
    content: str = ""
    block_map: dict[str, str] = Field(default_factory=dict)
    content_hash: str = ""
    archived: bool = False


class VaultArtifact(StrictBaseModel):
    """Metadata for a persisted research artifact (returned by persist_research_artifact)."""
    ticker: Ticker
    path: str
    filename: str
    source_type: str
    block_ids: list[str] = Field(default_factory=list)
    block_memory_ids: dict[str, str] = Field(default_factory=dict)
    memory_ids: list[str] = Field(default_factory=list)
    vector_error: str = ""


class CitationRef(StrictBaseModel):
    """A single resolved citation reference."""
    file_path: str
    block_id: str
    resolved_text: Optional[str] = None


class InsightRecord(StrictBaseModel):
    """An insight to be stored as a vector in Supabase."""
    ticker: Ticker
    summary: str
    embedding: list[float]
    metadata: dict = Field(default_factory=dict)
    source_priority: int = 0
    is_cited: bool = False


class MemoryRecord(StrictBaseModel):
    """A retrieved memory from Supabase."""
    id: str
    summary: str
    metadata: dict
    ticker: Ticker
    source_priority: int
    is_cited: bool
    created_at: datetime
    updated_at: datetime
    similarity: Optional[float] = None
