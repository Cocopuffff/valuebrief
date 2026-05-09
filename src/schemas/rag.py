"""
rag.py
~~~~~~
Hybrid RAG schema models: vault documents, artifacts, citations, vector memory records,
and thesis pillar types for the pillar-based memory layer.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import Field, field_validator

from schemas import StrictBaseModel, Ticker


# ── Pillar type literals ─────────────────────────────────────────────────────

PillarType = Literal[
    "moat",
    "growth",
    "risk",
    "valuation_assumption",
    "capital_allocation",
    "thesis_change",
]

PillarStatus = Literal[
    "supported",
    "weakened",
    "superseded",
    "contradicted",
    "stale",
]

PillarOutcomeStatus = Literal[
    "supported",
    "weakened",
    "revised",
    "contradicted",
    "stale",
]


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


class ResearchTopic(StrictBaseModel):
    """A structured research question assigned to a specific analyst side."""
    side: str  # "bull", "bear", or "neutral"
    question: str
    rationale: str
    evidence_memory_ids: list[str] = Field(default_factory=list)


class ThesisPillar(StrictBaseModel):
    """A curated investment-thesis pillar — the unit of vector memory retrieval.

    Each pillar represents one enduring claim about the business (moat, growth,
    risk, valuation assumption, capital allocation, or thesis change).  Pillars
    are created/evolved by the Judge and persisted by the Curator as individual
    vector memories with ``source_type = "thesis_pillar"``.
    """
    pillar_id: str = ""  # System-assigned stable identifier.
    candidate_ref: str = ""  # Judge-local reference for unresolved candidates.
    matched_prior_ref: str = ""  # Prior pillar ref selected by the Judge, if any.
    matched_pillar_id: str = ""  # Existing pillar_id selected by system/LLM adjudication.
    pillar_type: PillarType
    statement: str  # The core claim (1-2 sentences) — stored as memory.summary
    rationale: str = ""  # Supporting evidence and reasoning
    valuation_impact: str = ""  # How this pillar affects intrinsic value
    source_urls: list[str] = Field(default_factory=list)
    evidence_citations: list[str] = Field(default_factory=list)  # Vault block citations
    resurrection_reason: str = ""  # Required when reviving a similar inactive pillar.
    status: PillarStatus = "supported"


class PillarOutcome(StrictBaseModel):
    """The Judge's evaluation of a prior pillar after fresh research.

    Produced during the reconcile step and consumed by the Curator to update
    memory validity_status and handle transitions (supersede, demote, etc.).
    """
    memory_id: str  # UUID of the prior pillar memory in investment_memories
    pillar_id: str  # Matches ThesisPillar.pillar_id
    status: PillarOutcomeStatus  # Judge's determination after reviewing fresh evidence
    reason: str = ""  # Why the status was assigned
    replacement_statement: str = ""  # New statement if status is "revised"
    source_urls: list[str] = Field(default_factory=list)  # Evidence URLs

    @field_validator("status", mode="before")
    @classmethod
    def map_legacy_updated(cls, value: str) -> str:
        """Accept legacy Judge output while never storing ``updated``."""
        if str(value).strip().lower() == "updated":
            return "revised"
        return value
