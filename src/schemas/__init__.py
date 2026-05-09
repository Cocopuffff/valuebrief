"""
schemas
~~~~~~~
Bounded-context Pydantic schema package for the ValueBrief project.

Import from submodules directly (e.g. ``from schemas.finance import Asset``)
or use the curated re-exports here.
"""

from schemas.base import StrictBaseModel as StrictBaseModel, validate_ticker, validate_tickers, Ticker
from schemas.finance import FinancialMetrics as FinancialMetrics, Asset as Asset
from schemas.valuation import (
    DCFAssumptions as DCFAssumptions,
    DCFProjectionRow as DCFProjectionRow,
    TerminalValueDetails as TerminalValueDetails,
    DCFScenario as DCFScenario,
    ValuationModel as ValuationModel,
)
from schemas.routing import AgentNode as AgentNode
from schemas.rag import (
    VaultFrontmatter as VaultFrontmatter,
    VaultDocument as VaultDocument,
    VaultArtifact as VaultArtifact,
    CitationRef as CitationRef,
    InsightRecord as InsightRecord,
    MemoryRecord as MemoryRecord,
    ResearchTopic as ResearchTopic,
    ThesisPillar as ThesisPillar,
    PillarOutcome as PillarOutcome,
    PillarType,
    PillarStatus,
    PillarOutcomeStatus,
)
from schemas.maintenance import (
    DuplicateGroup as DuplicateGroup,
    DeduplicationReport as DeduplicationReport,
    DriftEntry as DriftEntry,
)
from schemas.provider import (
    SearchResult as SearchResult,
    NewsResult as NewsResult,
    ScrapeResult as ScrapeResult,
)
