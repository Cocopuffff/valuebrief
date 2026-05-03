from typing import Optional
from typing_extensions import Annotated, TypedDict
from schemas import Asset, ValuationModel
import operator

class WorkflowState(TypedDict):
    date: str
    run_datetime: str          # datetime.now().isoformat() at run start — used for artifact filename
    company: str
    ticker: str
    price_data: Optional[Asset]
    bull_thesis: str
    bear_thesis: str
    sources: Annotated[list[str], operator.add]
    judge_decision: str
    valuation: Optional[ValuationModel]
    final_report: str
    # Hybrid RAG fields
    citation_manifest: list[dict]      # [{file_path, block_id, resolved_text}]
    curator_log: str                   # Maintenance actions summary from Curator
    active_memory_ids: Annotated[list[str], operator.add]  # UUIDs created in this run
    vault_artifacts: Annotated[list[dict], operator.add]   # Persisted vault metadata

class ResearchState(TypedDict):
    date: str
    company: str
    ticker: str
    price_data: Optional[Asset]
    existing_valuation: Optional[ValuationModel]  # Prior valuation from Supabase, if any
    max_iterations: int
    iteration_count: Annotated[int, operator.add]
    research_topics: list[str]
    key_points: list[str]
    thesis: str
    sources: Annotated[list[str], operator.add]
    messages: Annotated[list[str], operator.add]

class Context:
    user_id: str
