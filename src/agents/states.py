from typing import Optional
from typing_extensions import Annotated, TypedDict
from provider import Asset
from models import ValuationModel
import operator
from datetime import date

class WorkflowState(TypedDict):
    date: str
    run_datetime: str          # datetime.now().isoformat() at run start — used for artifact filename
    company: str
    ticker: str
    price_data: Optional[Asset]
    bull_thesis: str
    bear_thesis: str
    sources: list[str]
    judge_decision: str
    valuation: Optional[ValuationModel]
    final_report: str

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
    sources: list[str]
    messages: Annotated[list[str], operator.add]

class Context:
    user_id: str