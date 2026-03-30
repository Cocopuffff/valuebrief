from typing import Optional
from typing_extensions import Annotated, TypedDict
from provider import Asset
from pydantic import HttpUrl
import operator
from datetime import date

class WorkflowState(TypedDict):
    date: str
    company: str
    ticker: str
    price_data: Optional[Asset]
    bull_thesis: str
    bear_thesis: str
    bull_sources: list[HttpUrl]
    bear_sources: list[HttpUrl]
    judge_decision: str
    final_report: str

class ResearchState(TypedDict):
    date: str
    company: str
    ticker: str
    price_data: Optional[Asset]
    max_iterations: int
    iteration_count: Annotated[int, operator.add]
    research_topics: list[str]
    key_points: list[str]
    thesis: str
    sources: list[HttpUrl]
    messages: Annotated[list[str], operator.add]