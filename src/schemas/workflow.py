"""
workflow.py
~~~~~~~~~~~
Pydantic validation helpers for LangGraph state boundaries.
"""

from schemas import StrictBaseModel, Ticker


class WorkflowStateModel(StrictBaseModel):
    """Validation model for WorkflowState at graph boundaries."""
    date: str
    run_datetime: str
    company: str
    ticker: Ticker
    bull_thesis: str = ""
    bear_thesis: str = ""
    sources: list[str] = []
    judge_decision: str = ""
    final_report: str = ""
    citation_manifest: list[dict] = []
    curator_log: str = ""
    active_memory_ids: list[str] = []
    vault_artifacts: list[dict] = []


class ResearchStateModel(StrictBaseModel):
    """Validation model for ResearchState at graph boundaries."""
    date: str
    company: str
    ticker: Ticker
    max_iterations: int = 3
    iteration_count: int = 0
    research_topics: list[str] = []
    key_points: list[str] = []
    thesis: str = ""
    sources: list[str] = []
    messages: list[str] = []
