"""
maintenance.py
~~~~~~~~~~~~~~
Schema models for vault deduplication and thesis drift tracking.
"""

from typing import Optional

from pydantic import Field

from schemas import StrictBaseModel, Ticker


class DuplicateGroup(StrictBaseModel):
    """A group of vault files that share the same content hash."""
    content_hash: str
    files: list[str] = Field(default_factory=list)
    kept: Optional[str] = None
    removed: list[str] = Field(default_factory=list)


class DeduplicationReport(StrictBaseModel):
    """Summary of a deduplication run."""
    ticker: Ticker
    total_files: int
    duplicate_groups: int
    files_removed: int
    files_kept: int
    groups: list[DuplicateGroup] = Field(default_factory=list)


class DriftEntry(StrictBaseModel):
    """A single thesis-drift record."""
    date: str
    old_verdict: str
    new_verdict: str
    old_expected_value: Optional[float] = None
    new_expected_value: Optional[float] = None
    delta_pct: Optional[float] = None
    key_changes: list[str] = Field(default_factory=list)
