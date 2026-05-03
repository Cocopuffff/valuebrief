"""
citations.py
~~~~~~~~~~~~
Citation resolver for the Hybrid RAG system.

Maps in-line citations in agent output (e.g. ``(See: 2026-04-27_abc.md#^block-123)``)
back to the local vault paragraph text, and extracts a full "citation manifest" from
a finished report so the Curator knows which sources were actually used.
"""

from __future__ import annotations

import re
from typing import Optional

from schemas import CitationRef
from utils.vault import VaultReader, VAULT_ROOT
from utils.logger import get_logger

logger = get_logger(__name__)

# Pattern:  (See: <filename>#^<block_id>)
# Also matches bare references without the "(See: ...)" wrapper.
_CITATION_RE = re.compile(
    r"\(See:\s*(?P<file>[^\s#]+\.md)#\^(?P<block>[a-zA-Z0-9-]+)\)"
)

# Looser pattern for any ^block-XXXX reference in text
_BLOCK_REF_RE = re.compile(r"\^(block-[a-f0-9]{8})")




def resolve_citation(citation_str: str, ticker: str) -> Optional[str]:
    """Parse a citation string and return the paragraph text from the local vault.

    Args:
        citation_str: A string like ``(See: 2026-04-27_abc.md#^block-123)``
        ticker:       Ticker symbol to locate the vault subdirectory.

    Returns:
        The paragraph text if found, else None.
    """
    match = _CITATION_RE.search(citation_str)
    if not match:
        return None

    filename = match.group("file")
    block_id = match.group("block")

    vault_path = VAULT_ROOT / ticker.upper() / filename
    if not vault_path.exists():
        logger.warning(f"[Citations] File not found: {vault_path}")
        return None

    reader = VaultReader()
    return reader.resolve_block(vault_path, block_id)


def build_citation_manifest(
    report_text: str,
    ticker: str,
) -> list[CitationRef]:
    """Extract all citation references from a report and resolve them.

    Scans the report text for patterns like ``(See: file.md#^block-id)`` and
    returns a list of CitationRef objects with resolved paragraph text.

    Args:
        report_text: The full text of the generated report.
        ticker:      Ticker symbol for vault lookup.

    Returns:
        List of CitationRef objects, each with resolved_text if available.
    """
    refs: list[CitationRef] = []
    seen: set[tuple[str, str]] = set()

    for match in _CITATION_RE.finditer(report_text):
        filename = match.group("file")
        block_id = match.group("block")

        key = (filename, block_id)
        if key in seen:
            continue
        seen.add(key)

        vault_path = VAULT_ROOT / ticker.upper() / filename
        resolved = None
        if vault_path.exists():
            reader = VaultReader()
            resolved = reader.resolve_block(vault_path, block_id)

        refs.append(CitationRef(
            file_path=str(vault_path),
            block_id=block_id,
            resolved_text=resolved,
        ))

    logger.info(f"[Citations] Extracted {len(refs)} citation(s) from report for {ticker}")
    return refs


def extract_block_ids(text: str) -> list[str]:
    """Extract all bare block IDs (^block-XXXX) from text."""
    return _BLOCK_REF_RE.findall(text)
