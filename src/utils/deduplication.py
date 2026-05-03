"""
deduplication.py
~~~~~~~~~~~~~~~~
Local deduplication engine for the vault.

Scans ``data/vault/{ticker}/`` and identifies files with identical or
near-identical content (SHA-256 hash match).  The Curator calls this
after each run to prevent redundant vectorisation.
"""

from __future__ import annotations

import os
import pathlib
from collections import defaultdict
from typing import Optional

from schemas import DuplicateGroup, DeduplicationReport
from utils.vault import VaultReader, VaultDocument, VAULT_ROOT
from utils.logger import get_logger

logger = get_logger(__name__)


def find_duplicates(
    ticker: str,
    root: Optional[pathlib.Path] = None,
) -> list[DuplicateGroup]:
    """Scan the vault for a ticker and group files by content hash.

    Only returns groups with 2+ files (i.e., actual duplicates).

    Args:
        ticker: Stock ticker symbol.
        root:   Override vault root path.

    Returns:
        List of DuplicateGroup objects.
    """
    vault_root = root or VAULT_ROOT
    reader = VaultReader(root=vault_root)
    docs: list[VaultDocument] = reader.list_documents(ticker)

    # Group by content_hash
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for doc in docs:
        if doc.content_hash:
            hash_groups[doc.content_hash].append(doc.path)

    # Filter to groups with duplicates
    groups: list[DuplicateGroup] = []
    for c_hash, paths in hash_groups.items():
        if len(paths) >= 2:
            groups.append(DuplicateGroup(content_hash=c_hash, files=sorted(paths)))

    return groups


def deduplicate(
    ticker: str,
    dry_run: bool = True,
    root: Optional[pathlib.Path] = None,
) -> DeduplicationReport:
    """Remove duplicate vault files for a ticker.

    Keeps the **earliest** file in each duplicate group (by filename, which
    starts with the date).  Removes the rest.

    Args:
        ticker:  Stock ticker symbol.
        dry_run: If True, report what would be removed but don't delete.
        root:    Override vault root path.

    Returns:
        A DeduplicationReport summarising actions.
    """
    vault_root = root or VAULT_ROOT
    ticker = ticker.upper()
    reader = VaultReader(root=vault_root)
    all_docs = reader.list_documents(ticker)
    groups = find_duplicates(ticker, root=vault_root)

    total_removed = 0
    processed_groups: list[DuplicateGroup] = []

    for group in groups:
        # Keep the earliest file (first when sorted alphabetically by path,
        # since filenames start with YYYY-MM-DD)
        sorted_files = sorted(group.files)
        kept = sorted_files[0]
        to_remove = sorted_files[1:]

        if not dry_run:
            for path in to_remove:
                try:
                    os.remove(path)
                    logger.info(f"[Dedup] 🗑️ Removed duplicate: {path}")
                except OSError as e:
                    logger.warning(f"[Dedup] Failed to remove {path}: {e}")

        total_removed += len(to_remove)
        processed_groups.append(DuplicateGroup(
            content_hash=group.content_hash,
            files=group.files,
            kept=kept,
            removed=to_remove,
        ))

    report = DeduplicationReport(
        ticker=ticker,
        total_files=len(all_docs),
        duplicate_groups=len(processed_groups),
        files_removed=total_removed,
        files_kept=len(all_docs) - total_removed,
        groups=processed_groups,
    )

    mode = "DRY RUN" if dry_run else "EXECUTED"
    logger.info(
        f"[Dedup] {mode} for {ticker}: "
        f"{report.total_files} files, {report.duplicate_groups} duplicate groups, "
        f"{report.files_removed} removed, {report.files_kept} kept"
    )
    return report
