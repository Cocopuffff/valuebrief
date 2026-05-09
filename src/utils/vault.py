"""
vault.py
~~~~~~~~
Local Vault file handler for Hybrid RAG "Cold Storage".

Directory structure:  data/vault/{ticker}/{YYYY-MM-DD}_{source_hash}.md

Each file has:
- YAML frontmatter (url, source_type, date_scraped, sentiment, block_count, content_hash)
- Markdown body with unique block IDs appended to each paragraph (^block-XXXX)
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import uuid
from datetime import date
from typing import Optional

import yaml

from schemas import VaultDocument
from utils.logger import get_logger

logger = get_logger(__name__)
MAX_BLOCK_CHARS = 2500
_BLOCK_ID_PATTERN = re.compile(
    r"(?:^|[ \t])\^(block-[a-f0-9]{8})(?=\s*(?:\n\n|$))",
    re.MULTILINE,
)
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+\S")
_RULE_PATTERN = re.compile(r"\s*[-*_]{3,}\s*")

VAULT_ROOT = pathlib.Path(
    os.getenv("VAULT_PATH",
              pathlib.Path(__file__).parent.parent.parent / "data" / "vault")
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _content_hash(text: str) -> str:
    """SHA-256 of whitespace-normalised content for deduplication."""
    normalised = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def _short_uuid() -> str:
    return uuid.uuid4().hex[:8]


def _is_heading(line: str) -> bool:
    return bool(_HEADING_PATTERN.match(line.strip()))


def _heading_level(line: str) -> int:
    match = _HEADING_PATTERN.match(line.strip())
    return len(match.group(1)) if match else 0


def _split_markdown_blocks(content: str) -> list[str]:
    """Split Markdown into logical blocks while preserving fenced code blocks."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    fence_marker = ""

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            current.append(line)
            continue

        if not in_fence and not stripped:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue

        current.append(line)

    if current:
        blocks.append("\n".join(current).strip())

    return [block for block in blocks if block]


def _apply_heading_context(content: str, headings: list[str]) -> str:
    if not headings:
        return content.strip()
    return "\n".join(headings).strip() + "\n\n" + content.strip()


def _cap_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].rstrip()
    return clipped


def _split_oversized_chunk(content: str, headings: list[str]) -> list[str]:
    """Split a chunk on paragraph boundaries, hard-capping each emitted block."""
    with_context = _apply_heading_context(content, headings)
    if len(with_context) <= MAX_BLOCK_CHARS:
        return [with_context]

    max_body_chars = MAX_BLOCK_CHARS
    prefix = ""
    if headings:
        prefix = "\n".join(headings).strip() + "\n\n"
        max_body_chars = max(200, MAX_BLOCK_CHARS - len(prefix))

    parts = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    chunks: list[str] = []
    current = ""

    def emit(body: str) -> None:
        if not body.strip():
            return
        if len(body) <= max_body_chars:
            chunks.append((prefix + body.strip()).strip())
            return
        remaining = body.strip()
        while remaining:
            piece = _cap_text(remaining, max_body_chars)
            if not piece:
                piece = remaining[:max_body_chars]
            chunks.append((prefix + piece).strip())
            remaining = remaining[len(piece):].strip()

    for part in parts:
        candidate = f"{current}\n\n{part}".strip() if current else part
        if len(candidate) <= max_body_chars:
            current = candidate
            continue
        emit(current)
        current = ""
        emit(part)

    emit(current)
    return chunks


def _tag_blocks(content: str) -> tuple[str, dict[str, str]]:
    """Split Markdown into source chunks and append a unique block ID to each.

    Returns the tagged content string and a {block_id: paragraph_text} map.
    """
    markdown_blocks = _split_markdown_blocks(content)
    tagged_parts: list[str] = []
    block_map: dict[str, str] = {}
    heading_stack: list[tuple[int, str]] = []

    for block in markdown_blocks:
        lines = block.splitlines()
        content_lines: list[str] = []

        for line in lines:
            if not content_lines and _is_heading(line):
                level = _heading_level(line)
                heading_stack = [
                    (existing_level, text)
                    for existing_level, text in heading_stack
                    if existing_level < level
                ]
                heading_stack.append((level, line.strip()))
                continue
            content_lines.append(line)

        content_block = "\n".join(content_lines).strip()
        if not content_block:
            continue
        # Skip blocks whose only "content" is horizontal rules (---, ***, ___).
        substantive_lines = [
            line for line in content_block.splitlines()
            if line.strip() and not _RULE_PATTERN.fullmatch(line.strip())
        ]
        if not substantive_lines:
            continue

        headings = [text for _, text in heading_stack]
        for chunk in _split_oversized_chunk(content_block, headings):
            block_id = f"block-{_short_uuid()}"
            tagged_parts.append(f"{chunk} ^{block_id}")
            block_map[block_id] = chunk

    return "\n\n".join(tagged_parts), block_map


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a Markdown file into YAML frontmatter dict and body content."""
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    try:
        fm = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        fm = {}

    body = parts[2].strip()
    return fm, body


def _extract_block_map(body: str) -> dict[str, str]:
    """Parse block IDs from tagged content.

    Block IDs look like: ... ^block-abcd1234
    """
    block_map: dict[str, str] = {}
    start = 0

    for match in _BLOCK_ID_PATTERN.finditer(body):
        block_id = match.group(1)
        clean = body[start:match.start()].strip()
        if clean:
            block_map[block_id] = clean
        start = match.end()

    return block_map


# ── VaultWriter ──────────────────────────────────────────────────────────────

class VaultWriter:
    """Writes research documents to the local vault."""

    def __init__(self, root: Optional[pathlib.Path] = None) -> None:
        self.root = root or VAULT_ROOT

    def write_document(
        self,
        ticker: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> pathlib.Path:
        """Write a Markdown document to the vault.

        Args:
            ticker:   Stock ticker (e.g. 'AAPL').
            content:  Raw Markdown content to store.
            metadata: Dict with optional keys: url, source_type, sentiment.

        Returns:
            The Path of the created file.
        """
        metadata = metadata or {}
        ticker = ticker.upper()
        today = date.today().isoformat()
        c_hash = _content_hash(content)

        # Build directory
        ticker_dir = self.root / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        # Tag paragraphs with block IDs
        tagged_content, block_map = _tag_blocks(content)

        # Build YAML frontmatter
        frontmatter = {
            "url": metadata.get("url", ""),
            "source_type": metadata.get("source_type", "analysis"),
            "date_scraped": today,
            "sentiment": metadata.get("sentiment"),
            "block_count": len(block_map),
            "content_hash": c_hash,
            "archived": False,
        }
        frontmatter.update(
            {
                key: value
                for key, value in metadata.items()
                if key not in frontmatter and value is not None
            }
        )

        # Compose final file content
        fm_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        file_content = f"---\n{fm_str}---\n\n{tagged_content}\n"

        # Filename: YYYY-MM-DD_<hash>.md
        filename = f"{today}_{c_hash}.md"
        file_path = ticker_dir / filename

        # Skip if identical content already exists
        if file_path.exists():
            logger.debug(f"[Vault] Skipping duplicate: {file_path}")
            return file_path

        file_path.write_text(file_content, encoding="utf-8")
        logger.info(f"[Vault] 📁 Written: {file_path} ({len(block_map)} blocks)")
        return file_path

    def write_synthesis(
        self,
        ticker: str,
        month: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> pathlib.Path:
        """Write or overwrite a monthly synthesis file.

        Unlike ``write_document``, this uses a predictable filename
        (``{month}_synthesis.md``) so it can be found and updated
        incrementally across daily runs.

        Args:
            ticker:   Stock ticker.
            month:    Month string, e.g. ``"2026-01"``.
            content:  The LLM-synthesised Markdown content.
            metadata: Optional extra frontmatter fields.

        Returns:
            The Path of the created/overwritten file.
        """
        metadata = metadata or {}
        ticker = ticker.upper()
        c_hash = _content_hash(content)

        ticker_dir = self.root / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        tagged_content, block_map = _tag_blocks(content)

        frontmatter = {
            "url": metadata.get("url", ""),
            "source_type": "monthly_synthesis",
            "date_scraped": date.today().isoformat(),
            "sentiment": metadata.get("sentiment"),
            "block_count": len(block_map),
            "content_hash": c_hash,
            "month": month,
            "archived": False,
        }
        frontmatter.update(
            {
                key: value
                for key, value in metadata.items()
                if key not in frontmatter and value is not None
            }
        )

        fm_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        file_content = f"---\n{fm_str}---\n\n{tagged_content}\n"

        file_path = ticker_dir / f"{month}_synthesis.md"
        file_path.write_text(file_content, encoding="utf-8")
        logger.info(f"[Vault] 📁 Synthesis written: {file_path} ({len(block_map)} blocks)")
        return file_path

    def write_pillar_dossier(
        self,
        *,
        ticker: str,
        pillar_id: str,
        pillar_type: str,
        status: str,
        version: int,
        statement: str,
        rationale: str = "",
        valuation_impact: str = "",
        source_urls: Optional[list[str]] = None,
        evidence_citations: Optional[list[str]] = None,
        current_memory_id: str = "",
        statement_hash: str = "",
        lifecycle_event: str = "supported",
        lifecycle_reason: str = "",
        resurrection_reason: str = "",
        merged_into_pillar_id: str = "",
    ) -> pathlib.Path:
        """Create or update the full local audit dossier for a thesis pillar."""
        ticker = ticker.upper()
        source_urls = source_urls or []
        evidence_citations = evidence_citations or []

        ticker_dir = self.root / ticker / "pillars"
        ticker_dir.mkdir(parents=True, exist_ok=True)
        file_path = ticker_dir / f"{pillar_id}.md"

        existing_history = ""
        if file_path.exists():
            try:
                _, old_body = _parse_frontmatter(file_path.read_text(encoding="utf-8"))
                marker = "## Lifecycle History"
                if marker in old_body:
                    existing_history = old_body.split(marker, 1)[1].strip()
            except Exception as e:
                logger.debug("[Vault] Failed to read existing pillar dossier %s: %s", file_path, e)

        history_line = (
            f"- {date.today().isoformat()} | {lifecycle_event}: "
            f"{lifecycle_reason or 'No additional rationale provided.'}"
        )
        if resurrection_reason:
            history_line += f" Resurrection rationale: {resurrection_reason}"
        if merged_into_pillar_id:
            history_line += f" Merged into: {merged_into_pillar_id}"

        history_lines = [history_line]
        if existing_history:
            history_lines.append(existing_history)

        def list_block(items: list[str]) -> str:
            if not items:
                return "- None recorded."
            return "\n".join(f"- {item}" for item in items)

        body = (
            f"# {pillar_id}\n\n"
            f"## Current Statement\n\n{statement or 'None recorded.'}\n\n"
            f"## Rationale\n\n{rationale or 'None recorded.'}\n\n"
            f"## Valuation Impact\n\n{valuation_impact or 'None recorded.'}\n\n"
            f"## Source URLs\n\n{list_block(source_urls)}\n\n"
            f"## Vault Citations\n\n{list_block(evidence_citations)}\n\n"
            f"## Supersession Chain\n\n"
            f"- Current memory ID: {current_memory_id or 'pending'}\n"
            f"- Version: {version}\n"
            f"- Merged into pillar ID: {merged_into_pillar_id or 'none'}\n\n"
            f"## Lifecycle History\n\n"
            + "\n".join(history_lines)
            + "\n"
        )

        frontmatter = {
            "source_type": "thesis_pillar_dossier",
            "ticker": ticker,
            "pillar_id": pillar_id,
            "pillar_type": pillar_type,
            "status": status,
            "version": version,
            "current_memory_id": current_memory_id,
            "statement_hash": statement_hash,
            "date_scraped": date.today().isoformat(),
            "archived": False,
        }
        fm_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        file_path.write_text(f"---\n{fm_str}---\n\n{body}", encoding="utf-8")
        logger.info("[Vault] 🏛️ Pillar dossier written: %s", file_path)
        return file_path


# ── VaultReader ──────────────────────────────────────────────────────────────

class VaultReader:
    """Reads and queries documents from the local vault."""

    def __init__(self, root: Optional[pathlib.Path] = None) -> None:
        self.root = root or VAULT_ROOT

    def read_document(self, path: str | pathlib.Path) -> VaultDocument:
        """Parse a vault Markdown file into a VaultDocument."""
        path = pathlib.Path(path)
        text = path.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        block_map = _extract_block_map(body)

        # Derive ticker from parent directory name
        ticker = path.parent.name

        return VaultDocument(
            path=str(path),
            ticker=ticker.upper(),
            url=fm.get("url"),
            source_type=fm.get("source_type", "analysis"),
            date_scraped=fm.get("date_scraped", ""),
            sentiment=fm.get("sentiment"),
            content=body,
            block_map=block_map,
            content_hash=fm.get("content_hash", ""),
            archived=fm.get("archived", False),
        )

    def list_documents(
        self,
        ticker: str,
        since: Optional[str] = None,
    ) -> list[VaultDocument]:
        """List all vault documents for a ticker, optionally filtered by date.

        Args:
            ticker: Stock ticker symbol.
            since:  ISO date string (YYYY-MM-DD). Only return files on or after this date.
        """
        ticker = ticker.upper()
        ticker_dir = self.root / ticker
        if not ticker_dir.exists():
            return []

        docs: list[VaultDocument] = []
        for md_file in sorted(ticker_dir.glob("*.md")):
            if since:
                # Filename starts with YYYY-MM-DD
                file_date = md_file.stem[:10]
                if file_date < since:
                    continue
            try:
                docs.append(self.read_document(md_file))
            except Exception as e:
                logger.warning(f"[Vault] Failed to parse {md_file}: {e}")

        return docs

    def resolve_block(self, path: str | pathlib.Path, block_id: str) -> Optional[str]:
        """Return the paragraph text for a given block ID in a vault file."""
        doc = self.read_document(path)
        return doc.block_map.get(block_id)

    def get_block_map(self, path: str | pathlib.Path) -> dict[str, str]:
        """Return the full {block_id: paragraph_text} map for a vault file."""
        doc = self.read_document(path)
        return doc.block_map

    def mark_archived(self, path: str | pathlib.Path) -> None:
        """Update the YAML frontmatter to set archived=true."""
        path = pathlib.Path(path)
        text = path.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        fm["archived"] = True

        fm_str = yaml.dump(fm, default_flow_style=False, sort_keys=False)
        path.write_text(f"---\n{fm_str}---\n\n{body}\n", encoding="utf-8")
        logger.info(f"[Vault] 📦 Archived: {path}")
