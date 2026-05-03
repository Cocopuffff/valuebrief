"""
research_persistence.py
~~~~~~~~~~~~~~~~~~~~~~~
Shared Hybrid RAG persistence helpers for per-run research artifacts.

The local vault is the durable source of truth. Vector insertion is best-effort:
if embeddings or Supabase writes fail, callers still receive the vault artifact
metadata so reports can cite the local Markdown blocks.
"""

from __future__ import annotations

import pathlib
from collections.abc import Awaitable, Callable
from typing import Any, Optional, Union

from utils.embeddings import get_embeddings
from utils.logger import get_logger
from utils.vault import VAULT_ROOT, VaultReader, VaultWriter
from utils.vector_memory import InsightRecord, batch_store_insights
from schemas import VaultArtifact

logger = get_logger(__name__)

EmbeddingFn = Callable[[list[str]], Awaitable[list[list[float]]]]
StoreFn = Callable[[list[InsightRecord]], Awaitable[list[str]]]

# Accept either a VaultArtifact instance or a serialised dict (from LangGraph state).
_ArtifactLike = Union[VaultArtifact, dict[str, Any]]


async def persist_research_artifact(
    *,
    ticker: str,
    content: str,
    source_type: str,
    metadata: Optional[dict[str, Any]] = None,
    source_priority: int = 1,
    vault_root: Optional[pathlib.Path] = None,
    embedding_fn: EmbeddingFn = get_embeddings,
    store_fn: StoreFn = batch_store_insights,
) -> VaultArtifact:
    """Persist one research artifact to the vault and vector memory.

    Returns a VaultArtifact Pydantic model.  Callers should serialise with
    ``artifact.model_dump(mode="json")`` before storing in LangGraph state.
    """
    clean_content = str(content or "").strip()
    if not clean_content:
        return VaultArtifact(
            ticker=ticker.upper(),
            path="",
            filename="",
            source_type=source_type,
        )

    ticker = ticker.upper()
    metadata = dict(metadata or {})
    metadata["source_type"] = source_type

    root = vault_root or VAULT_ROOT
    writer = VaultWriter(root=root)
    reader = VaultReader(root=root)

    path = writer.write_document(
        ticker=ticker,
        content=clean_content,
        metadata=metadata,
    )
    doc = reader.read_document(path)

    block_ids = list(doc.block_map.keys())
    memory_ids: list[str] = []
    block_memory_ids: dict[str, str] = {}
    vector_error = ""

    if doc.block_map:
        try:
            block_items = list(doc.block_map.items())
            embeddings = await embedding_fn([text for _, text in block_items])

            insights: list[InsightRecord] = []
            for (block_id, text), embedding in zip(block_items, embeddings):
                insight_metadata = {
                    **metadata,
                    "type": "vault_artifact",
                    "ticker": ticker,
                    "source_type": doc.source_type,
                    "local_path": str(path),
                    "filename": pathlib.Path(path).name,
                    "block_id": block_id,
                    "citation": f"{pathlib.Path(path).name}#^{block_id}",
                    "content_hash": doc.content_hash,
                }
                insights.append(
                    InsightRecord(
                        ticker=ticker,
                        summary=text,
                        embedding=embedding,
                        metadata=insight_metadata,
                        source_priority=source_priority,
                        is_cited=False,
                    )
                )

            memory_ids = await store_fn(insights)
            block_memory_ids = {
                block_id: memory_id
                for (block_id, _), memory_id in zip(block_items, memory_ids)
            }
        except Exception as e:
            vector_error = str(e)
            logger.warning(
                "[Persistence] Vector write failed for %s/%s; vault artifact kept: %s",
                ticker,
                doc.source_type,
                e,
            )

    return VaultArtifact(
        ticker=ticker,
        path=str(path),
        filename=pathlib.Path(path).name,
        source_type=doc.source_type,
        block_ids=block_ids,
        block_memory_ids=block_memory_ids,
        memory_ids=memory_ids,
        vector_error=vector_error,
    )


def _is_model(obj: _ArtifactLike) -> bool:
    """Return True if obj is a VaultArtifact Pydantic model instance."""
    return hasattr(obj, "memory_ids") and not isinstance(obj, dict)


def memory_ids_from_artifact(artifact: _ArtifactLike) -> list[str]:
    """Return vector memory IDs from a VaultArtifact or serialised artifact dict."""
    if _is_model(artifact):
        return [str(mid) for mid in artifact.memory_ids if mid]
    return [str(mid) for mid in artifact.get("memory_ids", []) if mid]


def format_artifact_citations(
    artifacts: list[_ArtifactLike],
    *,
    max_blocks_per_artifact: int = 2,
) -> str:
    """Build resolvable report citation lines from vault artifacts.

    Accepts a list of VaultArtifact instances or serialised dicts (from LangGraph state).
    """
    lines: list[str] = []
    for artifact in artifacts:
        if _is_model(artifact):
            filename = artifact.filename
            block_ids = list(artifact.block_ids or [])
            if not block_ids:
                block_ids = list((artifact.block_memory_ids or {}).keys())
            source_type = artifact.source_type
        else:
            filename = artifact.get("filename")
            if not filename and artifact.get("path"):
                filename = pathlib.Path(str(artifact["path"])).name
            if not filename:
                continue
            block_ids = list(artifact.get("block_ids") or [])
            if not block_ids:
                block_ids = list((artifact.get("block_memory_ids") or {}).keys())
            source_type = artifact.get("source_type", "analysis")

        if not filename or not block_ids:
            continue

        for block_id in block_ids[:max_blocks_per_artifact]:
            lines.append(f"- {source_type}: (See: {filename}#^{block_id})")

    return "\n".join(lines)
