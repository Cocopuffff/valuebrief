import asyncio
import yaml

from utils.research_persistence import (
    format_artifact_citations,
    persist_research_artifact,
)
from utils.vector_memory import InsightRecord
from utils.vault import VaultReader
from schemas import VaultArtifact


def test_persist_research_artifact_writes_vault_and_vectors(tmp_path):
    captured: list[InsightRecord] = []

    async def fake_embeddings(texts: list[str]) -> list[list[float]]:
        return [[float(i)] for i, _ in enumerate(texts)]

    async def fake_store(insights: list[InsightRecord]) -> list[str]:
        captured.extend(insights)
        return [f"memory-{i}" for i, _ in enumerate(insights)]

    artifact = asyncio.run(
        persist_research_artifact(
            ticker="crm",
            content="Revenue grew 15%.\n\nMargins expanded.",
            source_type="bull_thesis",
            metadata={
                "agent": "bull",
                "source_urls": ["https://example.com/a", "https://example.com/b"],
                "run_datetime": "2026-05-02T15:34:52",
            },
            vault_root=tmp_path,
            embedding_fn=fake_embeddings,
            store_fn=fake_store,
        )
    )

    assert isinstance(artifact, VaultArtifact)
    assert artifact.ticker == "CRM"
    assert artifact.source_type == "bull_thesis"
    assert len(artifact.block_ids) == 2
    assert artifact.memory_ids == ["memory-0", "memory-1"]
    assert set(artifact.block_memory_ids) == set(artifact.block_ids)

    doc = VaultReader(root=tmp_path).read_document(artifact.path)
    assert doc.source_type == "bull_thesis"
    assert len(doc.block_map) == 2

    assert len(captured) == 2
    assert captured[0].ticker == "CRM"
    assert captured[0].metadata["filename"] == artifact.filename
    assert captured[0].metadata["block_id"] == artifact.block_ids[0]
    assert captured[0].metadata["source_urls"] == [
        "https://example.com/a",
        "https://example.com/b",
    ]

    raw_text = open(artifact.path).read()
    fm = yaml.safe_load(raw_text.split("---")[1])
    assert fm["source_urls"] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_persist_research_artifact_keeps_vault_when_vectors_fail(tmp_path):
    async def failing_embeddings(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding unavailable")

    async def noop_store(insights: list[InsightRecord]) -> list[str]:
        raise AssertionError("store should not be called")

    artifact = asyncio.run(
        persist_research_artifact(
            ticker="NFLX",
            content="Key point.",
            source_type="bear_thesis",
            vault_root=tmp_path,
            embedding_fn=failing_embeddings,
            store_fn=noop_store,
        )
    )

    assert isinstance(artifact, VaultArtifact)
    assert artifact.memory_ids == []
    assert "embedding unavailable" in artifact.vector_error

    doc = VaultReader(root=tmp_path).read_document(artifact.path)
    assert doc.source_type == "bear_thesis"
    assert len(doc.block_map) > 0


def test_format_artifact_citations_uses_resolvable_pattern():
    artifact = VaultArtifact(
        ticker="AAPL",
        path="/vault/AAPL/2026-05-02_abcd.md",
        filename="2026-05-02_abcd.md",
        source_type="analyst_report",
        block_ids=["block-11111111", "block-22222222", "block-12345678"],
        block_memory_ids={
            "block-11111111": "mem-a",
            "block-22222222": "mem-b",
        },
        memory_ids=["mem-a", "mem-b"],
    )

    result = format_artifact_citations([artifact], max_blocks_per_artifact=2)

    assert "analyst_report" in result
    assert "(See: 2026-05-02_abcd.md#^block-11111111)" in result
    assert "(See: 2026-05-02_abcd.md#^block-22222222)" in result

    # Third block should be truncated
    assert "block-12345678" not in result
