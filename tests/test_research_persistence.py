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
    embedded_texts: list[str] = []

    async def fake_embeddings(texts: list[str]) -> list[list[float]]:
        embedded_texts.extend(texts)
        return [[float(i)] for i, _ in enumerate(texts)]

    async def fake_store(insights: list[InsightRecord]) -> list[str]:
        captured.extend(insights)
        return [f"memory-{i}" for i, _ in enumerate(insights)]

    async def fake_summary(texts: list[str]) -> list[str]:
        return [f"Summary {i}: {text[:12]}" for i, text in enumerate(texts)]

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
            summary_fn=fake_summary,
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
    assert embedded_texts == list(doc.block_map.values())
    assert captured[0].summary.startswith("Summary 0:")
    assert captured[0].summary != embedded_texts[0]
    assert captured[0].metadata["filename"] == artifact.filename
    assert captured[0].metadata["block_id"] == artifact.block_ids[0]
    assert captured[0].metadata["embedding_source"] == "vault_block"
    assert captured[0].metadata["summary_version"] == "rag-summary-v1"
    assert captured[0].metadata["source_chars"] == len(embedded_texts[0])
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


def test_persist_research_artifact_falls_back_when_summary_fails(tmp_path):
    captured: list[InsightRecord] = []

    async def fake_embeddings(texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    async def fake_store(insights: list[InsightRecord]) -> list[str]:
        captured.extend(insights)
        return ["memory-1"]

    async def failing_summary(texts: list[str]) -> list[str]:
        raise RuntimeError("summary unavailable")

    artifact = asyncio.run(
        persist_research_artifact(
            ticker="CRM",
            content="# Header\n\nRevenue grew 12% YoY. Margins expanded.",
            source_type="bull_thesis",
            vault_root=tmp_path,
            embedding_fn=fake_embeddings,
            store_fn=fake_store,
            summary_fn=failing_summary,
        )
    )

    assert artifact.memory_ids == ["memory-1"]
    assert len(captured) == 1
    assert "Revenue grew 12% YoY" in captured[0].summary
    assert "# Header" not in captured[0].summary


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


# ── vectorize=False ──────────────────────────────────────────────────────────

def test_persist_with_vectorize_false_skips_embeddings(tmp_path):
    """vectorize=False writes vault file but skips vector inserts entirely."""
    content = "# Test\n\nBlock content here.\n"
    from utils.vault import VaultReader

    async def fake_embeddings(texts: list[str]) -> list[list[float]]:
        raise AssertionError("Should not be called when vectorize=False")

    async def fake_store(insights: list) -> list[str]:
        raise AssertionError("Should not be called when vectorize=False")

    async def fake_summary(texts: list[str]) -> list[str]:
        raise AssertionError("Should not be called when vectorize=False")

    artifact = asyncio.run(
        persist_research_artifact(
            ticker="AAPL",
            content=content,
            source_type="fundamentals",
            vectorize=False,
            vault_root=tmp_path / "vault",
            embedding_fn=fake_embeddings,
            store_fn=fake_store,
            summary_fn=fake_summary,
        )
    )

    assert artifact.path
    assert artifact.filename
    assert artifact.memory_ids == []
    assert artifact.block_memory_ids == {}
    assert artifact.vector_error == ""

    # Vault file should exist and be readable
    reader = VaultReader(root=tmp_path / "vault")
    doc = reader.read_document(artifact.path)
    assert doc.source_type == "fundamentals"
    assert "Block content" in doc.content


def test_persist_with_vectorize_true_still_creates_vectors(tmp_path):
    """Sanity: vectorize=True (default) still works as before."""
    content = "# Test\n\nImportant research finding.\n"
    captured: list[InsightRecord] = []

    async def fake_embeddings(texts: list[str]) -> list[list[float]]:
        return [[float(i)] for i, _ in enumerate(texts)]

    async def fake_store(insights: list[InsightRecord]) -> list[str]:
        captured.extend(insights)
        return [f"memory-{i}" for i, _ in enumerate(insights)]

    async def fake_summary(texts: list[str]) -> list[str]:
        return [f"Summary {i}" for i, _ in enumerate(texts)]

    artifact = asyncio.run(
        persist_research_artifact(
            ticker="AAPL",
            content=content,
            source_type="test_analysis",
            vault_root=tmp_path / "vault",
            embedding_fn=fake_embeddings,
            store_fn=fake_store,
            summary_fn=fake_summary,
        )
    )
    # With our fake store, we should get back synthetic memory IDs
    assert len(artifact.memory_ids) > 0
    assert len(artifact.block_memory_ids) > 0
