import asyncio

import utils.citations as citations_mod
from agents import report_generator as report_generator_mod
from utils.vault import VaultReader, VaultWriter


def test_report_generator_injects_resolvable_vault_citations(monkeypatch, tmp_path):
    path = VaultWriter(root=tmp_path).write_document(
        ticker="CRM",
        content="Agentforce revenue is growing quickly.",
        metadata={"source_type": "bull_thesis"},
    )
    doc = VaultReader(root=tmp_path).read_document(path)
    block_id = next(iter(doc.block_map))

    async def fake_persist_research_artifact(**kwargs):
        from schemas.rag import VaultArtifact
        return VaultArtifact(
            ticker="CRM",
            path=str(path),
            filename=path.name,
            source_type="bull_thesis",
            block_ids=[block_id],
            block_memory_ids={block_id: "memory-1"},
            memory_ids=["memory-1"],
        )

    monkeypatch.setattr(citations_mod, "VAULT_ROOT", tmp_path)
    monkeypatch.setattr(
        report_generator_mod,
        "persist_research_artifact",
        fake_persist_research_artifact,
    )

    state = {
        "date": "2026-05-02",
        "run_datetime": "",
        "company": "Salesforce.com Inc",
        "ticker": "CRM",
        "price_data": None,
        "bull_thesis": "Bull thesis\nAgentforce revenue is growing quickly.",
        "bear_thesis": "Bear thesis",
        "sources": [],
        "judge_decision": "Final decision with Agentforce revenue is growing quickly.",
        "valuation": None,
        "final_report": "",
        "citation_manifest": [],
        "curator_log": "",
        "active_memory_ids": ["memory-1"],
        "vault_artifacts": [
            {
                "path": str(path),
                "filename": path.name,
                "source_type": "bull_thesis",
                "block_ids": [block_id],
                "block_memory_ids": {block_id: "memory-1"},
            }
        ],
        "rag_context": "",
        "retrieved_memory_ids": [],
        "research_topics": [],
        "thesis_pillars": [],
        "pillar_outcomes": [],
    }

    command = asyncio.run(report_generator_mod.report_generator(state))

    # No VAULT CITATIONS section (removed in pillar-based report)
    assert "VAULT CITATIONS" not in command.update["final_report"]
    # THESIS PILLARS section only appears when pillars are present
    # (empty list = no section)
    assert "Final decision" in command.update["final_report"]

    # Citation manifest key is present in the update
    assert "citation_manifest" in command.update


def test_report_generator_includes_thesis_pillars_section():
    """When thesis_pillars are present, they appear in the report."""
    state = {
        "date": "2026-05-02",
        "run_datetime": "",
        "company": "Salesforce.com Inc",
        "ticker": "CRM",
        "price_data": None,
        "bull_thesis": "Bull thesis",
        "bear_thesis": "Bear thesis",
        "sources": ["https://example.com/report"],
        "judge_decision": "Buy on weakness.",
        "valuation": None,
        "final_report": "",
        "citation_manifest": [],
        "curator_log": "",
        "active_memory_ids": [],
        "vault_artifacts": [],
        "rag_context": "",
        "retrieved_memory_ids": [],
        "research_topics": [],
        "thesis_pillars": [
            {
                "pillar_id": "CRM-moat-001",
                "pillar_type": "moat",
                "statement": "Salesforce has a wide economic moat.",
                "rationale": "High switching costs and ecosystem lock-in.",
                "valuation_impact": "Supports premium valuation multiple.",
                "source_urls": ["https://example.com/moat"],
                "evidence_citations": ["file.md#^block123"],
                "status": "supported",
            },
        ],
        "pillar_outcomes": [],
    }

    command = asyncio.run(report_generator_mod.report_generator(state))

    report = command.update["final_report"]
    assert "THESIS PILLARS" in report
    assert "Salesforce has a wide economic moat" in report
    assert "High switching costs" in report
    assert "https://example.com/moat" in report
    assert "file.md#^block123" in report
    # Sources section has the website source
    assert "https://example.com/report" in report
