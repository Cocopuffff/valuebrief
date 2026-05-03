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
        return {}

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
        "bull_thesis": "Bull thesis",
        "bear_thesis": "Bear thesis",
        "sources": [],
        "judge_decision": "Final decision",
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
    }

    command = asyncio.run(report_generator_mod.report_generator(state))

    assert f"(See: {path.name}#^{block_id})" in command.update["final_report"]
    assert command.update["citation_manifest"][0]["block_id"] == block_id
    assert "Agentforce revenue" in command.update["citation_manifest"][0]["resolved_text"]
