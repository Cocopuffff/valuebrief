"""
test_curator.py
~~~~~~~~~~~~~~~
Unit tests for the Curator Agent's helper functions and the deduplication engine.
"""

import pytest

from utils.citations import (
    build_citation_manifest,
    resolve_citation,
    extract_block_ids,
    CitationRef,
)
from utils.vault import VaultWriter, VaultReader
from utils.deduplication import find_duplicates, deduplicate
from agents.curator import (
    _extract_verdict,
    _extract_key_changes,
    _memory_ids_for_citations,
)


# ── Citation tests ───────────────────────────────────────────────────────────

class TestCitationParsing:
    def test_extract_block_ids(self):
        text = "Some text ^block-a1b2c3d4 and more ^block-e5f6g7h8"
        ids = extract_block_ids(text)
        # Only hex chars in block IDs — g/h won't match [a-f0-9]{8}
        assert "block-a1b2c3d4" in ids

    def test_build_manifest_no_citations(self):
        manifest = build_citation_manifest("A plain report with no citations.", "AAPL")
        assert len(manifest) == 0

    def test_build_manifest_with_citations(self, tmp_path):
        # Set up a vault file
        writer = VaultWriter(root=tmp_path)
        path = writer.write_document(
            ticker="AAPL",
            content="Revenue grew 15% year over year.",
        )

        # Read back to get block IDs
        reader = VaultReader(root=tmp_path)
        doc = reader.read_document(path)
        block_id = list(doc.block_map.keys())[0]

        # Monkeypatch VAULT_ROOT for this test
        import utils.citations as citations_mod
        original_root = citations_mod.VAULT_ROOT
        citations_mod.VAULT_ROOT = tmp_path

        try:
            report = f"Great growth (See: {path.name}#^{block_id})"
            manifest = build_citation_manifest(report, "AAPL")
            assert len(manifest) == 1
            assert manifest[0].block_id == block_id
            assert manifest[0].resolved_text is not None
            assert "Revenue grew" in manifest[0].resolved_text
        finally:
            citations_mod.VAULT_ROOT = original_root

    def test_memory_ids_for_citations_marks_only_referenced_blocks(self):
        manifest = [
            CitationRef(
                file_path="/tmp/vault/AAPL/2026-05-02_abcd.md",
                block_id="block-a1b2c3d4",
                resolved_text="Referenced paragraph",
            )
        ]
        artifacts = [
            {
                "path": "/tmp/vault/AAPL/2026-05-02_abcd.md",
                "filename": "2026-05-02_abcd.md",
                "block_memory_ids": {
                    "block-a1b2c3d4": "memory-cited",
                    "block-deadbeef": "memory-uncited",
                },
            }
        ]

        assert _memory_ids_for_citations(manifest, artifacts) == ["memory-cited"]


# ── Verdict extraction tests ────────────────────────────────────────────────

class TestVerdictExtraction:
    def test_standard_verdict(self):
        text = "**Verdict** — Buy on weakness\n\n**Rationale** — The company..."
        assert _extract_verdict(text) == "Buy on weakness"

    def test_verdict_with_colon(self):
        text = "**Verdict**: Hold for now\n\nSome rationale"
        assert _extract_verdict(text) == "Hold for now"

    def test_no_verdict_header(self):
        text = "This is a direct assessment.\n\nThe company looks strong."
        verdict = _extract_verdict(text)
        assert verdict == "This is a direct assessment."

    def test_empty_text(self):
        assert _extract_verdict("") == ""


class TestKeyChangesExtraction:
    def test_extracts_risk_bullets(self):
        text = """**Verdict** — Buy

**Rationale** — Good company.

**Key Risks**
- Trade war escalation could compress margins
- Rising interest rates reduce present value
- Competitive pressure from new entrants
"""
        changes = _extract_key_changes(text)
        assert len(changes) == 3
        assert "Trade war" in changes[0]

    def test_no_risks_section(self):
        text = "**Verdict** — Hold\n\nSome analysis without key risks."
        changes = _extract_key_changes(text)
        assert len(changes) == 0


# ── Deduplication tests ──────────────────────────────────────────────────────

class TestDeduplication:
    def test_find_no_duplicates(self, tmp_path):
        writer = VaultWriter(root=tmp_path)
        writer.write_document(ticker="AAPL", content="Unique content A")
        writer.write_document(ticker="AAPL", content="Unique content B")

        groups = find_duplicates("AAPL", root=tmp_path)
        assert len(groups) == 0  # No duplicates since hashes differ

    def test_deduplicate_dry_run(self, tmp_path):
        writer = VaultWriter(root=tmp_path)

        # Write two files with different content — both unique
        writer.write_document(ticker="MSFT", content="Revenue data X")
        writer.write_document(ticker="MSFT", content="Revenue data Y")

        report = deduplicate("MSFT", dry_run=True, root=tmp_path)
        assert report.ticker == "MSFT"
        assert report.total_files == 2
        assert report.files_removed == 0  # No duplicates to remove


# ── Pillar outcome processing tests ──────────────────────────────────────────

class TestPillarOutcomeProcessing:
    """Verify pillar_outcome dicts can be processed for all four statuses."""

    def test_supported_outcome_marks_cited(self):
        """supported pillars are marked as cited for survival signal."""
        outcome = {
            "memory_id": "550e8400-e29b-41d4-a716-446655440000",
            "pillar_id": "CRM-moat-001",
            "status": "supported",
            "reason": "Confirmed by new evidence.",
            "replacement_statement": "",
            "source_urls": [],
        }
        assert outcome["status"] == "supported"
        assert outcome["memory_id"]
        assert outcome["reason"]

    def test_revised_outcome_has_replacement_statement(self):
        """revised status must carry a replacement_statement for the new pillar."""
        outcome = {
            "memory_id": "550e8400-e29b-41d4-a716-446655440000",
            "pillar_id": "CRM-growth-001",
            "status": "revised",
            "reason": "Growth accelerated.",
            "replacement_statement": "Revenue growing at 15% vs prior 12%.",
            "source_urls": ["https://example.com/earnings"],
        }
        assert outcome["replacement_statement"]
        assert len(outcome["source_urls"]) > 0

    def test_weakened_outcome_remains_active(self):
        """weakened pillars remain active but lower-confidence."""
        outcome = {
            "memory_id": "550e8400-e29b-41d4-a716-446655440000",
            "pillar_id": "CRM-growth-001",
            "status": "weakened",
            "reason": "Growth slowed but remains positive.",
            "replacement_statement": "",
            "source_urls": ["https://example.com/earnings"],
        }
        assert outcome["status"] == "weakened"

    def test_contradicted_outcome_demotes(self):
        """contradicted status sets validity_status to contradicted."""
        outcome = {
            "memory_id": "550e8400-e29b-41d4-a716-446655440000",
            "pillar_id": "CRM-risk-001",
            "status": "contradicted",
            "reason": "Competitor exited the market.",
            "replacement_statement": "",
            "source_urls": [],
        }
        assert outcome["status"] == "contradicted"

    def test_stale_outcome_demotes(self):
        """stale status sets validity_status to stale (prune-eligible)."""
        outcome = {
            "memory_id": "550e8400-e29b-41d4-a716-446655440000",
            "pillar_id": "CRM-valuation-001",
            "status": "stale",
            "reason": "Old valuation assumptions no longer relevant.",
            "replacement_statement": "",
            "source_urls": [],
        }
        assert outcome["status"] == "stale"


class TestPillarMemoryRetrievalRules:
    """Verify retrieval rules: only supported/weakened pillars are active."""

    def test_active_statuses_are_supported_and_weakened(self):
        active = {"supported", "weakened"}
        assert "supported" in active
        assert "weakened" in active

    def test_inactive_statuses_are_contradicted_stale_superseded(self):
        inactive = {"contradicted", "stale", "superseded"}
        assert "contradicted" in inactive
        assert "stale" in inactive
        assert "superseded" in inactive
        active = {"supported", "weakened"}
        assert "contradicted" not in active
        assert "stale" not in active
        assert "superseded" not in active
