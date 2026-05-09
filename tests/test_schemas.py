"""
test_schemas.py
~~~~~~~~~~~~~~~
Pydantic schema validation tests for the schemas package.
"""

import pytest
from datetime import datetime

from schemas import *


# ── Finance ──────────────────────────────────────────────────────────────

class TestFinancialMetrics:
    def test_valid(self):
        fm = FinancialMetrics(pe_ratio=25.5, market_cap=1e9)
        assert fm.pe_ratio == 25.5

    def test_defaults(self):
        fm = FinancialMetrics()
        assert fm.pe_ratio is None
        assert fm.market_cap is None

    def test_rejects_extra(self):
        with pytest.raises(ValueError):
            FinancialMetrics(pe_ratio=10, unknown_field=42)


class TestAsset:
    def test_valid(self):
        a = Asset(ticker="AAPL", current_price=150.0)
        assert a.ticker == "AAPL"
        assert isinstance(a.fundamentals, FinancialMetrics)

    def test_rejects_missing_required(self):
        with pytest.raises(ValueError):
            Asset(ticker="AAPL")


# ── Valuation ────────────────────────────────────────────────────────────

class TestDCFAssumptions:
    def test_valid(self):
        a = DCFAssumptions(
            revenue_growth_stage_1=0.15,
            revenue_growth_stage_2=0.08,
            ebit_margin_target=0.25,
        )
        assert a.wacc == 0.10


class TestDCFScenario:
    def test_valid(self):
        s = DCFScenario(
            label="Base",
            probability=0.5,
            assumptions=DCFAssumptions(
                revenue_growth_stage_1=0.12,
                revenue_growth_stage_2=0.06,
                ebit_margin_target=0.20,
            ),
        )
        assert s.label == "Base"

    def test_invalid_label(self):
        with pytest.raises(ValueError):
            DCFScenario(
                label="Neutral",
                probability=0.5,
                assumptions=DCFAssumptions(
                    revenue_growth_stage_1=0.12,
                    revenue_growth_stage_2=0.06,
                    ebit_margin_target=0.20,
                ),
            )


class TestValuationModel:
    def test_compute_dcf(self):
        model = ValuationModel(
            ticker="AAPL",
            company="Apple Inc.",
            current_price=150.0,
            base_revenue=383_000_000_000,
            shares_outstanding=15_500_000_000,
            scenarios={
                "Base": DCFScenario(
                    label="Base",
                    probability=0.5,
                    assumptions=DCFAssumptions(
                        revenue_growth_stage_1=0.05,
                        revenue_growth_stage_2=0.03,
                        ebit_margin_target=0.30,
                    ),
                ),
                "Bull": DCFScenario(
                    label="Bull",
                    probability=0.3,
                    assumptions=DCFAssumptions(
                        revenue_growth_stage_1=0.08,
                        revenue_growth_stage_2=0.05,
                        ebit_margin_target=0.35,
                    ),
                ),
                "Bear": DCFScenario(
                    label="Bear",
                    probability=0.2,
                    assumptions=DCFAssumptions(
                        revenue_growth_stage_1=0.02,
                        revenue_growth_stage_2=0.01,
                        ebit_margin_target=0.25,
                    ),
                ),
            },
        )
        model.compute_dcf(2026)

        assert model.expected_value > 0
        assert model.dispersion_ratio >= 0
        for s in model.scenarios.values():
            assert s.intrinsic_value is not None
            assert s.intrinsic_value > 0
            assert s.dcf_table is not None
            assert len(s.dcf_table) == 10
            assert s.terminal_value_details is not None
            assert s.terminal_value_details.terminal_value > 0


# ── Routing ──────────────────────────────────────────────────────────────

class TestAgentNode:
    def test_values(self):
        assert AgentNode.SUPERVISOR == "supervisor"
        assert AgentNode.BULL == "bull_analyst"
        assert AgentNode.CURATOR == "curator_agent"


# ── RAG ──────────────────────────────────────────────────────────────────

class TestVaultDocument:
    def test_valid(self):
        doc = VaultDocument(path="/vault/AAPL/2026-01-01_abc.md", ticker="AAPL")
        assert doc.ticker == "AAPL"
        assert doc.source_type == "analysis"


class TestVaultArtifact:
    def test_valid(self):
        a = VaultArtifact(
            ticker="AAPL",
            path="/vault/AAPL/2026-01-01_abc.md",
            filename="2026-01-01_abc.md",
            source_type="bull_thesis",
            block_ids=["block-11111111"],
        )
        assert a.memory_ids == []
        assert a.vector_error == ""

    def test_serializable(self):
        a = VaultArtifact(
            ticker="AAPL",
            path="/vault/AAPL/test.md",
            filename="test.md",
            source_type="news",
            block_ids=["block-11111111"],
            block_memory_ids={"block-11111111": "mem-1"},
            memory_ids=["mem-1"],
        )
        d = a.model_dump(mode="json")
        assert d["ticker"] == "AAPL"
        assert d["block_ids"] == ["block-11111111"]
        assert isinstance(d, dict)


class TestCitationRef:
    def test_valid(self):
        ref = CitationRef(
            file_path="/vault/AAPL/test.md",
            block_id="block-a1b2c3d4",
            resolved_text="some text",
        )
        assert ref.resolved_text == "some text"


class TestInsightRecord:
    def test_valid(self):
        ir = InsightRecord(
            ticker="AAPL",
            summary="Strong revenue growth",
            embedding=[0.1, 0.2, 0.3],
        )
        assert ir.source_priority == 0
        assert ir.is_cited is False


class TestMemoryRecord:
    def test_valid(self):
        now = datetime.now()
        mr = MemoryRecord(
            id="uuid-1",
            summary="Growth insight",
            metadata={},
            ticker="AAPL",
            source_priority=1,
            is_cited=False,
            created_at=now,
            updated_at=now,
        )
        assert mr.similarity is None


# ── Maintenance ──────────────────────────────────────────────────────────

class TestDuplicateGroup:
    def test_valid(self):
        g = DuplicateGroup(
            content_hash="abc123",
            files=["/vault/AAPL/a.md", "/vault/AAPL/b.md"],
        )
        assert g.kept is None


class TestDeduplicationReport:
    def test_valid(self):
        r = DeduplicationReport(
            ticker="AAPL",
            total_files=10,
            duplicate_groups=2,
            files_removed=3,
            files_kept=7,
        )
        assert r.groups == []


class TestDriftEntry:
    def test_valid(self):
        de = DriftEntry(
            date="2026-05-01",
            old_verdict="Buy",
            new_verdict="Strong Buy",
            old_expected_value=200.0,
            new_expected_value=220.0,
            delta_pct=10.0,
            key_changes=["Revenue outlook improved"],
        )
        assert de.date == "2026-05-01"


# ── Provider ─────────────────────────────────────────────────────────────

class TestSearchResult:
    def test_valid_with_href(self):
        """DDGS uses 'href' which maps to 'url'."""
        sr = SearchResult(href="https://example.com", title="Example page")
        assert sr.url == "https://example.com"

    def test_rejects_missing_url(self):
        with pytest.raises(ValueError):
            SearchResult(title="No URL")

    def test_ignores_extra_fields(self):
        sr = SearchResult(
            href="https://example.com",
            title="Example",
            unknown_extra="should be ignored",
        )
        assert sr.url == "https://example.com"


class TestNewsResult:
    def test_valid(self):
        nr = NewsResult(
            url="https://news.example.com/1",
            title="Breaking news",
        )
        assert nr.snippet == ""
        assert nr.published_at is None

    def test_rejects_missing_url(self):
        with pytest.raises(ValueError):
            NewsResult(title="Missing URL")


class TestScrapeResult:
    def test_valid(self):
        sr = ScrapeResult(url="https://example.com", content="Page content")
        assert not sr.truncated


# ── ThesisPillar & PillarOutcome ────────────────────────────────────────────
from schemas.rag import ThesisPillar, PillarOutcome, PillarType, PillarStatus


class TestThesisPillar:
    def test_minimal_construction(self):
        p = ThesisPillar(
            pillar_id="CRM-moat-001",
            pillar_type="moat",
            statement="Salesforce has a wide economic moat.",
        )
        assert p.pillar_id == "CRM-moat-001"
        assert p.pillar_type == "moat"
        assert p.statement == "Salesforce has a wide economic moat."
        assert p.rationale == ""
        assert p.valuation_impact == ""
        assert p.source_urls == []
        assert p.evidence_citations == []
        assert p.status == "supported"

    def test_full_construction(self):
        p = ThesisPillar(
            pillar_id="CRM-risk-001",
            pillar_type="risk",
            statement="Competitive pressure from Microsoft.",
            rationale="Microsoft Dynamics 365 grew 25% YoY.",
            valuation_impact="Reduces terminal value growth assumption by 1%.",
            source_urls=["https://example.com/report"],
            evidence_citations=["file.md#^block123"],
            status="weakened",
        )
        assert p.status == "weakened"
        assert len(p.source_urls) == 1
        assert len(p.evidence_citations) == 1

    def test_rejects_invalid_pillar_type(self):
        with pytest.raises(ValueError):
            ThesisPillar(
                pillar_id="X-001",
                pillar_type="invalid_type",
                statement="Bad.",
            )

    def test_rejects_invalid_status(self):
        with pytest.raises(ValueError):
            ThesisPillar(
                pillar_id="X-001",
                pillar_type="moat",
                statement="Valid claim.",
                status="deleted",
            )

    def test_all_valid_pillar_types(self):
        valid: list[PillarType] = [
            "moat", "growth", "risk", "valuation_assumption",
            "capital_allocation", "thesis_change",
        ]
        for pt in valid:
            p = ThesisPillar(pillar_id="X-001", pillar_type=pt, statement="Test.")
            assert p.pillar_type == pt

    def test_all_valid_statuses(self):
        valid: list[PillarStatus] = ["supported", "weakened", "superseded", "contradicted", "stale"]
        for s in valid:
            p = ThesisPillar(pillar_id="X-001", pillar_type="moat", statement="Test.", status=s)
            assert p.status == s

    def test_rejects_legacy_updated_as_stored_status(self):
        with pytest.raises(ValueError):
            ThesisPillar(
                pillar_id="X-001",
                pillar_type="moat",
                statement="Test.",
                status="updated",
            )

    def test_serialization_roundtrip(self):
        p = ThesisPillar(
            pillar_id="CRM-moat-001",
            pillar_type="moat",
            statement="Salesforce has a wide moat.",
            rationale="High switching costs.",
            valuation_impact="Supports premium multiple.",
            source_urls=["https://example.com"],
            evidence_citations=["file.md#^block123"],
            status="supported",
        )
        data = p.model_dump(mode="json")
        p2 = ThesisPillar(**data)
        assert p2.pillar_id == p.pillar_id
        assert p2.source_urls == p.source_urls


class TestPillarOutcome:
    def test_minimal_construction(self):
        o = PillarOutcome(
            memory_id="550e8400-e29b-41d4-a716-446655440000",
            pillar_id="CRM-moat-001",
            status="supported",
        )
        assert o.memory_id == "550e8400-e29b-41d4-a716-446655440000"
        assert o.pillar_id == "CRM-moat-001"
        assert o.status == "supported"
        assert o.reason == ""
        assert o.replacement_statement == ""
        assert o.source_urls == []

    def test_updated_outcome_maps_to_revised(self):
        o = PillarOutcome(
            memory_id="550e8400-e29b-41d4-a716-446655440000",
            pillar_id="CRM-moat-001",
            status="updated",
            reason="Revenue growth accelerated beyond prior estimate.",
            replacement_statement="CRM now growing at 15% vs prior 12% assumption.",
            source_urls=["https://example.com/new-earnings"],
        )
        assert o.status == "revised"
        assert o.replacement_statement != ""

    def test_rejects_invalid_status(self):
        with pytest.raises(ValueError):
            PillarOutcome(
                memory_id="550e8400-e29b-41d4-a716-446655440000",
                pillar_id="X-001",
                status="deleted",
            )

    def test_serialization_roundtrip(self):
        o = PillarOutcome(
            memory_id="550e8400-e29b-41d4-a716-446655440000",
            pillar_id="CRM-moat-001",
            status="contradicted",
            reason="New competitor entered the market.",
            source_urls=["https://example.com/competitor"],
        )
        data = o.model_dump(mode="json")
        o2 = PillarOutcome(**data)
        assert o2.memory_id == o.memory_id
        assert o2.reason == o.reason
