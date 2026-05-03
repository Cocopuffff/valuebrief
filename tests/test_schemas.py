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


# ── Workflow ─────────────────────────────────────────────────────────────

class TestWorkflowStateModel:
    def test_valid_minimal(self):
        ws = WorkflowStateModel(
            date="2026-05-01",
            run_datetime="2026-05-01T12:00:00",
            company="Apple Inc.",
            ticker="AAPL",
        )
        assert ws.bull_thesis == ""
        assert ws.vault_artifacts == []

    def test_rejects_missing_required(self):
        with pytest.raises(ValueError):
            WorkflowStateModel(date="2026-05-01")


class TestResearchStateModel:
    def test_valid_minimal(self):
        rs = ResearchStateModel(
            date="2026-05-01",
            company="Apple Inc.",
            ticker="AAPL",
        )
        assert rs.max_iterations == 3
        assert rs.iteration_count == 0

    def test_defaults(self):
        rs = ResearchStateModel(
            date="2026-05-01",
            company="Tesla Inc.",
            ticker="TSLA",
        )
        assert rs.research_topics == []
        assert rs.key_points == []
