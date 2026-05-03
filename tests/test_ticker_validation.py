"""Tests for ticker validation — schema-level and application boundary."""

import pytest

from schemas.base import validate_ticker, validate_tickers
from schemas.finance import Asset
from schemas.maintenance import DeduplicationReport
from schemas.rag import VaultDocument, VaultArtifact, InsightRecord, MemoryRecord
from schemas.valuation import ValuationModel
from schemas.workflow import WorkflowStateModel, ResearchStateModel


# ── validate_ticker unit tests ────────────────────────────────────────────

def test_validate_ticker_strips_and_uppercases():
    assert validate_ticker(" crm ") == "CRM"


def test_validate_ticker_valid_suffix():
    assert validate_ticker("BRK.B") == "BRK.B"


def test_validate_ticker_lowercase_normalises():
    assert validate_ticker("aapl") == "AAPL"


def test_validate_ticker_rejects_invalid():
    with pytest.raises(ValueError, match="Invalid ticker"):
        validate_ticker("BAD-TICKER")


def test_validate_ticker_rejects_non_string():
    with pytest.raises(ValueError, match="Invalid ticker"):
        validate_ticker(123)  # type: ignore[arg-type]


def test_validate_ticker_empty_string():
    with pytest.raises(ValueError, match="Invalid ticker"):
        validate_ticker("")


def test_validate_ticker_too_long():
    with pytest.raises(ValueError, match="Invalid ticker"):
        validate_ticker("TOOLONGTICKER")


# ── Ticker type alias in schema models ────────────────────────────────────

def test_asset_ticker_normalises_lowercase():
    a = Asset(ticker="aapl", name="Apple Inc.", current_price=150.0)
    assert a.ticker == "AAPL"


def test_asset_ticker_strips_whitespace():
    a = Asset(ticker="  msft  ", name="Microsoft", current_price=400.0)
    assert a.ticker == "MSFT"


def test_valuation_model_rejects_invalid_ticker():
    with pytest.raises(ValueError, match="Invalid ticker"):
        ValuationModel(
            ticker="BAD-TICKER",
            company="Test",
            current_price=100.0,
            base_revenue=50_000_000_000.0,
            shares_outstanding=15_000_000_000.0,
        )


def test_vault_document_normalises_ticker():
    doc = VaultDocument(path="/vault/test.md", ticker="  aapl ", content="test")
    assert doc.ticker == "AAPL"


def test_vault_artifact_normalises_ticker():
    art = VaultArtifact(
        ticker="  brk.b  ", path="/vault/", filename="test.md", source_type="analysis"
    )
    assert art.ticker == "BRK.B"


def test_insight_record_normalises_ticker():
    rec = InsightRecord(ticker="crm", summary="test", embedding=[0.1, 0.2])
    assert rec.ticker == "CRM"


def test_memory_record_normalises_ticker():
    rec = MemoryRecord(
        id="mem-001",
        ticker="googl",
        summary="test",
        metadata={},
        source_priority=1,
        is_cited=False,
        created_at="2025-01-01T00:00:00",
        updated_at="2025-01-01T00:00:00",
    )
    assert rec.ticker == "GOOGL"


def test_deduplication_report_normalises_ticker():
    rpt = DeduplicationReport(
        ticker="  aapl ", total_files=5, duplicate_groups=1, files_removed=2, files_kept=3
    )
    assert rpt.ticker == "AAPL"


def test_workflow_state_model_normalises_ticker():
    ws = WorkflowStateModel(
        date="2025-01-01", run_datetime="2025-01-01T00:00:00", company="Test", ticker="  crm  "
    )
    assert ws.ticker == "CRM"


def test_research_state_model_normalises_ticker():
    rs = ResearchStateModel(
        date="2025-01-01", company="Test", ticker="  crm  "
    )
    assert rs.ticker == "CRM"


# ── validate_tickers helper tests (no Postgres / LangGraph dependency) ────


def test_validate_tickers_normalises_list():
    result = validate_tickers([" aapl ", "  crm  ", "BRK.B"])
    assert result == ["AAPL", "CRM", "BRK.B"]


def test_validate_tickers_rejects_invalid_in_list():
    with pytest.raises(ValueError, match="Invalid ticker"):
        validate_tickers(["AAPL", "BAD-TICKER", "MSFT"])


def test_validate_tickers_collects_all_invalid():
    with pytest.raises(ValueError) as exc:
        validate_tickers(["BAD-ONE", "BAD-TWO"])
    msg = str(exc.value)
    assert "BAD-ONE" in msg
    assert "BAD-TWO" in msg


def test_validate_tickers_empty_list():
    assert validate_tickers([]) == []


# ── Regression: ValuationModel snapshot / alias hydration ─────────────────


_COMPUTED_KEYS_DATA = {
    "ticker": "CRM",
    "company": "Salesforce",
    "current_price": 250.0,
    "base_revenue": 35_000_000_000.0,
    "shares_outstanding": 800_000_000.0,
    "scenarios": {},
    "expected_value": 261.53,
    "expected_cagr": 0.151,
    "dispersion_ratio": 1.205,
    "recommendation": "Strong Buy",
}


def test_valuation_model_accepts_known_computed_keys():
    """Known computed-property input keys are accepted via validation aliases."""
    m = ValuationModel(**_COMPUTED_KEYS_DATA)
    assert m.ticker == "CRM"
    assert m.expected_value_snapshot == 261.53
    assert m.expected_cagr_snapshot == 0.151
    assert m.dispersion_ratio_snapshot == 1.205
    assert m.recommendation_snapshot == "Strong Buy"


def test_valuation_model_rejects_unknown_extras():
    """Extra keys that are NOT known computed-property aliases still fail."""
    with pytest.raises(ValueError, match="extra_forbidden|Extra inputs"):
        ValuationModel(
            ticker="AAPL",
            company="Apple",
            current_price=150.0,
            base_revenue=100e9,
            shares_outstanding=16e9,
            scenarios={},
            bogus_field=999,
        )


def test_snapshot_fields_excluded_from_model_dump():
    """Snapshot fields must not appear in model_dump — only the computed
    properties (expected_value, expected_cagr, etc.) should be present."""
    m = ValuationModel(**_COMPUTED_KEYS_DATA)
    d = m.model_dump(mode="json")
    assert "expected_value" in d  # computed field
    assert "expected_cagr" in d
    assert "dispersion_ratio" in d
    assert "recommendation" in d
    assert "expected_value_snapshot" not in d
    assert "expected_cagr_snapshot" not in d
    assert "dispersion_ratio_snapshot" not in d
    assert "recommendation_snapshot" not in d


# ── Scenario-first behavior ────────────────────────────────────────────────


def test_scenario_first_ignores_snapshot_when_intrinsic_values_present():
    """When scenarios have intrinsic values, computed properties use them,
    ignoring snapshot values entirely."""
    from schemas.valuation import DCFScenario, DCFAssumptions
    m = ValuationModel(
        ticker="AAPL",
        company="Apple",
        current_price=150.0,
        base_revenue=100e9,
        shares_outstanding=16e9,
        scenarios={
            "Bear": DCFScenario(
                label="Bear", probability=0.25,
                assumptions=DCFAssumptions(
                    revenue_growth_stage_1=0.05, revenue_growth_stage_2=0.03,
                    ebit_margin_target=0.20, tax_rate=0.21, wacc=0.12,
                    terminal_growth=0.02,
                ),
                intrinsic_value=120.0,
            ),
            "Base": DCFScenario(
                label="Base", probability=0.50,
                assumptions=DCFAssumptions(
                    revenue_growth_stage_1=0.10, revenue_growth_stage_2=0.06,
                    ebit_margin_target=0.25, tax_rate=0.21, wacc=0.10,
                    terminal_growth=0.025,
                ),
                intrinsic_value=200.0,
            ),
            "Bull": DCFScenario(
                label="Bull", probability=0.25,
                assumptions=DCFAssumptions(
                    revenue_growth_stage_1=0.18, revenue_growth_stage_2=0.10,
                    ebit_margin_target=0.30, tax_rate=0.21, wacc=0.08,
                    terminal_growth=0.03,
                ),
                intrinsic_value=300.0,
            ),
        },
        # Snapshot values that should be IGNORED
        expected_value=999.99,
        expected_cagr=0.999,
        dispersion_ratio=99.99,
        recommendation="Wrong Snapshot",
    )

    # Expected value: probability-weighted sum from scenario intrinsic values
    expected = round(0.25 * 120.0 + 0.50 * 200.0 + 0.25 * 300.0, 2)
    assert m.expected_value == expected
    assert m.expected_value != 999.99  # snapshot ignored
    # Dispersion: (max - min) / expected_value
    assert m.dispersion_ratio != 99.99  # snapshot ignored
    # Recommendation: computed from expected_value vs current_price
    assert m.recommendation != "Wrong Snapshot"  # snapshot ignored


def test_empty_scenarios_fall_back_to_snapshots():
    """When scenarios have no intrinsic values, computed properties use snapshots."""
    m = ValuationModel(**_COMPUTED_KEYS_DATA)
    # Empty scenarios dict with no intrinsic values → uses snapshots
    assert m.expected_value == 261.53
    assert m.expected_cagr == 0.151
    assert m.dispersion_ratio == 1.205
    # recommendation always computes from expected_value + current_price
    # when current_price > 0, so it won't be the snapshot string,
    # but it IS computed from the expected_value fallback
    assert m.recommendation in ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell")


# ── Construction-site regression (db.py / judge.py) ────────────────────────


def test_db_row_with_summary_columns_hydrates_directly():
    """A representative DB row (with expected_value, expected_cagr, etc.
    as columns) must construct ValuationModel without filtering."""
    row = {
        "ticker": "CRM",
        "company": "Salesforce",
        "current_price": 250.0,
        "currency": "USD",
        "base_revenue": 35_000_000_000.0,
        "shares_outstanding": 800_000_000.0,
        "scenarios": {},
        "thesis_data": None,
        "valuation_data": None,
        "expected_value": 261.53,
        "expected_cagr": 0.151,
        "dispersion_ratio": 1.205,
        "recommendation": "Strong Buy",
        "last_updated": "2025-01-01T00:00:00",
    }
    m = ValuationModel(**row)
    assert m.ticker == "CRM"
    assert m.expected_value_snapshot == 261.53
    assert m.recommendation_snapshot == "Strong Buy"


def test_llm_json_with_computed_keys_accepted_without_filtering():
    """_parse_valuation_response receives LLM JSON that may include
    computed metric keys. The model must accept them directly."""
    llm_output = {
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "current_price": 180.0,
        "base_revenue": 383_000_000_000.0,
        "shares_outstanding": 15_500_000_000.0,
        "scenarios": {
            "Base": {
                "label": "Base",
                "probability": 0.5,
                "assumptions": {
                    "revenue_growth_stage_1": 0.08,
                    "revenue_growth_stage_2": 0.05,
                    "ebit_margin_target": 0.30,
                    "tax_rate": 0.21,
                    "wacc": 0.09,
                    "terminal_growth": 0.025,
                },
                "intrinsic_value": None,
            },
        },
        # LLM may hallucinate these — model should accept them as snapshots
        "expected_value": 250.0,
        "expected_cagr": 0.12,
        "dispersion_ratio": 0.5,
        "recommendation": "Buy",
    }
    m = ValuationModel(**llm_output)
    assert m.ticker == "AAPL"
    assert m.expected_value_snapshot == 250.0
    assert m.expected_cagr_snapshot == 0.12
    # After compute_dcf, the computed values override snapshots
    m.compute_dcf(base_year=2025)
    assert m.expected_value != 250.0  # computed, not snapshot
