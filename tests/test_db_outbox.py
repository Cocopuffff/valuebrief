import asyncio
import json
import uuid
from pathlib import Path

import pytest

from schemas.rag import InsightRecord
from schemas.valuation import DCFAssumptions, DCFScenario, ValuationModel
from utils import db, db_outbox, drift_tracker, vector_memory


class AdminShutdownError(Exception):
    sqlstate = "57P01"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def _raise_transient(*args, **kwargs):
    raise AdminShutdownError("terminating connection due to administrator command")


def _valuation_model() -> ValuationModel:
    return ValuationModel(
        ticker="CRM",
        company="Salesforce.com Inc",
        current_price=250.0,
        base_revenue=35_000_000_000.0,
        shares_outstanding=1_000_000_000.0,
        scenarios={
            "Base": DCFScenario(
                label="Base",
                probability=1.0,
                assumptions=DCFAssumptions(
                    revenue_growth_stage_1=0.08,
                    revenue_growth_stage_2=0.04,
                    ebit_margin_target=0.25,
                    tax_rate=0.21,
                    wacc=0.10,
                    terminal_growth=0.025,
                ),
            )
        },
    )


def test_append_and_successful_drain_removes_outbox(tmp_path):
    outbox = tmp_path / "db_cud_outbox.jsonl"
    seen: list[str] = []

    async def replay(entry: dict) -> None:
        seen.append(entry["id"])

    asyncio.run(
        db_outbox.append_db_outbox_entry(
            operation="upsert",
            table="valuations",
            ticker="crm",
            payload={"ticker": "CRM"},
            error="lost",
            path=outbox,
        )
    )
    result = asyncio.run(db_outbox.drain_db_outbox(path=outbox, replay_fn=replay))

    assert result.total == 1
    assert result.succeeded == 1
    assert result.retained == 0
    assert seen
    assert not outbox.exists()


def test_transient_replay_failure_keeps_entry_and_increments_attempts(tmp_path):
    outbox = tmp_path / "db_cud_outbox.jsonl"
    asyncio.run(
        db_outbox.append_db_outbox_entry(
            operation="upsert",
            table="valuations",
            ticker="CRM",
            payload={"ticker": "CRM"},
            error="lost",
            path=outbox,
        )
    )

    result = asyncio.run(db_outbox.drain_db_outbox(path=outbox, replay_fn=_raise_transient))
    entries = _read_jsonl(outbox)

    assert result.retained == 1
    assert entries[0]["attempts"] == 1
    assert "administrator command" in entries[0]["error"]


def test_corrupt_jsonl_lines_move_to_bad_file(tmp_path):
    outbox = tmp_path / "db_cud_outbox.jsonl"
    bad = tmp_path / "db_cud_outbox.bad.jsonl"
    outbox.write_text("{not-json}\n", encoding="utf-8")

    result = asyncio.run(db_outbox.drain_db_outbox(path=outbox, bad_path=bad))

    assert result.corrupted == 1
    assert not outbox.exists()
    assert "{not-json}" in bad.read_text(encoding="utf-8")


def test_non_transient_replay_failure_is_retained(tmp_path):
    outbox = tmp_path / "db_cud_outbox.jsonl"

    async def replay(entry: dict) -> None:
        raise ValueError("bad query shape")

    asyncio.run(
        db_outbox.append_db_outbox_entry(
            operation="upsert",
            table="valuations",
            ticker="CRM",
            payload={"ticker": "CRM"},
            error="lost",
            path=outbox,
        )
    )
    result = asyncio.run(db_outbox.drain_db_outbox(path=outbox, replay_fn=replay))

    assert result.retained == 1
    assert _read_jsonl(outbox)[0]["error"] == "bad query shape"


def test_failed_memory_insert_queues_exact_generated_id(monkeypatch, tmp_path):
    outbox = tmp_path / "db_cud_outbox.jsonl"
    fixed_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")

    monkeypatch.setattr(db_outbox, "OUTBOX_PATH", outbox)
    monkeypatch.setattr(vector_memory.uuid, "uuid4", lambda: fixed_id)
    monkeypatch.setattr(vector_memory, "run_db_operation", _raise_transient)

    memory_id = asyncio.run(
        vector_memory.store_insight(
            InsightRecord(
                ticker="CRM",
                summary="Durable growth.",
                embedding=[0.1, 0.2],
                metadata={"source_type": "thesis_pillar"},
                source_priority=2,
                is_cited=True,
            )
        )
    )
    entry = _read_jsonl(outbox)[0]

    assert memory_id == str(fixed_id)
    assert entry["table"] == "investment_memories"
    assert entry["operation"] == "insert"
    assert entry["payload"]["records"][0]["id"] == str(fixed_id)


def test_failed_valuation_upsert_queues_model_payload(monkeypatch, tmp_path):
    outbox = tmp_path / "db_cud_outbox.jsonl"

    monkeypatch.setattr(db_outbox, "OUTBOX_PATH", outbox)
    monkeypatch.setattr(db, "run_db_operation", _raise_transient)

    asyncio.run(db.upsert_valuation(_valuation_model()))
    entry = _read_jsonl(outbox)[0]

    assert entry["table"] == "valuations"
    assert entry["operation"] == "upsert"
    assert entry["payload"]["ticker"] == "CRM"
    assert entry["payload"]["expected_value"] is not None


def test_failed_drift_insert_queues_payload(monkeypatch, tmp_path):
    outbox = tmp_path / "db_cud_outbox.jsonl"

    monkeypatch.setattr(db_outbox, "OUTBOX_PATH", outbox)
    monkeypatch.setattr(drift_tracker, "run_db_operation", _raise_transient)

    asyncio.run(
        drift_tracker.record_drift(
            ticker="CRM",
            old_verdict="Hold",
            new_verdict="Buy",
            old_expected_value=100.0,
            new_expected_value=120.0,
            key_changes=["Margin expanded"],
        )
    )
    entry = _read_jsonl(outbox)[0]

    assert entry["table"] == "thesis_drifts"
    assert entry["operation"] == "insert"
    assert entry["payload"]["delta_pct"] == 20.0


def test_main_drains_outbox_before_workflow_construction():
    source = Path("src/main.py").read_text(encoding="utf-8")

    assert source.index("await drain_db_outbox()") < source.index("workflow = build_research_workflow")
