# pyright: reportArgumentType=false, reportOptionalSubscript=false
from agents import supervisor as supervisor_mod
from agents.supervisor import (
    _build_research_tasks,
    _dependencies_complete,
    _has_current_source_inventory,
    _next_pending_task,
    _source_inventory_from_vault,
)


def _state(**overrides):
    base = {
        "date": "2026-05-09",
        "run_datetime": "2026-05-09T10:00:00",
        "company": "Generic Corp Inc",
        "ticker": "GNC",
        "price_data": None,
        "bull_thesis": "",
        "bear_thesis": "",
        "sources": [],
        "judge_decision": "",
        "valuation": None,
        "final_report": "",
        "citation_manifest": [],
        "curator_log": "",
        "active_memory_ids": [],
        "vault_artifacts": [],
        "rag_context": "",
        "retrieved_memory_ids": [],
        "research_topics": [],
        "research_goal": "",
        "source_inventory": [],
        "research_tasks": [],
        "current_task_id": "",
        "research_findings": [],
        "research_synthesis": "",
        "retrieved_memory_outcomes": {},
        "thesis_pillars": [],
        "pillar_outcomes": [],
    }
    base.update(overrides)
    return base


def test_research_tasks_create_single_deep_agent_task():
    tasks = _build_research_tasks(_state(), source_inventory=[])

    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "deep_agent_research"
    assert tasks[0]["kind"] == "research_synthesis"
    assert "write_todos" in " ".join(tasks[0]["acceptance_criteria"])


def test_current_source_inventory_does_not_change_single_research_task():
    inventory = [{
        "ticker": "GNC",
        "source_type": "source_inventory",
        "title": "Current source inventory",
        "filed_at": "2026-05-09",
        "freshness": "current",
    }]
    tasks = _build_research_tasks(_state(), source_inventory=inventory)

    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "deep_agent_research"


def test_old_source_inventory_schedules_source_discovery():
    inventory = [{
        "ticker": "GNC",
        "source_type": "source_inventory",
        "title": "Old source inventory",
        "filed_at": "2026-02-17",
        "freshness": "current",
    }]
    assert _has_current_source_inventory(inventory, "2026-05-09") is False


def test_invalid_or_missing_filed_at_schedules_source_discovery():
    invalid_inventory = [{
        "ticker": "GNC",
        "source_type": "source_inventory",
        "title": "Invalid source inventory",
        "filed_at": "not-a-date",
        "freshness": "current",
    }]
    missing_inventory = [{
        "ticker": "GNC",
        "source_type": "source_inventory",
        "title": "Missing source inventory",
        "freshness": "current",
    }]

    assert _has_current_source_inventory(invalid_inventory, "2026-05-09") is False
    assert _has_current_source_inventory(missing_inventory, "2026-05-09") is False


def test_prior_pillars_do_not_add_outer_tasks():
    tasks = _build_research_tasks(
        _state(retrieved_memory_ids=["mem-1", "mem-2"]),
        source_inventory=[],
    )

    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "deep_agent_research"


def test_next_pending_task_respects_dependencies():
    tasks = [
        {"task_id": "T1", "status": "pending", "depends_on": []},
        {"task_id": "T2", "status": "pending", "depends_on": ["T1"]},
    ]

    assert _next_pending_task(tasks)["task_id"] == "T1"
    assert _dependencies_complete(tasks[1], tasks) is False


def test_blocked_dependency_is_terminal_not_deadlock():
    tasks = [
        {"task_id": "T1", "status": "blocked", "depends_on": []},
        {"task_id": "T2", "status": "pending", "depends_on": ["T1"]},
    ]

    assert _dependencies_complete(tasks[1], tasks) is True
    assert _next_pending_task(tasks)["task_id"] == "T2"


def test_source_inventory_from_vault_uses_parse_frontmatter(monkeypatch, tmp_path):
    ticker_dir = tmp_path / "GNC"
    ticker_dir.mkdir()
    source_path = ticker_dir / "2026-05-01_test.md"
    source_path.write_text("---\nsource_type: ignored\n---\n\nBody", encoding="utf-8")
    calls = []

    def fake_parse_frontmatter(text):
        calls.append(text)
        return {
            "source_type": "sec_filing",
            "form_type": "10-Q",
            "filing_date": "2026-05-01",
            "url": "https://www.sec.gov/example",
            "accession_number": "0000000000-26-000001",
        }, "Body"

    monkeypatch.setattr(supervisor_mod, "VAULT_ROOT", tmp_path)
    monkeypatch.setattr(supervisor_mod, "_parse_frontmatter", fake_parse_frontmatter)

    records = _source_inventory_from_vault("GNC")

    assert calls
    assert records[0]["form_type"] == "10-Q"
    assert records[0]["filed_at"] == "2026-05-01"
