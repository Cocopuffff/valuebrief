# pyright: reportArgumentType=false, reportOptionalSubscript=false
import asyncio

from agents import orchestration


def _workflow_state():
    return {
        "date": "2026-05-09",
        "run_datetime": "",
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
        "research_goal": "Objective research.",
        "source_inventory": [],
        "research_tasks": [
            {
                "task_id": "deep_agent_research",
                "kind": "research_synthesis",
                "title": "Run bounded Deep Agent research",
                "objective": "Run one Deep Agent research pass.",
                "status": "in_progress",
                "depends_on": [],
                "acceptance_criteria": [],
                "source_requirements": [],
                "evidence_memory_ids": [],
                "mandatory": True,
            }
        ],
        "current_task_id": "deep_agent_research",
        "research_findings": [],
        "research_synthesis": "",
        "retrieved_memory_outcomes": {},
        "thesis_pillars": [],
        "pillar_outcomes": [],
    }


def test_run_neutral_research_marks_task_complete(monkeypatch):
    async def fake_run_research_task(state):
        return {
            "findings": [
                {
                    "task_id": "business_model",
                    "title": "Business model",
                    "summary": "Business model finding.",
                    "key_points": ["Point."],
                    "source_urls": ["https://example.com"],
                    "citations": [],
                    "confidence": 0.8,
                    "artifact_path": "",
                    "memory_ids": [],
                    "needs_follow_up": False,
                    "follow_up_reason": "",
                },
                {
                    "task_id": "risk_assessment",
                    "title": "Risks",
                    "summary": "Risk finding.",
                    "key_points": ["Risk point."],
                    "source_urls": ["https://example.com/risk"],
                    "citations": [],
                    "confidence": 0.8,
                    "artifact_path": "",
                    "memory_ids": [],
                    "needs_follow_up": False,
                    "follow_up_reason": "",
                },
            ],
            "synthesis": "Task synthesis",
            "sources": ["https://example.com"],
        }

    monkeypatch.setattr(orchestration, "run_research_task", fake_run_research_task)

    command = asyncio.run(orchestration.run_neutral_research(_workflow_state()))

    assert command.update["current_task_id"] == ""
    assert command.update["research_tasks"][0]["status"] == "completed"
    assert command.update["research_findings"][0]["summary"] == "Business model finding."
    assert command.update["research_findings"][1]["summary"] == "Risk finding."
