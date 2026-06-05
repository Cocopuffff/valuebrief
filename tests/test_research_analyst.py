# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
import asyncio
import json
import pathlib

from schemas import ResearchFinding, ResearchFindingBundle, VaultArtifact
from agents import analysts
from utils import config as config_mod


def _task_state():
    return {
        "date": "2026-05-09",
        "run_datetime": "2026-05-09T10:00:00",
        "company": "Generic Corp Inc",
        "ticker": "GNC",
        "price_data": None,
        "existing_valuation": None,
        "research_goal": "Objective research.",
        "task": {
            "task_id": "T1",
            "kind": "business_model",
            "title": "Business model",
            "objective": "Analyze business model.",
        },
        "research_tasks": [],
        "source_inventory": [],
        "rag_context": "",
        "prior_findings": [],
        "sources": [],
        "finding": {},
        "synthesis": "",
        "vault_artifacts": [],
        "active_memory_ids": [],
    }


class FakeDeepAgent:
    def __init__(self, results):
        self.results = list(results)
        self.payloads = []

    async def ainvoke(self, payload, config):
        self.payloads.append(payload)
        return self.results.pop(0)


class FakeMessage:
    def __init__(self, content):
        self.content = content


def test_deepagent_skill_files_are_seeded(tmp_path):
    skill_dir = tmp_path / "skills" / "citation-quality"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("---\nname: citation-quality\n---\n\nUse citations.", encoding="utf-8")

    files = analysts._load_skill_files(lambda text: {"content": text}, root=tmp_path / "skills")

    assert "/skills/citation-quality/SKILL.md" in files
    assert files["/skills/citation-quality/SKILL.md"]["content"].endswith("Use citations.")


def test_project_skills_exist():
    paths = analysts._skill_markdown_paths()
    names = {path.parent.name for path in paths}

    assert {"sec-filings", "earnings-transcripts", "value-investing-research", "citation-quality"} <= names
    assert all(isinstance(path, pathlib.Path) for path in paths)


def test_project_skill_names_match_directories():
    for path in analysts._skill_markdown_paths():
        frontmatter = path.read_text(encoding="utf-8").split("---", 2)[1]
        assert f"name: {path.parent.name}" in frontmatter


def test_research_prompt_has_no_turn_count_language():
    lowered = analysts.RESEARCH_ANALYST_SYSTEM.lower()

    assert "remaining" not in lowered
    assert "max_iterations" not in lowered
    assert "buy or sell bias" in lowered


def test_extract_deep_agent_structured_response():
    finding = ResearchFinding(
        task_id="T2",
        title="Business model",
        summary="Subscription revenue is recurring.",
        source_urls=["https://example.com"],
    )
    bundle = ResearchFindingBundle(findings=[finding], synthesis="Synthesis")

    result = analysts._extract_deep_agent_output(
        {"structured_response": bundle},
        {"task_id": "T2", "title": "Business model"},
    )

    assert result.findings[0].task_id == "T2"
    assert result.findings[0].source_urls == ["https://example.com"]


def test_extract_deep_agent_final_json_response():
    payload = {
        "findings": [
            {
                "task_id": "T2",
                "title": "Business model",
                "summary": "Subscription revenue is recurring.",
                "source_urls": ["https://example.com"],
            }
        ],
        "synthesis": "Synthesis",
    }

    result = analysts._extract_deep_agent_output(
        {"messages": [FakeMessage(f"```json\n{json.dumps(payload)}\n```")]},
        {"task_id": "T2", "title": "Business model"},
    )

    assert result.findings[0].task_id == "T2"
    assert result.synthesis == "Synthesis"


def test_extract_deep_agent_missing_structured_response_raises():
    try:
        analysts._extract_deep_agent_output(
            {"messages": [{"content": "plain text instead of structured output"}]},
            {"task_id": "T2", "title": "Business model"},
        )
    except ValueError as e:
        message = str(e)
        assert "structured_response" in message
        assert "Expected ResearchFindingBundle shape" in message
        assert "plain text instead of structured output" not in message
    else:
        raise AssertionError("Expected ValueError")


def test_deepseek_reasoner_disables_tool_forced_response_format(monkeypatch):
    monkeypatch.setattr(analysts.config.research, "provider", analysts.Provider.DEEPSEEK)
    monkeypatch.setattr(analysts.config.research, "model", "deepseek-reasoner-v1")

    assert analysts._deep_agent_response_format() is None


def test_deepseek_reasoner_tool_config_falls_back(monkeypatch):
    monkeypatch.setenv("RESEARCH_TOOL_FALLBACK_MODEL", "deepseek-chat")
    cfg = config_mod.AgentConfig(
        provider=config_mod.Provider.DEEPSEEK,
        model="deepseek-reasoner",
        thinking=True,
    )

    result = config_mod._tool_compatible_config(cfg, "RESEARCH")

    assert result.model == "deepseek-chat"
    assert result.thinking is False
    assert cfg.model == "deepseek-reasoner"


def test_deepseek_thinking_tool_config_keeps_reasoning_model():
    cfg = config_mod.AgentConfig(
        provider=config_mod.Provider.DEEPSEEK,
        model="deepseek-v4-pro",
        thinking=True,
    )

    result = config_mod._tool_compatible_config(cfg, "BULL")

    assert result.model == "deepseek-v4-pro"
    assert result.thinking is True


def test_deepseek_thinking_bind_tools_strips_tool_choice():
    from langchain_core.tools import tool
    from utils.deepseek_thinking import ChatDeepSeekThinking

    @tool
    def dummy_tool(query: str) -> str:
        """Return the query unchanged."""
        return query

    model = ChatDeepSeekThinking(
        model="deepseek-v4-pro",
        api_key="dummy",
        extra_body={"thinking": {"type": "enabled"}},
    )
    bound = model.bind_tools([dummy_tool], tool_choice="required")

    assert "tools" in bound.kwargs
    assert "tool_choice" not in bound.kwargs


def test_run_research_task_retries_malformed_output(monkeypatch):
    fake_agent = FakeDeepAgent([
        {"structured_response": {"findings": []}},
        {
            "structured_response": {
                "findings": [
                    {
                        "task_id": "T1",
                        "title": "Business model",
                        "summary": "Valid finding.",
                        "key_points": ["Valid point (https://example.com)."],
                        "source_urls": ["https://example.com"],
                        "confidence": 0.8,
                    }
                ],
                "synthesis": "Valid synthesis\n\n## Candidate Investment Pillars\n- Durable moat.",
            }
        },
    ])

    async def fake_persist_research_artifact(**kwargs):
        return VaultArtifact(
            ticker=kwargs["ticker"],
            path="",
            filename="",
            source_type=kwargs["source_type"],
        )

    monkeypatch.setattr(analysts, "_create_deep_research_agent", lambda tools=None: fake_agent)
    monkeypatch.setattr(analysts, "_create_file_data_fn", lambda: (lambda text: {"content": text}))
    monkeypatch.setattr(analysts, "persist_research_artifact", fake_persist_research_artifact)

    result = asyncio.run(analysts.run_research_task(_task_state()))

    assert len(fake_agent.payloads) == 2
    retry_prompt = fake_agent.payloads[1]["messages"][0].content
    assert "Your previous response was malformed" in retry_prompt
    assert "empty findings" in retry_prompt
    assert "Expected ResearchFindingBundle shape" in retry_prompt
    assert result["finding"]["summary"] == "Valid finding."
    assert "Candidate Investment Pillars" in result["synthesis"]


def test_run_research_task_falls_back_after_retries(monkeypatch):
    fake_agent = FakeDeepAgent([
        {"structured_response": {"findings": []}},
        {"structured_response": {"findings": []}},
        {"structured_response": {"findings": []}},
    ])

    async def fake_persist_research_artifact(**kwargs):
        return VaultArtifact(
            ticker=kwargs["ticker"],
            path="",
            filename="",
            source_type=kwargs["source_type"],
        )

    monkeypatch.setattr(analysts, "_create_deep_research_agent", lambda tools=None: fake_agent)
    monkeypatch.setattr(analysts, "_create_file_data_fn", lambda: (lambda text: {"content": text}))
    monkeypatch.setattr(analysts, "persist_research_artifact", fake_persist_research_artifact)

    result = asyncio.run(analysts.run_research_task(_task_state()))

    assert len(fake_agent.payloads) == 2
    assert result["finding"]["needs_follow_up"] is True
    assert result["finding"]["confidence"] == 0.1
    assert "after retries" in result["finding"]["follow_up_reason"]


def test_validate_finding_bundle_extracts_summary_urls():
    bundle = ResearchFindingBundle(
        findings=[
            ResearchFinding(
                task_id="business_model",
                title="Business model",
                summary="Sales grew 10% (https://example.com/report).",
                key_points=["Recurring revenue remains high."],
            )
        ],
        synthesis="Synthesis\n\n## Candidate Investment Pillars\n- Revenue durability.",
    )

    validated = analysts._validate_finding_bundle(bundle)

    assert validated.findings[0].source_urls == ["https://example.com/report"]


def test_validate_finding_bundle_rejects_missing_semantic_fields():
    bundle = ResearchFindingBundle(
        findings=[
            ResearchFinding(
                task_id="source_inventory",
                title="Source inventory",
                summary="Found SEC filing.",
                source_urls=["https://www.sec.gov/example"],
            )
        ],
        synthesis="Synthesis without required heading.",
    )

    try:
        analysts._validate_finding_bundle(bundle)
    except ValueError as e:
        message = str(e)
        assert "key_points" in message
        assert "source_inventory" in message
        assert "Candidate Investment Pillars" in message
    else:
        raise AssertionError("Expected semantic validation failure")


def test_bounded_research_tool_returns_budget_exhausted():
    tools = analysts._build_research_tools(
        analysts.ResearchToolBudget(
            max_tool_calls=0,
            max_search_calls=0,
            max_scrape_calls=0,
        )
    )
    by_name = {tool.name: tool for tool in tools}

    result = asyncio.run(by_name["search"].ainvoke({"query": "CRM news"}))

    assert "budget exhausted" in result.lower()
    assert "synthesize" in result.lower()


def test_run_research_task_times_out(monkeypatch):
    class SlowAgent:
        async def ainvoke(self, payload, config):
            await asyncio.sleep(0.01)
            return {}

    async def fake_persist_research_artifact(**kwargs):
        return VaultArtifact(
            ticker=kwargs["ticker"],
            path="",
            filename="",
            source_type=kwargs["source_type"],
        )

    monkeypatch.setattr(analysts, "_create_deep_research_agent", lambda tools=None: SlowAgent())
    monkeypatch.setattr(analysts, "_create_file_data_fn", lambda: (lambda text: {"content": text}))
    monkeypatch.setattr(analysts, "persist_research_artifact", fake_persist_research_artifact)
    monkeypatch.setattr(analysts.config.research, "timeout_seconds", 0.001)

    result = asyncio.run(analysts.run_research_task(_task_state()))

    assert result["finding"]["needs_follow_up"] is True
    assert result["finding"]["confidence"] == 0.1
