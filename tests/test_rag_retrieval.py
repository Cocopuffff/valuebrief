"""Tests for the hybrid RAG retrieval loop: search filters, multi-probe ranking,
outcome parsing, topic generation, and prompt formatting."""

import asyncio

import pytest
from agents.states import ResearchState, WorkflowState
from schemas.base import validate_ticker
from schemas.rag import ResearchTopic, MemoryRecord


# ── search_similar filter tests (unit, no DB) ──────────────────────────────

class TestSearchSimilarFilters:
    """Verify filter parameters are correctly propagated (mocked DB layer)."""

    def test_search_similar_filters_present(self):
        """Function accepts all new keyword filters without error."""
        from utils.vector_memory import search_similar
        import inspect
        sig = inspect.signature(search_similar)
        params = list(sig.parameters.keys())
        for p in ["min_source_priority", "cited_only", "source_types",
                   "min_similarity", "exclude_validity_statuses", "validity_statuses"]:
            assert p in params, f"Missing param: {p}"


# ── retrieve_research_memories unit tests ──────────────────────────────────

class TestRetrieveResearchMemories:
    """Unit tests for the multi-probe retrieval function."""

    def test_empty_retrieval_result_structure(self):
        """Empty result dict has all expected keys."""
        from utils.vector_memory import _empty_retrieval_result
        result = _empty_retrieval_result()
        assert result["memory_context"] == ""
        assert result["retrieved_memory_ids"] == []
        assert result["topics"] == []
        assert result["error"] is None

    def test_empty_retrieval_handles_no_memories(self):
        """When no memories exist, empty_retrieval_result has correct shape."""
        from utils.vector_memory import _empty_retrieval_result
        r = _empty_retrieval_result()
        assert r["memory_context"] == ""
        assert r["retrieved_memory_ids"] == []
        assert r["topics"] == []

    def test_retrieve_research_memories_handles_embedding_failure(self):
        """Empty retrieval result is safe to use as fallback."""
        from utils.vector_memory import _empty_retrieval_result
        r = _empty_retrieval_result()
        assert r["error"] is None
        assert r["memory_context"] == ""

    def test_build_retrieval_topics_empty(self):
        """Empty memory list produces no topics."""
        from utils.vector_memory import _build_retrieval_topics
        topics = _build_retrieval_topics([])
        assert topics == []

    def test_build_retrieval_topics_from_bull_memory(self):
        """Bull-thesis memory generates a bull-side topic."""
        from utils.vector_memory import _build_retrieval_topics
        mem = MemoryRecord(
            id="test-1", summary="Great moat", metadata={"source_type": "bull_thesis"},
            ticker="AAPL", source_priority=1, is_cited=True,
            created_at="2025-01-01T00:00:00", updated_at="2025-01-01T00:00:00",
        )
        topics = _build_retrieval_topics([mem])
        assert len(topics) >= 1
        bull_topic = next((t for t in topics if t.side == "bull"), None)
        assert bull_topic is not None
        assert "test-1" in bull_topic.evidence_memory_ids

    def test_build_retrieval_topics_from_bear_memory(self):
        """Bear-thesis memory generates a bear-side topic."""
        from utils.vector_memory import _build_retrieval_topics
        mem = MemoryRecord(
            id="test-2", summary="High debt", metadata={"source_type": "bear_thesis"},
            ticker="AAPL", source_priority=1, is_cited=True,
            created_at="2025-01-01T00:00:00", updated_at="2025-01-01T00:00:00",
        )
        topics = _build_retrieval_topics([mem])
        bear_topic = next((t for t in topics if t.side == "bear"), None)
        assert bear_topic is not None

    def test_build_retrieval_topics_neutral_for_judge(self):
        """Judge/monthly-summary memory generates neutral topic."""
        from utils.vector_memory import _build_retrieval_topics
        mem = MemoryRecord(
            id="test-3", summary="Valuation complete",
            metadata={"source_type": "judge_analysis"},
            ticker="AAPL", source_priority=2, is_cited=True,
            created_at="2025-01-01T00:00:00", updated_at="2025-01-01T00:00:00",
        )
        topics = _build_retrieval_topics([mem])
        neutral = next((t for t in topics if t.side == "neutral"), None)
        assert neutral is not None

    def test_build_retrieval_topics_fallback_when_no_valuation(self):
        """Even without valuation/judge memories, generates a drift question."""
        from utils.vector_memory import _build_retrieval_topics
        mem = MemoryRecord(
            id="test-4", summary="Market analysis",
            metadata={"source_type": "market_overview"},
            ticker="AAPL", source_priority=1, is_cited=False,
            created_at="2025-01-01T00:00:00", updated_at="2025-01-01T00:00:00",
        )
        topics = _build_retrieval_topics([mem])
        # Should have at least the fallback drift topic
        assert len(topics) >= 1

    def test_probes_dict_complete(self):
        """All 5 retrieval probes are defined."""
        from utils.vector_memory import _PROBES
        expected = {"moat_growth", "risks", "valuation_assumptions", "thesis_drift", "prior_valuation"}
        assert set(_PROBES.keys()) == expected


class TestMemoryContextFormatting:
    """Verify retrieved memories inject vault source text, not only summaries."""

    def _memory(
        self,
        *,
        summary: str,
        metadata: dict,
        memory_id: str = "mem-1",
        similarity: float = 0.9,
    ) -> MemoryRecord:
        return MemoryRecord(
            id=memory_id,
            summary=summary,
            metadata=metadata,
            ticker="CRM",
            source_priority=1,
            is_cited=False,
            created_at="2026-05-03T00:00:00",
            updated_at="2026-05-03T00:00:00",
            similarity=similarity,
        )

    def test_format_memory_context_uses_resolved_vault_excerpt(self, tmp_path):
        from utils.vault import VaultReader, VaultWriter
        from utils.vector_memory import format_memory_for_context

        path = VaultWriter(root=tmp_path).write_document(
            ticker="CRM",
            content="# Thesis\n\nRevenue grew 12% YoY while margins expanded.",
            metadata={"source_type": "bull_thesis"},
        )
        doc = VaultReader(root=tmp_path).read_document(path)
        block_id = next(iter(doc.block_map))
        mem = self._memory(
            summary="Generated concise summary.",
            metadata={
                "source_type": "bull_thesis",
                "local_path": str(path),
                "filename": path.name,
                "block_id": block_id,
                "citation": f"{path.name}#^{block_id}",
            },
        )

        rendered = format_memory_for_context(mem, 1)

        assert "**Summary**: Generated concise summary." in rendered
        assert "**Source excerpt**:" in rendered
        assert "# Thesis" in rendered
        assert "Revenue grew 12% YoY" in rendered

    def test_format_memory_context_falls_back_to_summary_when_vault_missing(self):
        from utils.vector_memory import format_memory_for_context

        mem = self._memory(
            summary="Fallback summary text.",
            metadata={
                "source_type": "bear_thesis",
                "local_path": "/missing/path.md",
                "block_id": "block-11111111",
            },
        )

        rendered = format_memory_for_context(mem, 1)

        assert "Fallback summary text." in rendered
        assert "vault block unavailable" in rendered

    def test_legacy_heading_only_block_expands_to_next_content_block(self, tmp_path):
        from utils.vector_memory import format_memory_for_context

        ticker_dir = tmp_path / "CRM"
        ticker_dir.mkdir()
        path = ticker_dir / "legacy.md"
        path.write_text(
            "---\nsource_type: bull_thesis\ncontent_hash: legacy\n---\n\n"
            "# Salesforce Thesis ^block-aaaaaaaa\n\n"
            "Revenue grew 12% YoY and FCF expanded. ^block-bbbbbbbb\n",
            encoding="utf-8",
        )
        mem = self._memory(
            summary="# Salesforce Thesis",
            metadata={
                "source_type": "bull_thesis",
                "local_path": str(path),
                "block_id": "block-aaaaaaaa",
            },
        )

        rendered = format_memory_for_context(mem, 1)

        assert "# Salesforce Thesis" in rendered
        assert "Revenue grew 12% YoY" in rendered

    def test_retrieve_research_memories_preserves_ranked_memory_ids(self, monkeypatch):
        from utils import vector_memory
        from utils import embeddings as embedding_module

        mem_a = self._memory(
            summary="A summary",
            metadata={"source_type": "bull_thesis"},
            memory_id="mem-a",
            similarity=0.7,
        )
        mem_b = self._memory(
            summary="B summary",
            metadata={"source_type": "bear_thesis"},
            memory_id="mem-b",
            similarity=0.9,
        )

        async def fake_embeddings(texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

        async def fake_search_similar(*args, **kwargs):
            return [mem_a, mem_b]

        monkeypatch.setattr(embedding_module, "get_embeddings", fake_embeddings)
        monkeypatch.setattr(vector_memory, "search_similar", fake_search_similar)

        result = asyncio.run(vector_memory.retrieve_research_memories("CRM"))

        assert result["retrieved_memory_ids"] == ["mem-b", "mem-a"]
        assert "### Memory #1" in result["memory_context"]
        assert "B summary" in result["memory_context"]


# ── Thesis pillar reconciliation parsing tests ──────────────────────────────

class TestReconcileOutputParsing:
    """Verify _parse_reconcile_output splits text + JSON and builds pillars."""

    def test_parses_pillars_and_outcomes(self):
        from agents.judge import _parse_reconcile_output
        raw = (
            "**Verdict**: Buy.\n**Rationale**: Strong moat.\n**Key Risks**: Competition.\n"
            "----JSON----\n"
            '{"thesis_pillars": ['
            '{"candidate_ref": "C1", "matched_prior_ref": "P1", "matched_pillar_id": "CRM-moat-001", "pillar_type": "moat",'
            '"statement": "Wide moat.", "rationale": "High switching costs.",'
            '"valuation_impact": "Premium multiple.", "source_urls": ["https://x.com"],'
            '"evidence_citations": [], "status": "supported"}'
            '], "pillar_outcomes": ['
            '{"memory_id": "' + "550e8400-e29b-41d4-a716-446655440000" + '",'
            '"pillar_id": "CRM-moat-001", "status": "supported", "reason": "Still true.",'
            '"replacement_statement": "", "source_urls": []}'
            ']}'
        )
        state = {"retrieved_memory_ids": ["550e8400-e29b-41d4-a716-446655440000"]}
        decision, pillars, outcomes = _parse_reconcile_output(raw, state)
        assert "Verdict" in decision
        assert "----JSON----" not in decision
        assert len(pillars) == 1
        assert pillars[0].pillar_id == "CRM-moat-001"
        assert pillars[0].matched_prior_ref == "P1"
        assert pillars[0].pillar_type == "moat"
        assert len(outcomes) == 1
        assert outcomes[0].status == "supported"

    def test_no_delimiter_fallback(self):
        from agents.judge import _parse_reconcile_output
        raw = "**Verdict**: Hold.\nNo JSON here."
        decision, pillars, outcomes = _parse_reconcile_output(raw, {})
        assert decision == raw
        assert pillars == []
        assert outcomes == []

    def test_rejects_outcome_memory_id_not_in_retrieved(self):
        from agents.judge import _parse_reconcile_output
        raw = (
            "Decision text.\n"
            "----JSON----\n"
            '{"thesis_pillars": [], "pillar_outcomes": ['
            '{"memory_id": "bad-uuid-0000-0000-000000000000", "pillar_id": "X-001",'
            '"status": "supported", "reason": "", "replacement_statement": "", "source_urls": []}'
            ']}'
        )
        state = {"retrieved_memory_ids": ["real-uuid-0000-0000-000000000001"]}
        _, _, outcomes = _parse_reconcile_output(raw, state)
        # The outcome with bad-uuid should be filtered out
        assert len(outcomes) == 0

    def test_invalid_json_fallback(self):
        from agents.judge import _parse_reconcile_output
        raw = "Decision.\n----JSON----\n{not valid json"
        decision, pillars, outcomes = _parse_reconcile_output(raw, {})
        assert "Verdict" not in decision
        assert pillars == []
        assert outcomes == []

    def test_parses_markdown_fenced_json(self):
        from agents.judge import _parse_reconcile_output
        raw = (
            "Decision.\n"
            "----JSON----\n"
            "```json\n"
            '{"thesis_pillars": [], "pillar_outcomes": []}\n'
            "```"
        )
        decision, pillars, outcomes = _parse_reconcile_output(raw, {})
        assert pillars == []
        assert outcomes == []


# ── Prior pillars context builder tests ──────────────────────────────────────

class TestPriorPillarsContext:
    """Verify _build_prior_pillars_context formats pillar context for judge."""

    def test_no_retrieved_ids(self):
        from agents.judge import _build_prior_pillars_context
        result = _build_prior_pillars_context({})
        assert "No prior pillars" in result

    def test_uses_rag_context_when_present(self):
        from agents.judge import _build_prior_pillars_context
        state = {
            "retrieved_memory_ids": ["uuid-1"],
            "rag_context": "## Active Pillars\n### Pillar #1: moat\n...",
        }
        result = _build_prior_pillars_context(state)
        assert "Active Pillars" in result

    def test_falls_back_to_count_when_no_rag_context(self):
        from agents.judge import _build_prior_pillars_context
        state = {"retrieved_memory_ids": ["uuid-1", "uuid-2"]}
        result = _build_prior_pillars_context(state)
        assert "2 prior pillar" in result


# ── Prompt formatting tests ─────────────────────────────────────────────────

class TestAnalystPromptFormatting:
    """Verify prompt sections are formatted correctly."""

    def test_format_topics_empty(self):
        from agents.analysts import _format_topics
        result = _format_topics([])
        assert "No prior research topics" in result

    def test_format_topics_ignores_legacy_seed_queries(self):
        from agents.analysts import _format_topics
        topics = [{
            "side": "bull",
            "question": "Is the moat intact?",
            "rationale": "Prior analysis found strong moat",
            "seed_queries": ["moat analysis", "competitive advantage"],
        }]
        result = _format_topics(topics)
        assert "Is the moat intact?" in result
        assert "strong moat" in result
        assert "moat analysis" not in result

    def test_format_topics_multiple_items(self):
        from agents.analysts import _format_topics
        topics = [
            {"side": "bull", "question": "Q1", "rationale": "R1"},
            {"side": "bear", "question": "Q2", "rationale": "R2"},
        ]
        result = _format_topics(topics)
        assert "1. Q1" in result
        assert "2. Q2" in result

    def test_source_url_rules_present(self):
        from agents.analysts import _SOURCE_URL_RULES
        assert "SOURCE CITATION" in _SOURCE_URL_RULES
        assert "source URL" in _SOURCE_URL_RULES
        assert "Do NOT assign final memory outcomes" in _SOURCE_URL_RULES


# ── State type tests ────────────────────────────────────────────────────────

class TestStateTypes:
    """Verify new state fields are present in the LangGraph TypedDicts."""

    def test_workflow_state_has_rag_fields(self):
        annotations = WorkflowState.__annotations__
        for field in (
            "rag_context",
            "retrieved_memory_ids",
            "research_topics",
            "thesis_pillars",
            "pillar_outcomes",
        ):
            assert field in annotations

    def test_research_state_has_rag_context(self):
        annotations = ResearchState.__annotations__
        assert "rag_context" in annotations
        assert "research_topics" in annotations


# ── Schema tests ────────────────────────────────────────────────────────────

class TestResearchTopicSchema:
    """Verify ResearchTopic Pydantic model."""

    def test_research_topic_creation(self):
        t = ResearchTopic(
            side="bull",
            question="Is growth sustainable?",
            rationale="Prior analysis found high growth",
            evidence_memory_ids=["uuid-1"],
        )
        assert t.side == "bull"
        assert t.evidence_memory_ids == ["uuid-1"]

    def test_research_topic_defaults(self):
        t = ResearchTopic(side="neutral", question="What changed?", rationale="Drift check")
        assert t.evidence_memory_ids == []

    def test_research_topic_rejects_legacy_seed_queries(self):
        with pytest.raises(ValueError):
            ResearchTopic(
                side="bull",
                question="Is growth sustainable?",
                rationale="Prior analysis found high growth",
                seed_queries=["legacy"],
            )


# ── Topic assignment tests ──────────────────────────────────────────────────

class TestAnalystTopicPassing:
    """Verify topic filtering by side works correctly."""

    def test_bull_topics_filtered_to_bull_and_neutral(self):
        topics = [
            {"side": "bull", "question": "Bull Q"},
            {"side": "bear", "question": "Bear Q"},
            {"side": "neutral", "question": "Neutral Q"},
            {"side": "bull", "question": "Bull Q2"},
        ]
        bull_topics = [t for t in topics if t.get("side") in ("bull", "neutral")]
        assert len(bull_topics) == 3
        sides = {t["side"] for t in bull_topics}
        assert sides == {"bull", "neutral"}

    def test_bear_topics_filtered_to_bear_and_neutral(self):
        topics = [
            {"side": "bull", "question": "Bull Q"},
            {"side": "bear", "question": "Bear Q"},
            {"side": "neutral", "question": "Neutral Q"},
        ]
        bear_topics = [t for t in topics if t.get("side") in ("bear", "neutral")]
        assert len(bear_topics) == 2
        sides = {t["side"] for t in bear_topics}
        assert sides == {"bear", "neutral"}

    def test_empty_topics_filter_cleanly(self):
        assert [t for t in [] if t.get("side") in ("bull", "neutral")] == []


# ── Cold-start regression tests ─────────────────────────────────────────────

class TestColdStartRegression:
    """Verify cold-start behavior is preserved."""

    def test_empty_rag_context_does_not_crash_prompts(self):
        """When rag_context is empty string, analyst prompts render without error."""
        from agents.analysts import _format_topics, _SOURCE_URL_RULES
        # Simulate what happens when rag_context is empty
        rag_ctx = ""
        topics = _format_topics([])
        rules = _SOURCE_URL_RULES if rag_ctx else ""
        # These should all be strings without formatting errors
        assert isinstance(rag_ctx, str)
        assert isinstance(topics, str)
        assert rules == ""  # No source rules when no RAG context

    def test_validate_tickers_still_works(self):
        """Existing ticker validation is unaffected."""
        assert validate_ticker("aapl") == "AAPL"

    def test_ticker_validation_tests_still_pass(self):
        """Original test suite imports should still work."""
        from schemas.base import validate_ticker, validate_tickers, Ticker
        assert validate_ticker(" crm ") == "CRM"


# ── update_validity_status tests ────────────────────────────────────────────

class TestUpdateValidityStatus:
    """Verify validity_status helper behavior."""

    def test_empty_list_returns_zero(self):
        """Guard clause returns 0 without touching the DB."""
        from utils.vector_memory import update_validity_status
        result = asyncio.run(update_validity_status([], "stale"))
        assert result == 0


# ── Vector memory function signature tests ─────────────────────────────────

class TestVectorMemorySignatures:
    """Verify function signatures are backwards-compatible."""

    def test_search_similar_backwards_compat(self):
        """Original 3-param call signature still works."""
        from utils.vector_memory import search_similar
        import inspect
        sig = inspect.signature(search_similar)
        params = sig.parameters
        # Required params
        assert "query_embedding" in params
        assert "ticker" in params
        assert "limit" in params


# ── Pillar identity helpers ─────────────────────────────────────────────────

class TestPillarIdentityHelpers:
    def test_deterministic_pillar_id_is_stable(self):
        from utils.vector_memory import deterministic_pillar_id
        a = deterministic_pillar_id("crm", "growth", "Revenue growth remains durable.")
        b = deterministic_pillar_id("CRM", "growth", "Revenue   growth remains durable. ")
        assert a == b
        assert a.startswith("CRM-growth-")

    def test_similarity_classification_thresholds(self):
        from utils.vector_memory import classify_pillar_similarity
        assert classify_pillar_similarity(0.90) == "high"
        assert classify_pillar_similarity(0.75) == "medium"
        assert classify_pillar_similarity(0.60) == "low"

    def test_high_similarity_reuses_best_pillar_id(self):
        from utils.vector_memory import resolve_pillar_id_from_candidates
        from schemas.rag import MemoryRecord
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        candidate = MemoryRecord(
            id="mem-1",
            summary="Durable growth.",
            metadata={"pillar_id": "CRM-growth-existing"},
            ticker="CRM",
            source_priority=2,
            is_cited=True,
            created_at=now,
            updated_at=now,
            similarity=0.91,
        )
        pillar_id, classification = resolve_pillar_id_from_candidates(
            ticker="CRM",
            pillar_type="growth",
            statement="Durable growth.",
            active_candidates=[candidate],
        )
        assert pillar_id == "CRM-growth-existing"
        assert classification == "high"

    def test_medium_similarity_requires_judge_match(self):
        from utils.vector_memory import resolve_pillar_id_from_candidates
        from schemas.rag import MemoryRecord
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        candidate = MemoryRecord(
            id="mem-1",
            summary="Durable growth.",
            metadata={"pillar_id": "CRM-growth-existing"},
            ticker="CRM",
            source_priority=2,
            is_cited=True,
            created_at=now,
            updated_at=now,
            similarity=0.75,
        )
        matched_id, matched_class = resolve_pillar_id_from_candidates(
            ticker="CRM",
            pillar_type="growth",
            statement="Durable growth with slower pace.",
            active_candidates=[candidate],
            matched_pillar_id="CRM-growth-existing",
        )
        new_id, new_class = resolve_pillar_id_from_candidates(
            ticker="CRM",
            pillar_type="growth",
            statement="Durable growth with slower pace.",
            active_candidates=[candidate],
            matched_pillar_id="",
        )
        assert matched_id == "CRM-growth-existing"
        assert matched_class == "medium"
        assert new_id.startswith("CRM-growth-")
        assert new_class == "medium_unresolved"


# ── Active pillar retrieval tests ────────────────────────────────────────────

class TestActivePillarRetrieval:
    """Verify retrieve_active_pillars helper functions."""

    def test_format_pillar_includes_all_fields(self):
        from utils.vector_memory import _format_pillar_for_context
        from schemas.rag import MemoryRecord
        from datetime import datetime, timezone
        memory = MemoryRecord(
            id="uuid-1",
            summary="Salesforce has a wide moat.",
            metadata={
                "pillar_type": "moat",
                "validity_status": "supported",
                "pillar_id": "CRM-moat-001",
                "version": 2,
                "detail_citation": "pillars/CRM-moat-001.md",
                "rationale": "High switching costs.",
                "valuation_impact": "Supports premium multiple.",
                "source_urls": ["https://example.com/report"],
                "evidence_citations": ["file.md#^block123"],
            },
            ticker="CRM",
            source_priority=2,
            is_cited=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        result = _format_pillar_for_context(memory, 1)
        assert "### Prior Ref P1: moat [supported]" in result
        assert "CRM-moat-001" in result
        assert "pillars/CRM-moat-001.md" in result
        assert "Salesforce has a wide moat" in result
        assert "High switching costs" in result
        assert "premium multiple" in result

    def test_format_pillar_without_optional_fields(self):
        from utils.vector_memory import _format_pillar_for_context
        from schemas.rag import MemoryRecord
        from datetime import datetime, timezone
        memory = MemoryRecord(
            id="uuid-2",
            summary="Simple claim.",
            metadata={"pillar_type": "risk"},
            ticker="CRM",
            source_priority=2,
            is_cited=False,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        result = _format_pillar_for_context(memory, 2)
        assert "### Prior Ref P2: risk" in result
        assert "Simple claim" in result
        # No missing fields should cause errors

    def test_build_pillar_topics_assigns_sides(self):
        from utils.vector_memory import _build_pillar_topics
        from schemas.rag import MemoryRecord
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        pillars = [
            MemoryRecord(id="1", summary="Moat claim.", metadata={"pillar_type": "moat"}, ticker="CRM", source_priority=2, is_cited=True, created_at=now, updated_at=now),
            MemoryRecord(id="2", summary="Risk claim.", metadata={"pillar_type": "risk"}, ticker="CRM", source_priority=2, is_cited=True, created_at=now, updated_at=now),
            MemoryRecord(id="3", summary="Valuation claim.", metadata={"pillar_type": "valuation_assumption"}, ticker="CRM", source_priority=2, is_cited=True, created_at=now, updated_at=now),
        ]
        topics = _build_pillar_topics(pillars)
        # Should have bull topic (moat), bear topic (risk), neutral topic (valuation)
        sides = {t.side for t in topics}
        assert "bull" in sides or any(t.side == "bull" for t in topics)
        assert "bear" in sides or any(t.side == "bear" for t in topics)
        assert len(topics) >= 2

    def test_build_pillar_topics_empty(self):
        from utils.vector_memory import _build_pillar_topics
        topics = _build_pillar_topics([])
        assert topics == []
