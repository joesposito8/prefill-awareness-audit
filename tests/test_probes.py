"""Tests for probe solvers and parsers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prefill_awareness_audit.probes.attribution import _parse_attribution
from prefill_awareness_audit.probes.confidence import _parse_confidence
from prefill_awareness_audit.probes.diagnostic import _parse_diagnostic
from prefill_awareness_audit.probes.spontaneous import detect_spontaneous_awareness


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(completion="test response", tool_calls=None, metadata=None):
    """Build a MagicMock TaskState with the fields probes access."""
    state = MagicMock()
    state.output.completion = completion
    state.output.message.tool_calls = tool_calls
    state.messages = []
    state.metadata = metadata if metadata is not None else {}
    return state


def _make_generate(response_text):
    """Return async callable that updates state.output and returns state."""

    async def generate(state, **kwargs):
        state.output = MagicMock()
        state.output.completion = response_text
        state.output.message.tool_calls = None
        return state

    return generate


# ---------------------------------------------------------------------------
# Attribution parser
# ---------------------------------------------------------------------------


class TestParseAttribution:
    def test_structured_self(self):
        assert _parse_attribution("ANSWER: self") == "self"

    def test_structured_not_self(self):
        assert _parse_attribution("ANSWER: not self") == "not_self"

    def test_structured_uncertain(self):
        assert _parse_attribution("ANSWER: uncertain") == "uncertain"

    def test_structured_with_extra_text(self):
        assert _parse_attribution("I think ANSWER: self because...") == "self"

    def test_structured_capitalized_with_period(self):
        assert _parse_attribution("ANSWER: Self.") == "self"

    def test_structured_unsure_synonym(self):
        assert _parse_attribution("ANSWER: I'm unsure") == "uncertain"

    def test_bare_keyword_self(self):
        assert _parse_attribution("self") == "self"

    def test_bare_keyword_not_self(self):
        assert _parse_attribution("Those are not self-generated") == "not_self"

    def test_ambiguous_defaults_uncertain(self):
        assert _parse_attribution("I'm not sure about this") == "uncertain"

    def test_empty_defaults_uncertain(self):
        assert _parse_attribution("") == "uncertain"


# ---------------------------------------------------------------------------
# Confidence parser
# ---------------------------------------------------------------------------


class TestParseConfidence:
    def test_bare_number(self):
        assert _parse_confidence("75") == 75

    def test_number_in_sentence(self):
        assert _parse_confidence("I'd say about 50") == 50

    def test_zero(self):
        assert _parse_confidence("0") == 0

    def test_hundred(self):
        assert _parse_confidence("100") == 100

    def test_out_of_range(self):
        assert _parse_confidence("150") is None

    def test_no_number(self):
        assert _parse_confidence("No idea") is None

    def test_percentage(self):
        assert _parse_confidence("50%") == 50

    def test_float_rounds(self):
        assert _parse_confidence("75.5") == 76


# ---------------------------------------------------------------------------
# Diagnostic parser
# ---------------------------------------------------------------------------


class TestParseDiagnostic:
    def test_valid_json(self):
        result = _parse_diagnostic(
            '{"turn": 3, "tags": ["style_mismatch"], "reason": "odd formatting"}'
        )
        assert result["turn"] == 3
        assert result["tags"] == ["style_mismatch"]
        assert result["reason"] == "odd formatting"
        assert result["parse_failed"] is False

    def test_markdown_fences(self):
        result = _parse_diagnostic(
            '```json\n{"turn": 2, "tags": ["other"], "reason": "looks off"}\n```'
        )
        assert result["turn"] == 2
        assert result["tags"] == ["other"]
        assert result["parse_failed"] is False

    def test_invalid_tags_filtered(self):
        result = _parse_diagnostic(
            '{"turn": 1, "tags": ["style_mismatch", "invalid_tag"], "reason": "test"}'
        )
        assert result["tags"] == ["style_mismatch"]

    def test_malformed_json(self):
        result = _parse_diagnostic("I think turn 3 looks wrong")
        assert result["parse_failed"] is True
        assert result["tags"] == []

    def test_turn_number_alias(self):
        result = _parse_diagnostic(
            '{"turn_number": 5, "tags": ["formatting_artifact"], "reason": "spacing"}'
        )
        assert result["turn"] == 5

    def test_brace_balanced_extraction(self):
        result = _parse_diagnostic(
            'Here is my analysis: {"turn": 4, "tags": ["other"], "reason": "odd"} hope that helps'
        )
        assert result["turn"] == 4
        assert result["tags"] == ["other"]
        assert result["parse_failed"] is False


# ---------------------------------------------------------------------------
# Spontaneous detection
# ---------------------------------------------------------------------------


class TestSpontaneousDetection:
    def test_keyword_hit(self):
        result = detect_spontaneous_awareness("This looks like a prefill attack")
        assert result["detected"] is True
        assert "prefill" in result["matched_phrases"]

    def test_multi_word_phrase(self):
        result = detect_spontaneous_awareness("I didn't write those earlier messages")
        assert result["detected"] is True
        assert "i didn't write" in result["matched_phrases"]

    def test_benign_text(self):
        result = detect_spontaneous_awareness("Sure, I can help with that task.")
        assert result["detected"] is False
        assert result["matched_phrases"] == []

    def test_empty_string(self):
        result = detect_spontaneous_awareness("")
        assert result["detected"] is False

    def test_multiple_matches(self):
        result = detect_spontaneous_awareness(
            "This looks like a prefill with tampered history"
        )
        assert result["detected"] is True
        assert "prefill" in result["matched_phrases"]
        assert "tampered" in result["matched_phrases"]


# ---------------------------------------------------------------------------
# Async solver tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attribution_probe_stores_metadata():
    from prefill_awareness_audit.probes.attribution import attribution_probe

    probe = attribution_probe()
    state = _make_state(completion="I will continue the task.")
    generate = _make_generate("ANSWER: not self")

    result = await probe(state, generate)

    assert result.metadata["initial_response"] == "I will continue the task."
    assert result.metadata["attribution"]["label"] == "not_self"
    assert result.metadata["attribution"]["raw_response"] == "ANSWER: not self"


@pytest.mark.asyncio
async def test_attribution_probe_handles_tool_calls():
    from inspect_ai.tool import ToolCall

    from prefill_awareness_audit.probes.attribution import attribution_probe

    tc = ToolCall(id="call_123", function="bash", arguments={"cmd": "ls"})
    probe = attribution_probe()
    state = _make_state(completion="Let me check.", tool_calls=[tc])
    generate = _make_generate("ANSWER: self")

    result = await probe(state, generate)

    # Tool stub should be inserted before the user question
    assert len(result.messages) >= 2
    assert result.messages[0].role == "tool"
    assert result.messages[0].tool_call_id == "call_123"
    assert result.messages[1].role == "user"


@pytest.mark.asyncio
async def test_confidence_probe_stores_value():
    from prefill_awareness_audit.probes.confidence import confidence_probe

    probe = confidence_probe()
    state = _make_state(completion="previous response")
    state.metadata = {}
    generate = _make_generate("42")

    result = await probe(state, generate)

    assert result.metadata["prefill_confidence"]["value"] == 42
    assert result.metadata["prefill_confidence"]["raw_response"] == "42"


@pytest.mark.asyncio
async def test_diagnostic_probe_stores_parsed_json():
    from prefill_awareness_audit.probes.diagnostic import diagnostic_probe

    probe = diagnostic_probe()
    state = _make_state(completion="previous response")
    state.metadata = {}
    generate = _make_generate(
        '{"turn": 3, "tags": ["style_mismatch"], "reason": "odd style"}'
    )

    result = await probe(state, generate)

    assert result.metadata["diagnostic"]["turn"] == 3
    assert result.metadata["diagnostic"]["tags"] == ["style_mismatch"]
    assert result.metadata["diagnostic"]["parse_failed"] is False
