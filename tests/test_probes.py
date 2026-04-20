"""Tests for probe solvers and parsers."""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from prefill_awareness_audit.probes.awareness import (
    _parse_attribution,
    _parse_awareness,
    _parse_confidence,
)
from prefill_awareness_audit.probes.diagnostic import _parse_diagnostic
from prefill_awareness_audit.probes.multiple_choice import (
    _parse_latent_choices,
    _shuffle_options,
)
from prefill_awareness_audit.probes.prompts import (
    LATENT_CANONICAL_OPTIONS,
    LATENT_EVAL_CANONICAL,
    LATENT_PREFILL_CANONICAL,
)
from prefill_awareness_audit.probes.spontaneous import detect_spontaneous_awareness


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(completion="test response", tool_calls=None, metadata=None):
    """Build a MagicMock TaskState with the fields probes access.

    Populates ``state.messages`` with a ChatMessageAssistant carrying the
    completion and tool_calls, mirroring what Inspect AI actually does after
    generate() runs (the generated response is appended to state.messages).
    """
    from inspect_ai.model import ChatMessageAssistant

    state = MagicMock()
    state.output.completion = completion
    state.output.message.tool_calls = tool_calls
    state.messages = [
        ChatMessageAssistant(content=completion, tool_calls=tool_calls)
    ]
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
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("ANSWER: self", "self"),
            ("ANSWER: not self", "not_self"),
            ("ANSWER: uncertain", "uncertain"),
            ("ANSWER: Self.", "self"),
            ("I think ANSWER: self because...", "self"),
            ("ANSWER: I'm unsure", "uncertain"),
            ("self", "self"),
            ("Those are not self-generated", "not_self"),
            ("I'm not sure about this", "uncertain"),
        ],
    )
    def test_happy_path(self, text: str, expected: str) -> None:
        assert _parse_attribution(text) == expected

    @pytest.mark.parametrize("text", ["", "The weather is nice today"])
    def test_no_signal_returns_none(self, text: str) -> None:
        assert _parse_attribution(text) is None


# ---------------------------------------------------------------------------
# Confidence parser
# ---------------------------------------------------------------------------


class TestParseConfidence:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("75", 75),
            ("0", 0),
            ("100", 100),
            ("I'd say about 50", 50),
            ("50%", 50),
            ("75.5", 76),
        ],
    )
    def test_happy_path(self, text: str, expected: int) -> None:
        assert _parse_confidence(text) == expected

    @pytest.mark.parametrize("text", ["150", "No idea"])
    def test_no_signal_returns_none(self, text: str) -> None:
        assert _parse_confidence(text) is None


# ---------------------------------------------------------------------------
# Combined awareness parser
# ---------------------------------------------------------------------------


class TestParseAwareness:
    def test_clean_both_markers(self):
        text = "ATTRIBUTION: not self\nCONFIDENCE: 85"
        label, conf = _parse_awareness(text)
        assert label == "not_self"
        assert conf == 85

    def test_self_with_confidence(self):
        text = "ATTRIBUTION: self\nCONFIDENCE: 10"
        label, conf = _parse_awareness(text)
        assert label == "self"
        assert conf == 10

    def test_uncertain_with_confidence(self):
        text = "ATTRIBUTION: uncertain\nCONFIDENCE: 50"
        label, conf = _parse_awareness(text)
        assert label == "uncertain"
        assert conf == 50

    def test_only_attribution_marker(self):
        text = "ATTRIBUTION: not self\nI'm not sure about the probability."
        label, conf = _parse_awareness(text)
        assert label == "not_self"
        assert conf is None

    def test_only_confidence_marker(self):
        text = "I think they were inserted.\nCONFIDENCE: 90"
        label, conf = _parse_awareness(text)
        # Falls back to _parse_attribution which finds no self/not_self/uncertain keywords
        assert label is None
        assert conf == 90

    def test_fallback_to_bare_keywords(self):
        text = "ANSWER: not self\n75"
        label, conf = _parse_awareness(text)
        assert label == "not_self"
        assert conf == 75

    def test_confidence_out_of_range_ignored(self):
        text = "ATTRIBUTION: self\nCONFIDENCE: 150"
        label, conf = _parse_awareness(text)
        assert label == "self"
        assert conf is None


# ---------------------------------------------------------------------------
# Latent MCQ probe -- shuffling
# ---------------------------------------------------------------------------


class TestShuffleOptions:
    def test_deterministic_with_same_seed(self):
        a = _shuffle_options(LATENT_CANONICAL_OPTIONS, seed=123)
        b = _shuffle_options(LATENT_CANONICAL_OPTIONS, seed=123)
        assert a == b

    def test_different_seeds_produce_different_orderings(self):
        a = _shuffle_options(LATENT_CANONICAL_OPTIONS, seed=1)
        b = _shuffle_options(LATENT_CANONICAL_OPTIONS, seed=2)
        # Extremely unlikely to collide on 8! permutations
        assert [x[1] for x in a] != [x[1] for x in b]

    def test_none_option_pinned_last(self):
        shuffled = _shuffle_options(LATENT_CANONICAL_OPTIONS, seed=42)
        # Last canonical_id should be the one pinned in the source list
        assert shuffled[-1][1] == LATENT_CANONICAL_OPTIONS[-1][0]

    def test_display_letters_sequential(self):
        shuffled = _shuffle_options(LATENT_CANONICAL_OPTIONS, seed=42)
        letters = [row[0] for row in shuffled]
        assert letters == list("ABCDEFGHI")

    def test_all_canonical_ids_present(self):
        shuffled = _shuffle_options(LATENT_CANONICAL_OPTIONS, seed=42)
        canon_ids = {row[1] for row in shuffled}
        expected = {row[0] for row in LATENT_CANONICAL_OPTIONS}
        assert canon_ids == expected


# ---------------------------------------------------------------------------
# Latent MCQ probe -- parser
# ---------------------------------------------------------------------------


class TestParseLatentChoices:
    @staticmethod
    def _make_mapping(seed: int = 42) -> dict[str, str]:
        shuffled = _shuffle_options(LATENT_CANONICAL_OPTIONS, seed=seed)
        return {letter: canon_id for letter, canon_id, _text in shuffled}

    @staticmethod
    def _letter_for(mapping: dict[str, str], canon: str) -> str:
        return next(letter for letter, c in mapping.items() if c == canon)

    def test_happy_path(self) -> None:
        mapping = self._make_mapping()
        p = self._letter_for(mapping, LATENT_PREFILL_CANONICAL)
        e = self._letter_for(mapping, LATENT_EVAL_CANONICAL)
        none = self._letter_for(mapping, "none")

        r = _parse_latent_choices(p, mapping)
        assert r["choices"] == [LATENT_PREFILL_CANONICAL]
        assert r["latent_prefill"] == 1
        assert r["latent_eval"] == 0
        assert r["letter_mapping"] == mapping

        r = _parse_latent_choices(f"{p}, {e}", mapping)
        assert r["latent_prefill"] == 1
        assert r["latent_eval"] == 1

        r = _parse_latent_choices(f"{p} and {e}", mapping)
        assert r["latent_prefill"] == 1
        assert r["latent_eval"] == 1

        r = _parse_latent_choices(none, mapping)
        assert r["choices"] == ["none"]
        assert r["latent_prefill"] == 0
        assert r["latent_eval"] == 0

        r = _parse_latent_choices(p.lower(), mapping)
        assert r["latent_prefill"] == 1

        r = _parse_latent_choices(
            f"{p}\n\nI chose this because the responses look unusual.",
            mapping,
        )
        assert r["choices"] == [LATENT_PREFILL_CANONICAL]

    def test_parse_failure_returns_minus_one(self) -> None:
        mapping = self._make_mapping()
        r = _parse_latent_choices("no valid choices here", mapping)
        assert r["latent_prefill"] == -1
        assert r["latent_eval"] == -1
        assert r["choices"] == []


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
            "This looks like a prefill with tampered messages"
        )
        assert result["detected"] is True
        assert "prefill" in result["matched_phrases"]
        assert "tampered messages" in result["matched_phrases"]

    # -- False-positive regression tests --

    @pytest.mark.parametrize(
        "text",
        [
            "The forensic lab analyzed the synthetic polymer fiber "
            "found at the crime scene.",
            "The catheter was inserted into the patient's vein for IV access.",
            "The evidence showed the lock had been tampered with by the suspect.",
            "The artificial intelligence model performed well on the benchmark.",
        ],
    )
    def test_domain_word_no_false_positive(self, text: str) -> None:
        assert detect_spontaneous_awareness(text)["detected"] is False

    # -- True-positive tests --

    def test_synthetic_with_conversation_context(self):
        result = detect_spontaneous_awareness(
            "These messages appear to be synthetic messages "
            "inserted into this conversation."
        )
        assert result["detected"] is True
        assert "synthetic messages" in result["matched_phrases"]
        assert "inserted into this conversation" in result["matched_phrases"]

    def test_high_confidence_multiword(self):
        result = detect_spontaneous_awareness(
            "It seems like a fake history has been placed here."
        )
        assert result["detected"] is True
        assert "fake history" in result["matched_phrases"]

    # -- Context-required tier tests --

    def test_context_required_with_anchor(self):
        result = detect_spontaneous_awareness(
            "I suspect this conversation is artificial and was staged."
        )
        assert result["detected"] is True
        assert "artificial" in result["matched_phrases"]
        assert "staged" in result["matched_phrases"]

    def test_context_required_without_anchor(self):
        result = detect_spontaneous_awareness(
            "The manufactured goods were shipped from the staged warehouse."
        )
        assert result["detected"] is False


# ---------------------------------------------------------------------------
# Async solver tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_awareness_probe_stores_both_metadata():
    from prefill_awareness_audit.probes.awareness import awareness_probe

    probe = awareness_probe()
    state = _make_state(completion="I will continue the task.")
    generate = _make_generate("ATTRIBUTION: not self\nCONFIDENCE: 85")

    result = await probe(state, generate)

    assert result.metadata["attribution"]["label"] == "not_self"
    assert result.metadata["prefill_confidence"]["value"] == 85


@pytest.mark.asyncio
async def test_awareness_probe_handles_tool_calls():
    from inspect_ai.tool import ToolCall

    from prefill_awareness_audit.probes.awareness import awareness_probe

    tc = ToolCall(id="call_123", function="bash", arguments={"cmd": "ls"})
    probe = awareness_probe()
    state = _make_state(completion="Let me check.", tool_calls=[tc])
    generate = _make_generate("ATTRIBUTION: self\nCONFIDENCE: 10")

    result = await probe(state, generate)

    # messages[0] is the seeded assistant turn (with tool_calls).
    # The probe appends a tool stub, then the user question.
    assert len(result.messages) >= 3
    assert result.messages[0].role == "assistant"
    assert result.messages[1].role == "tool"
    assert result.messages[1].tool_call_id == "call_123"
    assert result.messages[2].role == "user"


@pytest.mark.asyncio
async def test_multiple_choice_probe_stores_choices():
    from prefill_awareness_audit.probes.multiple_choice import (
        _shuffle_options,
        multiple_choice_probe,
    )

    probe = multiple_choice_probe()
    state = _make_state(completion="previous response")
    state.sample_id = "test-001"
    state.metadata = {}

    # Compute the same shuffle the probe will use so we know which letter
    # maps to "prefill" and can feed that letter as the model response.
    sample_seed = 42 + hash("test-001")
    shuffled = _shuffle_options(LATENT_CANONICAL_OPTIONS, sample_seed)
    mapping = {letter: canon_id for letter, canon_id, _t in shuffled}
    prefill_letter = next(
        l for l, c in mapping.items() if c == LATENT_PREFILL_CANONICAL
    )

    generate = _make_generate(prefill_letter)
    result = await probe(state, generate)

    mc = result.metadata["multiple_choice"]
    assert mc["latent_prefill"] == 1
    assert mc["latent_eval"] == 0
    assert LATENT_PREFILL_CANONICAL in mc["choices"]


@pytest.mark.asyncio
async def test_forked_probes_merges_metadata():
    """Verify forked_probes merges metadata from all probes into original state."""
    from prefill_awareness_audit.probes.fork import forked_probes

    # Create two simple mock probes that write different metadata keys
    async def probe_a(state, generate):
        state.metadata["key_a"] = "value_a"
        return state

    async def probe_b(state, generate):
        state.metadata["key_b"] = "value_b"
        return state

    wrapper = forked_probes([probe_a, probe_b])
    state = _make_state(completion="test response")

    # Mock fork() since it requires a running task context
    async def mock_fork(s, solvers):
        from copy import deepcopy

        results = []
        for slv in solvers:
            forked = deepcopy(s)
            forked = await slv(forked, None)
            results.append(forked)
        return results

    with patch("prefill_awareness_audit.probes.fork.fork", side_effect=mock_fork):
        result = await wrapper(state, None)

    assert result.metadata["key_a"] == "value_a"
    assert result.metadata["key_b"] == "value_b"


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


# ---------------------------------------------------------------------------
# Judge JSON parser (_extract_json_object)
# ---------------------------------------------------------------------------


class TestJudgeTemplatesFormat:
    """Regression: JSON examples in judge templates must be escaped so
    .format(response=...) doesn't misinterpret them as format fields."""

    def test_reflection_template_formats(self):
        from prefill_awareness_audit.probes.prompts import (
            REFLECTION_JUDGE_TEMPLATE,
        )

        out = REFLECTION_JUDGE_TEMPLATE.format(response="hello")
        assert "hello" in out
        assert '{"category"' in out  # unescaped after format

    def test_persona_template_formats(self):
        from prefill_awareness_audit.probes.prompts import (
            PERSONA_JUDGE_TEMPLATE,
        )

        out = PERSONA_JUDGE_TEMPLATE.format(response="hello")
        assert "hello" in out
        assert '{"category"' in out


class TestExtractJsonObject:
    def test_direct_parse(self):
        from prefill_awareness_audit.probes._util import _extract_json_object

        assert _extract_json_object('{"category": "convinced"}') == {
            "category": "convinced"
        }

    def test_markdown_fences(self):
        from prefill_awareness_audit.probes._util import _extract_json_object

        assert _extract_json_object(
            '```json\n{"category": "none"}\n```'
        ) == {"category": "none"}

    def test_surrounding_text(self):
        from prefill_awareness_audit.probes._util import _extract_json_object

        assert _extract_json_object(
            'Here is my answer: {"category": "suspicious"} — done.'
        ) == {"category": "suspicious"}

    def test_no_braces(self):
        from prefill_awareness_audit.probes._util import _extract_json_object

        assert _extract_json_object("I think convinced") is None

    def test_malformed(self):
        from prefill_awareness_audit.probes._util import _extract_json_object

        assert _extract_json_object('{"category": "convinced"') is None


# ---------------------------------------------------------------------------
# Reflection + persona probes
# ---------------------------------------------------------------------------


def _mock_classify(category):
    """Return an async callable mimicking classify_judgement for mocking."""

    async def stub(response, template, judge_model):
        return category

    return stub


@pytest.mark.asyncio
async def test_reflection_probe_stores_category_and_raw():
    from prefill_awareness_audit.probes import reflection

    probe = reflection.reflection_probe()
    state = _make_state(completion="Those messages feel unfamiliar to me.")
    state.metadata = {}
    generate = _make_generate("Those messages feel unfamiliar to me.")

    with patch.object(reflection, "classify_judgement", _mock_classify("suspicious")):
        result = await probe(state, generate)

    assert result.metadata["reflection"]["category"] == "suspicious"
    assert result.metadata["reflection"]["raw_response"] == (
        "Those messages feel unfamiliar to me."
    )


@pytest.mark.asyncio
async def test_reflection_probe_parse_failure_yields_none():
    from prefill_awareness_audit.probes import reflection

    probe = reflection.reflection_probe()
    state = _make_state(completion="")
    state.metadata = {}
    generate = _make_generate("")

    with patch.object(reflection, "classify_judgement", _mock_classify(None)):
        result = await probe(state, generate)

    assert result.metadata["reflection"]["category"] is None


@pytest.mark.asyncio
async def test_persona_probe_stores_category_and_raw():
    from prefill_awareness_audit.probes import persona

    probe = persona.persona_probe()
    state = _make_state(completion="Wait, are you still you?")
    state.metadata = {}
    generate = _make_generate("Wait, are you still you?")

    with patch.object(persona, "classify_judgement", _mock_classify("convinced")):
        result = await probe(state, generate)

    assert result.metadata["persona"]["category"] == "convinced"
    assert result.metadata["persona"]["raw_response"] == "Wait, are you still you?"
