"""Tests for intervention dataset transformers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser

from prefill_awareness_audit.interventions.deterministic import (
    apply_deterministic_cleanup,
    deterministic_intervention,
    fix_serialization_artifacts,
    normalize_markdown,
    normalize_thinking_tags,
    normalize_whitespace,
    strip_provenance_markers,
    strip_provenance_only,
)
from prefill_awareness_audit.interventions.validator import (
    _content_to_text,
    validate_intervention,
)
from prefill_awareness_audit.types import (
    AuditTarget,
    Condition,
    InterventionConstraints,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    messages: list[tuple[str, str]] | None = None,
    sample_id: str = "test-001",
) -> Sample:
    """Build a Sample with list[ChatMessage] input.

    Args:
        messages: List of (role, content) tuples. Defaults to a typical
            user/assistant/user/assistant conversation.
        sample_id: The sample ID.
    """
    if messages is None:
        messages = [
            ("user", "Hello"),
            ("assistant", "Hi there! How can I help?"),
            ("user", "Tell me about Python."),
            ("assistant", "Python is a programming language."),
        ]

    _role_map = {
        "user": ChatMessageUser,
        "assistant": ChatMessageAssistant,
        "tool": ChatMessageTool,
    }

    chat_messages = []
    for role, content in messages:
        cls = _role_map[role]
        if role == "tool":
            chat_messages.append(
                cls(content=content, tool_call_id="tc_001", function="test_fn")
            )
        else:
            chat_messages.append(cls(content=content))

    return Sample(input=chat_messages, target="test", id=sample_id)


def _default_constraints() -> InterventionConstraints:
    return InterventionConstraints()


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------


class TestNormalizeWhitespace:
    def test_trailing_spaces_stripped(self):
        assert normalize_whitespace("hello   \nworld  ") == "hello\nworld"

    def test_crlf_normalized(self):
        assert normalize_whitespace("hello\r\nworld\rtest") == "hello\nworld\ntest"

    def test_multiple_blank_lines_collapsed(self):
        assert normalize_whitespace("a\n\n\n\nb") == "a\n\nb"

    def test_single_blank_line_preserved(self):
        assert normalize_whitespace("a\n\nb") == "a\n\nb"

    def test_tabs_trailing_stripped(self):
        assert normalize_whitespace("hello\t\nworld") == "hello\nworld"


# ---------------------------------------------------------------------------
# Markdown normalization
# ---------------------------------------------------------------------------


class TestNormalizeMarkdown:
    def test_header_spacing_normalized(self):
        assert normalize_markdown("##No space") == "## No space"

    def test_header_extra_spaces_normalized(self):
        assert normalize_markdown("#   Too many") == "# Too many"

    def test_star_list_markers_normalized(self):
        assert normalize_markdown("* item one\n* item two") == "- item one\n- item two"

    def test_plus_list_markers_normalized(self):
        assert normalize_markdown("+ item one") == "- item one"

    def test_existing_dash_markers_unchanged(self):
        assert normalize_markdown("- item one") == "- item one"

    def test_tilde_code_fences_normalized(self):
        assert normalize_markdown("~~~python\ncode\n~~~") == "```python\ncode\n```"

    def test_nested_list_markers(self):
        assert normalize_markdown("  * nested") == "  - nested"


# ---------------------------------------------------------------------------
# Thinking tag normalization
# ---------------------------------------------------------------------------


class TestNormalizeThinkingTags:
    def test_uppercase_thinking(self):
        result = normalize_thinking_tags("<THINKING>text</THINKING>")
        assert result == "<thinking>text</thinking>"

    def test_ant_thinking(self):
        result = normalize_thinking_tags("<antThinking>text</antThinking>")
        assert result == "<thinking>text</thinking>"

    def test_attributes_stripped(self):
        result = normalize_thinking_tags('<thinking type="internal">text</thinking>')
        assert result == "<thinking>text</thinking>"

    def test_normal_text_unchanged(self):
        text = "This is regular text without thinking tags."
        assert normalize_thinking_tags(text) == text


# ---------------------------------------------------------------------------
# Provenance marker stripping
# ---------------------------------------------------------------------------


class TestStripProvenanceMarkers:
    def test_resp_id_stripped(self):
        text = "Response resp_abc123def456789012 was generated"
        result = strip_provenance_markers(text)
        assert "resp_" not in result
        assert "was generated" in result

    def test_call_id_stripped(self):
        text = "call_1234567890abcdef1234567890abcdef"
        assert strip_provenance_markers(text).strip() == ""

    def test_msg_id_stripped(self):
        text = "msg_aabbccdd11223344"
        assert strip_provenance_markers(text).strip() == ""

    def test_iso_timestamp_stripped(self):
        text = "Created at 2024-01-15T10:30:00Z in production"
        result = strip_provenance_markers(text)
        assert "2024" not in result
        assert "in production" in result

    def test_iso_timestamp_with_offset(self):
        text = "Time: 2024-03-20T14:30:00+05:30 end"
        result = strip_provenance_markers(text)
        assert "2024" not in result
        assert "end" in result

    def test_model_metadata_stripped(self):
        text = 'Config: "model": "gpt-4-turbo" was used'
        result = strip_provenance_markers(text)
        assert '"model"' not in result

    def test_normal_text_preserved(self):
        text = "This is a normal message without any markers."
        assert strip_provenance_markers(text) == text

    def test_short_hex_not_stripped(self):
        # IDs shorter than 16 hex chars should not be stripped
        text = "resp_abc123"
        assert strip_provenance_markers(text) == text


# ---------------------------------------------------------------------------
# Serialization artifact fixing
# ---------------------------------------------------------------------------


class TestFixSerializationArtifacts:
    def test_double_escaped_newlines(self):
        assert fix_serialization_artifacts("line1\\\\nline2") == "line1\nline2"

    def test_escaped_forward_slash(self):
        assert fix_serialization_artifacts("path\\/to\\/file") == "path/to/file"

    def test_unicode_escapes(self):
        assert fix_serialization_artifacts("caf\\u00e9") == "caf\u00e9"

    def test_normal_text_unchanged(self):
        text = "Normal text with actual\nnewlines"
        assert fix_serialization_artifacts(text) == text


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------


class TestApplyDeterministicCleanup:
    def test_all_operations_applied(self):
        text = (
            "##Header  \r\n"
            "* item\n\n\n\n"
            "<THINKING>thought</THINKING>\n"
            "resp_aabbccdd00112233 metadata\n"
            "path\\/to\\/file"
        )
        result = apply_deterministic_cleanup(text)
        assert "## Header" in result
        assert "- item" in result
        assert "\r" not in result
        assert "\n\n\n" not in result
        assert "<thinking>" in result
        assert "</thinking>" in result
        assert "resp_" not in result
        assert "\\/" not in result


# ---------------------------------------------------------------------------
# Provenance-only stripping
# ---------------------------------------------------------------------------


class TestStripProvenanceOnly:
    def test_strips_provenance_preserves_formatting(self):
        text = "* item resp_aabbccdd00112233\n##Header"
        result = strip_provenance_only(text)
        # Provenance stripped but formatting not normalized
        assert "resp_" not in result
        assert "* item" in result  # list marker NOT normalized
        assert "##Header" in result  # header NOT normalized


# ---------------------------------------------------------------------------
# Deterministic intervention (Sample-level)
# ---------------------------------------------------------------------------


class TestDeterministicIntervention:
    def test_modifies_only_target_indices(self):
        sample = _make_sample([
            ("user", "Hello"),
            ("assistant", "Hi   \nthere"),
            ("user", "Question"),
            ("assistant", "Answer   \nhere"),
        ])
        result = deterministic_intervention(sample, [1], _default_constraints())
        # Index 1 should be cleaned
        assert result.input[1].text == "Hi\nthere"
        # Index 3 should be untouched
        assert result.input[3].text == "Answer   \nhere"

    def test_skips_protected_roles(self):
        sample = _make_sample([
            ("user", "Hello   "),
            ("assistant", "Hi   "),
        ])
        # Even if user index is in indices, it should be skipped
        result = deterministic_intervention(sample, [0, 1], _default_constraints())
        assert result.input[0].text == "Hello   "  # user: protected
        assert result.input[1].text == "Hi"  # assistant: cleaned

    def test_original_sample_unchanged(self):
        sample = _make_sample([
            ("user", "Hello"),
            ("assistant", "Hi   \nthere"),
        ])
        original_text = sample.input[1].text
        deterministic_intervention(sample, [1], _default_constraints())
        assert sample.input[1].text == original_text

    def test_out_of_range_index_ignored(self):
        sample = _make_sample([
            ("user", "Hello"),
            ("assistant", "Hi"),
        ])
        # Should not raise
        result = deterministic_intervention(sample, [5], _default_constraints())
        assert len(result.input) == 2

    def test_provenance_only_on_protected(self):
        sample = _make_sample([
            ("user", "Hello resp_aabbccdd00112233"),
            ("assistant", "Hi   \nthere"),
        ])
        result = deterministic_intervention(
            sample,
            [1],
            _default_constraints(),
            provenance_only_indices=[0],
        )
        # User message: provenance stripped but whitespace preserved
        assert "resp_" not in result.input[0].text
        assert "Hello" in result.input[0].text
        # Assistant: full cleanup
        assert result.input[1].text == "Hi\nthere"

    def test_str_input_returns_unmodified(self):
        sample = Sample(input="plain string", target="test", id="str-001")
        result = deterministic_intervention(sample, [0], _default_constraints())
        assert result.input == "plain string"


# ---------------------------------------------------------------------------
# Validator: _content_to_text
# ---------------------------------------------------------------------------


class TestContentToText:
    def test_str_content(self):
        assert _content_to_text("hello") == "hello"

    def test_list_content(self):
        from inspect_ai._util.content import ContentText

        content = [ContentText(text="part1"), ContentText(text="part2")]
        assert _content_to_text(content) == "part1\npart2"

    def test_empty_list(self):
        assert _content_to_text([]) == ""


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class TestValidator:
    def test_valid_intervention_passes(self):
        original = _make_sample()
        modified = original.model_copy(deep=True)
        modified.input[1].text = "Modified assistant message"
        passed, violations = validate_intervention(
            original, modified, [1], _default_constraints()
        )
        assert passed
        assert violations == []

    @pytest.mark.parametrize(
        "mutate, keyword",
        [
            (lambda m: m.input.append(ChatMessageAssistant(content="extra")), "count"),
            (lambda m: m.input.__setitem__(1, ChatMessageUser(content="wrong role")), "role"),
            (lambda m: setattr(m.input[0], "text", "Modified user message"), "protected"),
            (lambda m: setattr(m.input[3], "text", "Modified non-target"), "non-target"),
        ],
        ids=["message_count", "role_order", "protected_role", "non_target"],
    )
    def test_validation_failures(self, mutate, keyword):
        original = _make_sample()
        modified = original.model_copy(deep=True)
        mutate(modified)
        passed, violations = validate_intervention(
            original, modified, [1], _default_constraints()
        )
        assert not passed
        assert any(keyword in v.lower() for v in violations)

    def test_tool_message_modified_fails(self):
        original = _make_sample([
            ("user", "Hello"),
            ("assistant", "Let me check"),
            ("tool", "Tool result data"),
            ("assistant", "The result is..."),
        ])
        modified = original.model_copy(deep=True)
        modified.input[2].text = "Modified tool output"
        passed, violations = validate_intervention(
            original, modified, [1], _default_constraints()
        )
        assert not passed
        assert any("tool" in v.lower() for v in violations)

    def test_empty_message_fails(self):
        original = _make_sample()
        modified = original.model_copy(deep=True)
        modified.input[1].text = "   "  # whitespace-only = empty
        passed, violations = validate_intervention(
            original, modified, [1], _default_constraints()
        )
        assert not passed
        assert any("empty" in v.lower() for v in violations)

    def test_multiple_violations_reported(self):
        original = _make_sample()
        modified = original.model_copy(deep=True)
        modified.input[0].text = "Modified user"  # protected
        modified.input[1].text = ""  # empty
        modified.input[3].text = "Changed non-target"  # non-target
        passed, violations = validate_intervention(
            original, modified, [1], _default_constraints()
        )
        assert not passed
        assert len(violations) >= 3


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------


class TestResolveScope:
    def _make_profile(self, target_indices=None):
        from prefill_awareness_audit.types import AuditProfile

        target_indices = target_indices or [1]
        return AuditProfile(
            name="test",
            main_score_fields=["score"],
            target_span_selector=lambda s: AuditTarget(
                sample_id=str(s.id),
                message_indices=target_indices,
                target_kind="single_turn",
            ),
            allowed_conditions=list(Condition),
            intervention_constraints=_default_constraints(),
        )

    def test_probe_only_returns_empty(self):
        from prefill_awareness_audit.interventions import _resolve_scope

        sample = _make_sample()
        profile = self._make_profile()
        indices, prov = _resolve_scope(Condition.PROBE_ONLY, sample, profile)
        assert indices == []
        assert prov == []

    def test_local_returns_target_indices(self):
        from prefill_awareness_audit.interventions import _resolve_scope

        sample = _make_sample()
        profile = self._make_profile(target_indices=[1, 3])
        indices, prov = _resolve_scope(
            Condition.LOCAL_DETERMINISTIC, sample, profile
        )
        assert indices == [1, 3]
        assert prov == []

    def test_block_returns_all_assistant_before_last(self):
        from prefill_awareness_audit.interventions import _resolve_scope

        sample = _make_sample([
            ("user", "Hi"),
            ("assistant", "Hello"),
            ("user", "More"),
            ("assistant", "Sure"),
            ("user", "Last"),
            ("assistant", "Final"),  # continuation point (last message)
        ])
        profile = self._make_profile()
        indices, prov = _resolve_scope(
            Condition.BLOCK_DETERMINISTIC, sample, profile
        )
        # Assistant messages at indices 1 and 3, not 5 (last)
        assert indices == [1, 3]
        assert prov == []

    def test_global_returns_non_protected_and_provenance_only(self):
        from prefill_awareness_audit.interventions import _resolve_scope

        sample = _make_sample([
            ("user", "Hi"),
            ("assistant", "Hello"),
            ("user", "More"),
            ("assistant", "Sure"),
        ])
        profile = self._make_profile()
        indices, prov = _resolve_scope(
            Condition.GLOBAL_DETERMINISTIC, sample, profile
        )
        # Assistants (1, 3) get full cleanup; users (0, 2) get provenance only
        assert indices == [1, 3]
        assert prov == [0, 2]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestApplyIntervention:
    def _make_profile(
        self,
        allowed_conditions=None,
        constraints=None,
        semantic_guard=None,
    ):
        from prefill_awareness_audit.types import AuditProfile

        return AuditProfile(
            name="test",
            main_score_fields=["score"],
            target_span_selector=lambda s: AuditTarget(
                sample_id=str(s.id),
                message_indices=[1],
                target_kind="single_turn",
            ),
            allowed_conditions=allowed_conditions or list(Condition),
            intervention_constraints=constraints or _default_constraints(),
            semantic_guard=semantic_guard,
        )

    @pytest.mark.asyncio
    async def test_disallowed_condition_raises(self):
        from prefill_awareness_audit.interventions import apply_intervention

        with pytest.raises(ValueError, match="not in profile.allowed_conditions"):
            await apply_intervention(
                [_make_sample()],
                Condition.LOCAL_DETERMINISTIC,
                self._make_profile(allowed_conditions=[Condition.PROBE_ONLY]),
            )

    @pytest.mark.asyncio
    async def test_deterministic_dispatches_correctly(self):
        from prefill_awareness_audit.interventions import apply_intervention

        sample = _make_sample([
            ("user", "Hello"),
            ("assistant", "Hi   \nthere"),
            ("user", "Question"),
            ("assistant", "Answer"),
        ])
        result = await apply_intervention(
            [sample], Condition.LOCAL_DETERMINISTIC, self._make_profile()
        )
        assert len(result) == 1
        # Index 1 should have trailing spaces cleaned
        assert result[0].input[1].text == "Hi\nthere"

    @pytest.mark.asyncio
    async def test_failed_validation_excludes_sample(self):
        from prefill_awareness_audit.interventions import apply_intervention

        # Semantic guard that always fails
        def always_fail(orig, mod):
            return ["Semantic check failed"]

        result = await apply_intervention(
            [_make_sample()],
            Condition.LOCAL_DETERMINISTIC,
            self._make_profile(semantic_guard=always_fail),
        )
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_semantic_guard_passes(self):
        from prefill_awareness_audit.interventions import apply_intervention

        def always_pass(orig, mod):
            return []

        result = await apply_intervention(
            [_make_sample()],
            Condition.LOCAL_DETERMINISTIC,
            self._make_profile(semantic_guard=always_pass),
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Rewrite (mocked model)
# ---------------------------------------------------------------------------


class TestRewriteIntervention:
    def _mock_model(self, responses=None):
        """Build a mock Model whose generate() returns canned responses."""
        model = AsyncMock()
        if responses is None:
            responses = ["Rewritten text"]

        call_count = {"n": 0}

        async def mock_generate(messages, **kwargs):
            idx = min(call_count["n"], len(responses) - 1)
            call_count["n"] += 1
            output = MagicMock()
            output.completion = responses[idx]
            return output

        model.generate = mock_generate
        return model

    @pytest.mark.asyncio
    async def test_rewrite_calls_model(self, tmp_path):
        from prefill_awareness_audit.interventions.rewrite import rewrite_intervention

        sample = _make_sample([
            ("user", "Hello"),
            ("assistant", "Context message for style"),
            ("user", "Question"),
            ("assistant", "Target to rewrite"),
        ])
        mock = self._mock_model([
            # First call: style card
            "Formality: conversational\nSentence structure: short",
            # Second call: rewrite
            "Rewritten target message",
        ])

        with patch(
            "prefill_awareness_audit.interventions.rewrite.get_model",
            return_value=mock,
        ):
            result = await rewrite_intervention(
                sample, [3], _default_constraints(), cache_dir=tmp_path
            )

        assert result.input[3].text == "Rewritten target message"
        # Original unchanged
        assert sample.input[3].text == "Target to rewrite"

    @pytest.mark.asyncio
    async def test_style_card_cached(self, tmp_path):
        """Second target message in same sample should reuse style card."""
        from prefill_awareness_audit.interventions.rewrite import rewrite_intervention

        sample = _make_sample([
            ("user", "Hello"),
            ("assistant", "Context message for style"),
            ("user", "Question"),
            ("assistant", "Target one"),
            ("user", "More"),
            ("assistant", "Target two"),
        ])

        call_count = {"n": 0}

        async def counting_generate(messages, **kwargs):
            call_count["n"] += 1
            output = MagicMock()
            if call_count["n"] == 1:
                output.completion = "Formality: casual"
            else:
                output.completion = f"Rewritten {call_count['n']}"
            return output

        mock = AsyncMock()
        mock.generate = counting_generate

        with patch(
            "prefill_awareness_audit.interventions.rewrite.get_model",
            return_value=mock,
        ):
            await rewrite_intervention(
                sample, [3, 5], _default_constraints(), cache_dir=tmp_path
            )

        # 1 style card + 2 rewrites = 3 total calls (not 2 style cards + 2 rewrites)
        assert call_count["n"] == 3

    @pytest.mark.asyncio
    async def test_rewrite_cache_hit(self, tmp_path):
        """Running the same rewrite twice should use the disk cache."""
        from prefill_awareness_audit.interventions.rewrite import rewrite_intervention

        sample = _make_sample([
            ("user", "Hello"),
            ("assistant", "Context"),
            ("user", "Q"),
            ("assistant", "Target"),
        ])

        call_count = {"n": 0}

        async def counting_generate(messages, **kwargs):
            call_count["n"] += 1
            output = MagicMock()
            if call_count["n"] == 1:
                output.completion = "Formality: formal"
            else:
                output.completion = "Rewritten text"
            return output

        mock = AsyncMock()
        mock.generate = counting_generate

        with patch(
            "prefill_awareness_audit.interventions.rewrite.get_model",
            return_value=mock,
        ):
            # First run: 1 style card + 1 rewrite = 2 calls
            await rewrite_intervention(
                sample, [3], _default_constraints(), cache_dir=tmp_path
            )
            calls_after_first = call_count["n"]

            # Second run: should hit cache, 0 new calls
            await rewrite_intervention(
                sample, [3], _default_constraints(), cache_dir=tmp_path
            )
            calls_after_second = call_count["n"]

        assert calls_after_first == 2
        assert calls_after_second == 2  # No new calls

    @pytest.mark.asyncio
    async def test_rewrite_retries_on_empty(self, tmp_path):
        from prefill_awareness_audit.interventions.rewrite import (
            _rewrite_single_message,
        )

        mock = self._mock_model(["", "Retry succeeded"])
        result = await _rewrite_single_message(
            "Original text", "Formality: casual", mock
        )
        assert result == "Retry succeeded"

    @pytest.mark.asyncio
    async def test_rewrite_keeps_original_on_double_failure(self, tmp_path):
        from prefill_awareness_audit.interventions.rewrite import (
            _rewrite_single_message,
        )

        mock = self._mock_model(["", ""])
        result = await _rewrite_single_message(
            "Original text", "Formality: casual", mock
        )
        assert result == "Original text"

    @pytest.mark.asyncio
    async def test_skips_protected_roles(self, tmp_path):
        from prefill_awareness_audit.interventions.rewrite import rewrite_intervention

        sample = _make_sample([
            ("user", "Hello"),
            ("assistant", "Context"),
            ("user", "Protected content"),
            ("assistant", "Target"),
        ])

        mock = self._mock_model([
            "Formality: casual",
            "Rewritten",
        ])

        with patch(
            "prefill_awareness_audit.interventions.rewrite.get_model",
            return_value=mock,
        ):
            result = await rewrite_intervention(
                sample, [2, 3], _default_constraints(), cache_dir=tmp_path
            )

        # User message at index 2 should be unchanged (protected)
        assert result.input[2].text == "Protected content"

    @pytest.mark.asyncio
    async def test_global_rewrite_not_dispatched(self):
        """GLOBAL_DETERMINISTIC is valid but GLOBAL_REWRITE doesn't exist."""
        # Condition enum doesn't have GLOBAL_REWRITE, so this is enforced
        # at the type level. Verify the enum values.
        conditions = [c.value for c in Condition]
        assert "GLOBAL_REWRITE" not in conditions
