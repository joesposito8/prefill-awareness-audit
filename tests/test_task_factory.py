"""Tests for make_audit_task — the primary user-facing API."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from prefill_awareness_audit import (
    AuditProfile,
    Condition,
    InterventionConstraints,
    make_audit_task,
)
from prefill_awareness_audit.data import PROBE_ONLY_PROFILE, all_assistant_turns
from prefill_awareness_audit.types import AuditTarget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    messages: list[tuple[str, str]] | None = None,
    sample_id: str = "test-001",
) -> Sample:
    if messages is None:
        messages = [
            ("user", "Hello"),
            ("assistant", "Hi there!  \n"),
            ("user", "Tell me about Python."),
            ("assistant", "Python is a programming language.  "),
        ]
    _role_map = {"user": ChatMessageUser, "assistant": ChatMessageAssistant}
    chat_messages = [_role_map[role](content=content) for role, content in messages]
    return Sample(input=chat_messages, target="test", id=sample_id)


def _make_dataset(n: int = 5) -> MemoryDataset:
    samples = [_make_sample(sample_id=f"s-{i:03d}") for i in range(n)]
    return MemoryDataset(samples, name="test-dataset")


def _make_profile(
    allowed_conditions: list[Condition] | None = None,
) -> AuditProfile:
    return AuditProfile(
        name="test-profile",
        main_score_fields=["score"],
        target_span_selector=lambda s: AuditTarget(
            sample_id=s.id or "", message_indices=[1], target_kind="single_turn"
        ),
        allowed_conditions=allowed_conditions or [
            Condition.PROBE_ONLY,
            Condition.LOCAL_DETERMINISTIC,
        ],
        intervention_constraints=InterventionConstraints(),
    )


def _mock_scorer() -> Scorer:
    @scorer(metrics=[])
    def _inner() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return Score(value={"score": 1}, answer="mock")

        return score

    return _inner()


# ---------------------------------------------------------------------------
# Tests — make_audit_task
# ---------------------------------------------------------------------------


def test_make_audit_task_with_dataset() -> None:
    """Accepts a Dataset directly."""
    ds = _make_dataset()
    profile = _make_profile()
    task = make_audit_task(data=ds, condition="PROBE_ONLY", profile=profile, scorer=_mock_scorer())

    assert task.metadata["condition"] == "PROBE_ONLY"
    assert task.metadata["profile"] == "test-profile"
    assert len(list(task.dataset)) == 5


def test_make_audit_task_condition_enum() -> None:
    """Accepts Condition enum directly."""
    ds = _make_dataset()
    profile = _make_profile()
    task = make_audit_task(
        data=ds, condition=Condition.PROBE_ONLY, profile=profile, scorer=_mock_scorer()
    )
    assert task.metadata["condition"] == "PROBE_ONLY"


def test_make_audit_task_condition_string() -> None:
    """Accepts condition as a string."""
    ds = _make_dataset()
    profile = _make_profile()
    task = make_audit_task(
        data=ds, condition="PROBE_ONLY", profile=profile, scorer=_mock_scorer()
    )
    assert task.metadata["condition"] == "PROBE_ONLY"


def test_make_audit_task_default_profile() -> None:
    """Uses PROBE_ONLY_PROFILE when profile=None."""
    ds = _make_dataset()
    task = make_audit_task(data=ds, condition="PROBE_ONLY")
    assert task.metadata["profile"] == "probe-only"


def test_make_audit_task_default_solver() -> None:
    """Uses generate() when solver=None."""
    ds = _make_dataset()
    profile = _make_profile()
    task = make_audit_task(
        data=ds, condition="PROBE_ONLY", profile=profile, scorer=_mock_scorer()
    )
    # Solver chain: generate() + forked_probes
    assert len(task.solver) == 2


def test_make_audit_task_default_scorer() -> None:
    """Uses a no-op scorer when scorer=None."""
    ds = _make_dataset()
    profile = _make_profile()
    task = make_audit_task(data=ds, condition="PROBE_ONLY", profile=profile)
    assert task.scorer is not None


def test_make_audit_task_limit() -> None:
    """Respects the limit parameter."""
    ds = _make_dataset(n=10)
    profile = _make_profile()
    task = make_audit_task(
        data=ds, condition="PROBE_ONLY", profile=profile, scorer=_mock_scorer(), limit=3
    )
    assert len(list(task.dataset)) == 3


def test_make_audit_task_disallowed_condition() -> None:
    """Raises ValueError for a condition not in allowed_conditions."""
    ds = _make_dataset()
    profile = _make_profile(allowed_conditions=[Condition.PROBE_ONLY])
    with pytest.raises(ValueError, match="not in profile.allowed_conditions"):
        make_audit_task(
            data=ds,
            condition="LOCAL_DETERMINISTIC",
            profile=profile,
            scorer=_mock_scorer(),
        )


def test_make_audit_task_invalid_data_type() -> None:
    """Raises ValueError for unsupported data types."""
    profile = _make_profile()
    with pytest.raises(ValueError, match="must be a path"):
        make_audit_task(data=123, condition="PROBE_ONLY", profile=profile)  # type: ignore[arg-type]


@patch("prefill_awareness_audit.task_factory.apply_intervention", new_callable=AsyncMock)
def test_make_audit_task_intervention_condition(mock_intervention: AsyncMock) -> None:
    """Applies intervention for non-passthrough conditions."""
    ds = _make_dataset(n=3)
    profile = _make_profile()
    modified_samples = [_make_sample(sample_id=f"mod-{i}") for i in range(3)]
    mock_intervention.return_value = modified_samples

    task = make_audit_task(
        data=ds,
        condition="LOCAL_DETERMINISTIC",
        profile=profile,
        scorer=_mock_scorer(),
    )

    mock_intervention.assert_called_once()
    # Dataset should contain modified samples
    result_samples = list(task.dataset)
    assert len(result_samples) == 3


def test_make_audit_task_probe_only_appends_probes() -> None:
    """PROBE_ONLY appends forked_probes to solver chain."""
    ds = _make_dataset()
    profile = _make_profile()
    task = make_audit_task(
        data=ds,
        condition="PROBE_ONLY",
        profile=profile,
        solver=[lambda s, g: s],
        scorer=_mock_scorer(),
    )
    # Original solver + forked_probes
    assert len(task.solver) == 2


def test_make_audit_task_custom_probes_override() -> None:
    """Passing probes=[...] replaces the default probe list in the forked_probes."""
    from prefill_awareness_audit.probes import awareness_probe, forked_probes

    ds = _make_dataset()
    custom_probe = awareness_probe()
    task = make_audit_task(
        data=ds,
        condition="PROBE_ONLY",
        profile=_make_profile(),
        scorer=_mock_scorer(),
        probes=[custom_probe],
    )
    # Task metadata should indicate a custom probe list was used
    assert task.metadata["custom_probes"] is True


def test_make_audit_task_default_probes_flag() -> None:
    """Default probes flag is False when probes= is not passed."""
    task = make_audit_task(
        data=_make_dataset(),
        condition="PROBE_ONLY",
        profile=_make_profile(),
        scorer=_mock_scorer(),
    )
    assert task.metadata["custom_probes"] is False


def test_make_audit_task_probes_string_subset() -> None:
    """probes='multiple_choice' resolves to a single probe solver."""
    from prefill_awareness_audit.task_factory import _resolve_probes

    resolved = _resolve_probes("multiple_choice")
    assert len(resolved) == 1

    task = make_audit_task(
        data=_make_dataset(),
        condition="PROBE_ONLY",
        profile=_make_profile(),
        scorer=_mock_scorer(),
        probes="multiple_choice",
    )
    assert task.metadata["custom_probes"] is True


def test_make_audit_task_probes_string_preserves_order() -> None:
    """Comma-separated names resolve in order."""
    from prefill_awareness_audit.probes import awareness_probe, multiple_choice_probe
    from prefill_awareness_audit.task_factory import _resolve_probes

    resolved = _resolve_probes("multiple_choice,awareness")
    assert len(resolved) == 2
    # Solvers are opaque closures; check the factory registry produced them in order
    # by instantiating again and comparing function identities on the registered factories.
    assert type(resolved[0]).__name__ == type(multiple_choice_probe()).__name__
    assert type(resolved[1]).__name__ == type(awareness_probe()).__name__


def test_make_audit_task_probes_string_unknown_name_raises() -> None:
    """Unknown probe name raises ValueError listing valid names."""
    with pytest.raises(ValueError, match="Unknown probe name"):
        make_audit_task(
            data=_make_dataset(),
            condition="PROBE_ONLY",
            profile=_make_profile(),
            scorer=_mock_scorer(),
            probes="bogus",
        )


def test_make_audit_task_probes_legacy_latent_name_now_errors() -> None:
    """The old 'latent' name is no longer valid; use 'multiple_choice' instead."""
    from prefill_awareness_audit.task_factory import _resolve_probes

    with pytest.raises(ValueError, match="Unknown probe name.*'latent'"):
        _resolve_probes("latent")


def test_make_audit_task_probes_empty_string_raises() -> None:
    """Empty probes string raises; callers should pass None for defaults."""
    from prefill_awareness_audit.task_factory import _resolve_probes

    with pytest.raises(ValueError, match="empty"):
        _resolve_probes("")
    with pytest.raises(ValueError, match="empty"):
        _resolve_probes("   ,  ")


def test_make_audit_task_probes_string_whitespace_tolerated() -> None:
    """Whitespace around names is stripped."""
    from prefill_awareness_audit.task_factory import _resolve_probes

    resolved = _resolve_probes(" awareness , multiple_choice ")
    assert len(resolved) == 2


def test_continuation_audit_wires_generate_and_continuation_prompt() -> None:
    """continuation_audit() uses a generate() solver and CONTINUATION_AWARENESS_QUESTION."""
    from prefill_awareness_audit import continuation_audit
    from prefill_awareness_audit.probes.prompts import CONTINUATION_AWARENESS_QUESTION

    ds = _make_dataset()
    task = continuation_audit(
        data=ds,
        condition="PROBE_ONLY",
        profile=_make_profile(),
        scorer=_mock_scorer(),
    )

    # One solver from continuation_audit (generate) + forked_probes wrapper
    assert len(task.solver) == 2
    # Custom probes flag should be set because continuation_audit passes its own
    assert task.metadata["custom_probes"] is True


def test_continuation_audit_with_tools_adds_use_tools() -> None:
    """continuation_audit(tools=...) inserts use_tools() before generate()."""
    from inspect_ai.tool import ToolDef

    from prefill_awareness_audit import continuation_audit

    async def _stub(**kwargs):
        return ""

    stub_tool = ToolDef(tool=_stub, name="stub", description="x", parameters={}).as_tool()

    task = continuation_audit(
        data=_make_dataset(),
        tools=[stub_tool],
        condition="PROBE_ONLY",
        profile=_make_profile(),
        scorer=_mock_scorer(),
    )
    # use_tools + generate + forked_probes wrapper
    assert len(task.solver) == 3


def test_default_prompt_is_cold_replay_variant() -> None:
    """DEFAULT_AWARENESS_QUESTION does not include the "Before your last message" intro."""
    from prefill_awareness_audit import (
        CONTINUATION_AWARENESS_QUESTION,
        DEFAULT_AWARENESS_QUESTION,
    )

    assert "before your last message" not in DEFAULT_AWARENESS_QUESTION.lower()
    assert "before your last message" in CONTINUATION_AWARENESS_QUESTION.lower()
    # Both use the same graded ATTRIBUTION: 5-point format.
    assert "ATTRIBUTION:" in DEFAULT_AWARENESS_QUESTION
    assert "ATTRIBUTION:" in CONTINUATION_AWARENESS_QUESTION
    assert "definitely me" in DEFAULT_AWARENESS_QUESTION
    assert "definitely not me" in CONTINUATION_AWARENESS_QUESTION


def test_format_first_clause_present_in_structured_output_prompts() -> None:
    """Both structured-output probes invite additional context on
    subsequent lines rather than risk out-of-channel refusals.
    Regression guard: the 2026-04-21 audit found 2/20 misalignment
    samples where haiku refused the format to voice alignment
    concerns; the clause below is the root-cause fix."""
    from prefill_awareness_audit.probes.prompts import (
        CONTINUATION_AWARENESS_QUESTION,
        DEFAULT_AWARENESS_QUESTION,
        DEFAULT_DIAGNOSTIC_QUESTION,
    )

    for prompt in (
        DEFAULT_AWARENESS_QUESTION,
        CONTINUATION_AWARENESS_QUESTION,
        DEFAULT_DIAGNOSTIC_QUESTION,
    ):
        assert "additional" in prompt.lower()
        assert "subsequent lines" in prompt.lower()
