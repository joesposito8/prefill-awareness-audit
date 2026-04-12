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
    RewritePolicy,
    make_audit_task,
)
from prefill_awareness_audit.data import DEFAULT_PROFILE, all_assistant_turns
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
            Condition.BASELINE,
            Condition.PROBE_ONLY,
            Condition.LOCAL_DETERMINISTIC,
        ],
        rewrite_policy=RewritePolicy(),
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
        data=ds, condition=Condition.BASELINE, profile=profile, scorer=_mock_scorer()
    )
    assert task.metadata["condition"] == "BASELINE"


def test_make_audit_task_condition_string() -> None:
    """Accepts condition as a string."""
    ds = _make_dataset()
    profile = _make_profile()
    task = make_audit_task(
        data=ds, condition="BASELINE", profile=profile, scorer=_mock_scorer()
    )
    assert task.metadata["condition"] == "BASELINE"


def test_make_audit_task_default_profile() -> None:
    """Uses DEFAULT_PROFILE when profile=None."""
    ds = _make_dataset()
    task = make_audit_task(data=ds, condition="PROBE_ONLY")
    assert task.metadata["profile"] == "default"


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
    profile = _make_profile(allowed_conditions=[Condition.BASELINE, Condition.PROBE_ONLY])
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


def test_make_audit_task_baseline_no_probes() -> None:
    """BASELINE condition does not append probes."""
    ds = _make_dataset()
    profile = _make_profile()
    task = make_audit_task(
        data=ds,
        condition="BASELINE",
        profile=profile,
        solver=[lambda s, g: s],
        scorer=_mock_scorer(),
    )
    # Only the original solver, no probes
    assert len(task.solver) == 1


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
