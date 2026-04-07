"""Integration tests — end-to-end audit_task and audit_tasks composition."""

from __future__ import annotations

import pytest

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from prefill_awareness_audit import (
    AuditProfile,
    Condition,
    RewritePolicy,
    audit_task,
    audit_tasks,
)
from prefill_awareness_audit.types import AuditTarget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Conditions that don't require model calls (no rewrite).
ALL_CONDITIONS = [
    Condition.BASELINE,
    Condition.PROBE_ONLY,
    Condition.LOCAL_DETERMINISTIC,
]


def _make_sample(
    messages: list[tuple[str, str]] | None = None,
    sample_id: str = "test-001",
) -> Sample:
    if messages is None:
        messages = [
            ("user", "Hello"),
            ("assistant", "Hi there!  \n"),  # trailing whitespace for cleanup test
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
    rewrite_policy: RewritePolicy | None = None,
) -> AuditProfile:
    return AuditProfile(
        name="test-profile",
        main_score_fields=["score"],
        target_span_selector=lambda s: AuditTarget(
            sample_id=s.id or "", message_indices=[1], target_kind="single_turn"
        ),
        allowed_conditions=allowed_conditions or list(ALL_CONDITIONS),
        rewrite_policy=rewrite_policy or RewritePolicy(),
    )


def _mock_scorer() -> Scorer:
    @scorer(metrics=[])
    def _inner() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return Score(value={"score": 1}, answer="mock")

        return score

    return _inner()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_no_probes_no_intervention():
    ds = _make_dataset()
    solver = [lambda s, g: s]  # placeholder solver
    profile = _make_profile()

    task = await audit_task(ds, solver, _mock_scorer(), Condition.BASELINE, profile)

    # Solver chain has only the original solver (no probes)
    assert len(task.solver) == 1
    # Dataset unchanged
    assert len(list(task.dataset)) == 5
    # Metadata
    assert task.metadata["condition"] == "BASELINE"
    assert task.metadata["profile"] == "test-profile"


@pytest.mark.asyncio
async def test_probe_only_appends_probes():
    ds = _make_dataset()
    solver = [lambda s, g: s]
    profile = _make_profile()

    task = await audit_task(ds, solver, _mock_scorer(), Condition.PROBE_ONLY, profile)

    # Original solver + 3 probes
    assert len(task.solver) == 4
    # Dataset unchanged (no intervention for PROBE_ONLY)
    assert len(list(task.dataset)) == 5
    assert task.metadata["condition"] == "PROBE_ONLY"


@pytest.mark.asyncio
async def test_local_deterministic_applies_intervention_and_probes():
    ds = _make_dataset()
    solver = [lambda s, g: s]
    profile = _make_profile()

    task = await audit_task(
        ds, solver, _mock_scorer(), Condition.LOCAL_DETERMINISTIC, profile
    )

    # Probes appended
    assert len(task.solver) == 4

    # Dataset was modified: trailing whitespace should be cleaned on index 1
    modified_samples = list(task.dataset)
    assert len(modified_samples) == 5
    for sample in modified_samples:
        msgs = sample.input
        # Index 1 is the assistant message that had trailing whitespace
        assert "  " not in msgs[1].content, "Trailing whitespace not cleaned"

    assert task.metadata["condition"] == "LOCAL_DETERMINISTIC"


@pytest.mark.asyncio
async def test_disallowed_condition_raises():
    ds = _make_dataset()
    solver = [lambda s, g: s]
    profile = _make_profile(allowed_conditions=[Condition.BASELINE])

    with pytest.raises(ValueError, match="not in profile.allowed_conditions"):
        await audit_task(ds, solver, _mock_scorer(), Condition.LOCAL_DETERMINISTIC, profile)


@pytest.mark.asyncio
async def test_rewrite_policy_blocks_disallowed_mechanism():
    ds = _make_dataset()
    solver = [lambda s, g: s]
    # Allow B-D in conditions but block it in policy
    profile = _make_profile(
        allowed_conditions=[
            Condition.BASELINE,
            Condition.PROBE_ONLY,
            Condition.LOCAL_DETERMINISTIC,
            Condition.LOCAL_REWRITE,
            Condition.BLOCK_DETERMINISTIC,
        ],
        rewrite_policy=RewritePolicy(allow_block_deterministic=False),
    )

    with pytest.raises(ValueError, match="not allowed by the rewrite policy"):
        await audit_task(
            ds, solver, _mock_scorer(), Condition.BLOCK_DETERMINISTIC, profile
        )


@pytest.mark.asyncio
async def test_metadata_populated():
    ds = _make_dataset()
    solver = [lambda s, g: s]
    profile = _make_profile()

    task = await audit_task(ds, solver, _mock_scorer(), Condition.PROBE_ONLY, profile)

    assert task.metadata["condition"] == "PROBE_ONLY"
    assert task.metadata["profile"] == "test-profile"
    rp = task.metadata["rewrite_policy"]
    assert rp["allow_local_deterministic"] is True
    assert rp["allow_local_rewrite"] is True
    assert rp["allow_block_deterministic"] is False


@pytest.mark.asyncio
async def test_audit_tasks_returns_all_conditions():
    ds = _make_dataset()
    solver = [lambda s, g: s]
    profile = _make_profile()

    tasks = await audit_tasks(ds, solver, _mock_scorer(), profile)

    assert len(tasks) == len(ALL_CONDITIONS)
    conditions_returned = {t.metadata["condition"] for t in tasks}
    conditions_expected = {c.value for c in ALL_CONDITIONS}
    assert conditions_returned == conditions_expected
