"""Main composition point — builds an Inspect Task for a given audit condition."""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import asdict
from pathlib import Path

from inspect_ai import Task
from inspect_ai.dataset import Dataset, MemoryDataset, json_dataset
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Solver, TaskState, generate

from .interventions import apply_intervention
from .probes import awareness_probe, counterfactual_probe, diagnostic_probe, forked_probes
from .scoring import audit_scorer
from .types import AuditProfile, Condition

_NO_INTERVENTION = frozenset({Condition.BASELINE, Condition.PROBE_ONLY})


def build_audit_task(
    dataset: Dataset,
    solver: list[Solver],
    scorer: Scorer,
    condition: Condition,
    profile: AuditProfile,
) -> Task:
    """Build an Inspect Task for a given audit condition (sync).

    This is the sync core of task construction.  The caller is responsible
    for applying interventions to the dataset before calling this function
    (use :func:`apply_intervention` for intervention conditions).

    For ``BASELINE`` and ``PROBE_ONLY`` the dataset can be passed through
    unmodified.

    Args:
        dataset: The benchmark dataset (already intervention-modified if needed).
        solver: The benchmark's solver chain (should end with generate()).
        scorer: The benchmark's scorer (will be wrapped by audit_scorer).
        condition: Which audit condition to run.
        profile: Benchmark-specific audit configuration.

    Returns:
        A configured Inspect Task ready for eval().

    Raises:
        ValueError: If the condition is not in the profile's allowed_conditions.
    """
    if condition not in profile.allowed_conditions:
        raise ValueError(
            f"Condition {condition} is not in profile.allowed_conditions "
            f"({[c.value for c in profile.allowed_conditions]})"
        )

    # Append probes for all non-baseline conditions
    if condition == Condition.BASELINE:
        task_solver = list(solver)
    else:
        task_solver = list(solver) + [
            forked_probes([
                awareness_probe(),
                counterfactual_probe(),
                diagnostic_probe(),
            ])
        ]

    # Always wrap scorer for uniform output schema
    task_scorer = audit_scorer(scorer, profile.main_score_fields)

    return Task(
        dataset=dataset,
        solver=task_solver,
        scorer=task_scorer,
        metadata={
            "condition": condition.value,
            "profile": profile.name,
            "rewrite_policy": {
                k: v
                for k, v in asdict(profile.rewrite_policy).items()
                if isinstance(v, bool)
            },
        },
    )


async def audit_task(
    dataset: Dataset,
    solver: list[Solver],
    scorer: Scorer,
    condition: Condition,
    profile: AuditProfile,
) -> Task:
    """Build an Inspect Task that runs one audit condition.

    Validates the condition against the profile, applies the appropriate
    intervention to the dataset, appends probes after generate() for
    non-baseline conditions, and wraps the scorer to capture both original
    benchmark dimensions and awareness dimensions.

    This is an async convenience wrapper around :func:`build_audit_task`
    that also handles intervention application.

    Args:
        dataset: The benchmark dataset.
        solver: The benchmark's solver chain (should end with generate()).
        scorer: The benchmark's scorer (will be wrapped by audit_scorer).
        condition: Which audit condition to run.
        profile: Benchmark-specific audit configuration.

    Returns:
        A configured Inspect Task ready for eval().

    Raises:
        ValueError: If the condition is not in the profile's allowed_conditions.
    """
    # Apply intervention (modifies dataset for non-passthrough conditions)
    if condition in _NO_INTERVENTION:
        task_dataset: Dataset = dataset
    else:
        modified_samples = await apply_intervention(
            list(dataset), condition, profile
        )
        task_dataset = MemoryDataset(
            modified_samples, name=getattr(dataset, "name", None)
        )

    return build_audit_task(task_dataset, solver, scorer, condition, profile)


async def audit_tasks(
    dataset: Dataset,
    solver: list[Solver],
    scorer: Scorer,
    profile: AuditProfile,
) -> list[Task]:
    """Build one Task per admissible condition for parallel execution.

    Returns a list suitable for ``inspect eval(tasks, max_tasks=N)``.

    Args:
        dataset: The benchmark dataset.
        solver: The benchmark's solver chain.
        scorer: The benchmark's scorer.
        profile: Benchmark-specific audit configuration.

    Returns:
        List of Tasks, one per condition in profile.allowed_conditions.
    """
    return [
        await audit_task(dataset, solver, scorer, condition, profile)
        for condition in profile.allowed_conditions
    ]


def _noop_scorer() -> Scorer:
    """Create a no-op scorer that returns an empty value dict."""

    @scorer(metrics=[])
    def _inner() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return Score(value={}, answer="")

        return score

    return _inner()


def make_audit_task(
    data: str | Path | Dataset,
    condition: str | Condition,
    profile: AuditProfile | None = None,
    scorer: Scorer | None = None,
    solver: list[Solver] | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> Task:
    """Build an Inspect Task for one audit condition.

    This is the primary user-facing API.  It resolves defaults, handles
    data loading, applies interventions (including the sync/async split),
    and delegates to :func:`build_audit_task`.

    Args:
        data: Dataset, path to a JSONL file, or an Inspect Dataset.
        condition: Audit condition name or enum value.
        profile: Benchmark-specific audit configuration.  Defaults to
            ``DEFAULT_PROFILE`` (BASELINE + PROBE_ONLY only).
        scorer: Benchmark scorer.  Defaults to a no-op scorer (probes only).
        solver: Benchmark solver chain.  Defaults to ``[generate()]``.
        limit: Maximum number of samples.
        seed: Random seed (stored in task metadata).

    Returns:
        A configured Inspect Task ready for ``inspect eval``.

    Raises:
        ValueError: If the condition is not in the profile's allowed_conditions,
            or if ``data`` is an unrecognised type.
    """
    # Resolve condition
    if isinstance(condition, str):
        cond = Condition[condition]
    else:
        cond = condition

    # Resolve profile
    if profile is None:
        from .data import DEFAULT_PROFILE

        profile = DEFAULT_PROFILE

    # Resolve data to a Dataset
    if isinstance(data, (str, Path)):
        dataset: Dataset = json_dataset(str(data))
    elif isinstance(data, Dataset):
        dataset = data
    else:
        raise ValueError(
            f"data must be a path (str/Path) or an Inspect Dataset, got {type(data).__name__}"
        )

    # Apply limit
    if limit is not None:
        samples = list(dataset)[:limit]
        dataset = MemoryDataset(samples, name=getattr(dataset, "name", None))

    # Resolve solver and scorer
    task_solver = list(solver) if solver is not None else [generate()]
    task_scorer = scorer if scorer is not None else _noop_scorer()

    # Apply intervention for non-passthrough conditions
    if cond not in _NO_INTERVENTION:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            modified = pool.submit(
                asyncio.run,
                apply_intervention(list(dataset), cond, profile),
            ).result()
        dataset = MemoryDataset(modified, name=getattr(dataset, "name", None))

    return build_audit_task(dataset, task_solver, task_scorer, cond, profile)
