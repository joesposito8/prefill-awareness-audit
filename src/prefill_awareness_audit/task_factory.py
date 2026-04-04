"""Main composition point — builds an Inspect Task for a given audit condition."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inspect_ai import Task
    from inspect_ai.dataset import Dataset
    from inspect_ai.scorer import Scorer
    from inspect_ai.solver import Solver

    from .types import AuditProfile, Condition


def audit_task(
    dataset: Dataset,
    solver: list[Solver],
    scorer: Scorer,
    condition: Condition,
    profile: AuditProfile,
) -> Task:
    """Build an Inspect Task that runs one audit condition.

    Applies the intervention specified by `condition` to the dataset,
    appends probes after generate(), and wraps the scorer to capture
    both original benchmark dimensions and awareness dimensions.
    """
    raise NotImplementedError("audit_task will be implemented in a later task")
