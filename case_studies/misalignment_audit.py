"""Misalignment-continuation case study — AuditProfile and @task definition.

Integration for the UKGovernmentBEIS/misalignment-continuation benchmark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prefill_awareness_audit import AuditProfile, Condition, RewritePolicy, audit_task

if TYPE_CHECKING:
    from inspect_ai import Task
    from inspect_ai.dataset import Sample

    from prefill_awareness_audit import AuditTarget


def get_target_span(sample: Sample) -> AuditTarget:
    """Identify the target assistant span for intervention."""
    raise NotImplementedError("Will be implemented after studying the dataset structure")


def load_dataset():
    """Load the misalignment-continuation dataset."""
    raise NotImplementedError("Will be implemented in a later task")


def misalignment_continuation_scorer():
    """Scorer for the misalignment-continuation benchmark."""
    raise NotImplementedError("Will be implemented in a later task")


profile = AuditProfile(
    name="misalignment-continuation",
    main_score_fields=["aligned", "confession"],
    target_span_selector=get_target_span,
    allowed_conditions=[
        Condition.BASELINE,
        Condition.PROBE_ONLY,
        Condition.LOCAL_DETERMINISTIC,
        Condition.LOCAL_REWRITE,
        Condition.BLOCK_DETERMINISTIC,
        Condition.BLOCK_REWRITE,
        Condition.GLOBAL_DETERMINISTIC,
    ],
    rewrite_policy=RewritePolicy(
        allow_block_deterministic=True,
        allow_block_rewrite=True,
        allow_global_deterministic=True,
    ),
)


# @task
# def misalignment_audit(condition: str = "BASELINE") -> Task:
#     return audit_task(
#         dataset=load_dataset(),
#         solver=[...],
#         scorer=misalignment_continuation_scorer(),
#         condition=Condition[condition],
#         profile=profile,
#     )
