"""Interventions -- dataset transformers parameterized by (scope, mechanism).

The ``apply_intervention`` function is the single entry point. It validates
the condition against the profile's allowed_conditions, resolves scope,
dispatches to the appropriate transformer, and runs structural validation
on the result.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from prefill_awareness_audit.interventions.deterministic import (
    deterministic_intervention,
)
from prefill_awareness_audit.interventions.rewrite import (
    DEFAULT_REWRITER_MODEL,
    rewrite_intervention,
)
from prefill_awareness_audit.interventions.validator import validate_intervention
from prefill_awareness_audit.types import Condition

if TYPE_CHECKING:
    from inspect_ai.dataset import Sample

    from prefill_awareness_audit.types import AuditProfile

logger = logging.getLogger(__name__)

# Conditions that use the rewrite mechanism (async model calls).
_REWRITE_CONDITIONS = frozenset({
    Condition.LOCAL_REWRITE,
    Condition.BLOCK_REWRITE,
})

# Conditions that require no dataset transformation.
_PASSTHROUGH_CONDITIONS = frozenset({
    Condition.PROBE_ONLY,
})


def _resolve_scope(
    condition: Condition,
    sample: Sample,
    profile: AuditProfile,
) -> tuple[list[int], list[int]]:
    """Determine which message indices to intervene on.

    Returns:
        (intervention_indices, provenance_only_indices).
        ``intervention_indices`` get the full treatment (cleanup or rewrite).
        ``provenance_only_indices`` get only provenance marker stripping
        (used for protected roles in global deterministic scope).
    """
    messages = sample.input
    if not isinstance(messages, list):
        return [], []

    if condition in _PASSTHROUGH_CONDITIONS:
        return [], []

    if condition in (Condition.LOCAL_DETERMINISTIC, Condition.LOCAL_REWRITE):
        target = profile.target_span_selector(sample)
        return target.message_indices, []

    if condition in (Condition.BLOCK_DETERMINISTIC, Condition.BLOCK_REWRITE):
        # All assistant messages before the last message (the continuation point)
        indices = [
            i
            for i in range(len(messages) - 1)
            if messages[i].role == "assistant"
        ]
        return indices, []

    if condition == Condition.GLOBAL_DETERMINISTIC:
        protected = set(profile.intervention_constraints.protected_roles)
        full_indices = [
            i for i, m in enumerate(messages) if m.role not in protected
        ]
        provenance_indices = [
            i for i, m in enumerate(messages) if m.role in protected
        ]
        return full_indices, provenance_indices

    return [], []


async def apply_intervention(
    samples: list[Sample],
    condition: Condition,
    profile: AuditProfile,
    rewriter_model: str = DEFAULT_REWRITER_MODEL,
    cache_dir: Path | None = None,
) -> list[Sample]:
    """Apply an intervention to a list of samples.

    Validates the condition against the profile's allowed_conditions,
    transforms each sample, runs structural validation, and optionally
    runs the semantic guard. Returns only samples that pass both
    validation phases.

    Args:
        samples: Original samples (not modified).
        condition: The audit condition to apply.
        profile: Benchmark-specific audit configuration.
        rewriter_model: Inspect AI model string for rewrite conditions.
        cache_dir: Directory for rewrite caches.

    Returns:
        List of (possibly modified) samples that passed validation.

    Raises:
        ValueError: If the condition is not in the profile's
            allowed_conditions.
    """
    if condition not in profile.allowed_conditions:
        raise ValueError(
            f"Condition {condition} is not in profile.allowed_conditions "
            f"({[c.value for c in profile.allowed_conditions]})"
        )

    if condition in _PASSTHROUGH_CONDITIONS:
        return list(samples)

    is_rewrite = condition in _REWRITE_CONDITIONS
    passed_samples: list[Sample] = []
    total = len(samples)
    passed_structural = 0
    passed_semantic = 0
    excluded = 0

    for sample in samples:
        indices, provenance_only_indices = _resolve_scope(
            condition, sample, profile
        )

        if is_rewrite:
            modified = await rewrite_intervention(
                sample,
                indices,
                profile.intervention_constraints,
                rewriter_model=rewriter_model,
                cache_dir=cache_dir,
            )
        else:
            modified = deterministic_intervention(
                sample,
                indices,
                profile.intervention_constraints,
                provenance_only_indices=provenance_only_indices,
            )

        # Phase 1: structural validation
        valid, violations = validate_intervention(
            sample, modified, indices, profile.intervention_constraints
        )
        if not valid:
            logger.info(
                "Sample %s failed structural validation: %s",
                sample.id,
                "; ".join(violations),
            )
            excluded += 1
            continue

        passed_structural += 1

        # Phase 2: semantic guard (optional)
        if profile.semantic_guard is not None:
            semantic_violations = profile.semantic_guard(sample, modified)
            if semantic_violations:
                logger.info(
                    "Sample %s failed semantic guard: %s",
                    sample.id,
                    "; ".join(semantic_violations),
                )
                excluded += 1
                continue

        passed_semantic += 1
        passed_samples.append(modified)

    # If no semantic guard, passed_semantic == passed_structural
    if profile.semantic_guard is None:
        passed_semantic = passed_structural

    logger.info(
        "Intervention %s: %d total, %d passed structural, %d passed semantic, %d excluded",
        condition,
        total,
        passed_structural,
        passed_semantic,
        excluded,
    )

    return passed_samples
