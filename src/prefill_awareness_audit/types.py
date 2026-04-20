"""User-facing types for the prefill awareness audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import Score


class Condition(StrEnum):
    """Audit conditions — each is an independent intervention branch."""

    PROBE_ONLY = "PROBE_ONLY"
    LOCAL_DETERMINISTIC = "LOCAL_DETERMINISTIC"
    LOCAL_REWRITE = "LOCAL_REWRITE"
    BLOCK_DETERMINISTIC = "BLOCK_DETERMINISTIC"
    BLOCK_REWRITE = "BLOCK_REWRITE"
    GLOBAL_DETERMINISTIC = "GLOBAL_DETERMINISTIC"


class Attribution(StrEnum):
    """Model's claim about a prefilled assistant turn's origin."""

    SELF = "self"
    NOT_SELF = "not_self"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class AuditTarget:
    """Per-sample target span identifying which messages to intervene on."""

    sample_id: str
    message_indices: list[int]
    target_kind: str  # "single_turn" or "assistant_block"


@dataclass(frozen=True)
class InterventionConstraints:
    """Structural invariants every intervention must preserve.

    Declares which roles cannot be modified. Message count and role
    order are always preserved — the validator rejects any intervention
    that changes them.
    """

    protected_roles: tuple[str, ...] = ("user", "tool")


@dataclass
class AuditProfile:
    """The one object a benchmark integration provides to configure the audit."""

    name: str
    main_score_fields: list[str]
    target_span_selector: Callable[[Sample], AuditTarget]
    allowed_conditions: list[Condition]
    intervention_constraints: InterventionConstraints
    semantic_guard: Callable[[Sample, Sample], list[str]] | None = None


@dataclass
class ConditionSummary:
    """Metrics for a single condition, extracted from an Inspect eval log."""

    condition: Condition
    model: str
    metrics: dict[str, float] = field(default_factory=dict)
    sample_count: int = 0


@dataclass
class ComparisonTable:
    """Cross-condition comparison output from the compare tool."""

    conditions: list[ConditionSummary] = field(default_factory=list)
    deltas_vs_probe_only: dict[str, dict[str, float]] | None = None
    model_comparison: dict[str, list[ConditionSummary]] | None = None
