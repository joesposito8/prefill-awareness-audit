"""User-facing types for the prefill awareness audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import Score


class Condition(StrEnum):
    """Audit conditions — each is an independent intervention branch."""

    BASELINE = "BASELINE"
    PROBE_ONLY = "PROBE_ONLY"
    LOCAL_DETERMINISTIC = "LOCAL_DETERMINISTIC"
    LOCAL_REWRITE = "LOCAL_REWRITE"
    BLOCK_DETERMINISTIC = "BLOCK_DETERMINISTIC"
    BLOCK_REWRITE = "BLOCK_REWRITE"
    GLOBAL_DETERMINISTIC = "GLOBAL_DETERMINISTIC"


@dataclass(frozen=True)
class AuditTarget:
    """Per-sample target span identifying which messages to intervene on."""

    sample_id: str
    message_indices: list[int]
    target_kind: str  # "single_turn" or "assistant_block"


@dataclass(frozen=True)
class RewritePolicy:
    """Permission-based policy controlling which interventions are admissible."""

    allow_local_deterministic: bool = True
    allow_local_rewrite: bool = True
    allow_block_deterministic: bool = False
    allow_block_rewrite: bool = False
    allow_global_deterministic: bool = False
    protected_roles: tuple[str, ...] = ("user", "tool")
    preserve_message_count: bool = True
    preserve_role_order: bool = True


@dataclass
class AuditProfile:
    """The one object a benchmark integration provides to configure the audit."""

    name: str
    main_score_fields: list[str]
    target_span_selector: Callable[[Sample], AuditTarget]
    allowed_conditions: list[Condition]
    rewrite_policy: RewritePolicy
    semantic_guard: Callable[[Sample, Sample], list[str]] | None = None
    baseline_control_loader: Callable | None = None
    entanglement_notes: str | None = None


@dataclass
class Scorecard:
    """Computed post-hoc from all condition logs. Contains all 10 score families."""

    # 1. Main benchmark scores: condition -> {score_field: value}
    benchmark_scores: dict[Condition, dict[str, float]] = field(default_factory=dict)

    # 2. Direct awareness: condition -> {self_rate, not_self_rate, uncertain_rate}
    direct_awareness: dict[Condition, dict[str, float]] = field(default_factory=dict)

    # 3. Confidence: condition -> {mean_tamper_confidence, median_tamper_confidence}
    confidence: dict[Condition, dict[str, float]] = field(default_factory=dict)

    # 4. Diagnostic reasons: condition -> {reason_tag_distribution, flagged_turn_distribution}
    diagnostic_reasons: dict[Condition, dict[str, Any]] = field(default_factory=dict)

    # 5. Spontaneous awareness: condition -> {spontaneous_suspicion_rate}
    spontaneous_awareness: dict[Condition, dict[str, float]] = field(default_factory=dict)

    # 6. Control baselines (optional, externally provided)
    control_baselines: dict[str, float] | None = None

    # 7. Awareness gaps vs control (optional)
    awareness_gaps: dict[str, float] | None = None

    # 8. Awareness-benchmark coupling
    awareness_benchmark_coupling: dict[str, float] = field(default_factory=dict)

    # 9. Intervention response: condition -> {awareness_delta, main_score_delta}
    intervention_response: dict[Condition, dict[str, float]] = field(default_factory=dict)

    # 10. Residual awareness
    residual_awareness: dict[str, float] = field(default_factory=dict)

    # Classification label
    classification: str | None = None
