"""Prefill Awareness Audit — audit scaffold for transcript-based AI evals."""

from .data import (
    PROBE_ONLY_PROFILE,
    all_assistant_turns,
    find_eval_log,
    latest_assistant_block,
    load_conversations,
    load_from_eval_log,
)
from .task_factory import continuation_audit, make_audit_task
from .types import (
    AuditProfile,
    AuditTarget,
    ComparisonTable,
    Condition,
    ConditionSummary,
    InterventionConstraints,
)

__all__ = [
    # Primary API
    "make_audit_task",
    "continuation_audit",
    "PROBE_ONLY_PROFILE",
    # Data loading
    "load_conversations",
    "load_from_eval_log",
    "find_eval_log",
    "all_assistant_turns",
    "latest_assistant_block",
    # Types
    "AuditProfile",
    "AuditTarget",
    "ComparisonTable",
    "Condition",
    "ConditionSummary",
    "InterventionConstraints",
]
