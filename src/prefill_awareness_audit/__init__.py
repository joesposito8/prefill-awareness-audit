"""Prefill Awareness Audit — audit scaffold for transcript-based AI evals."""

from .data import (
    PROBE_ONLY_PROFILE,
    all_assistant_turns,
    find_eval_log,
    load_conversations,
    load_from_eval_log,
)
from .task_factory import audit_task, audit_tasks, build_audit_task, make_audit_task
from .types import (
    AuditProfile,
    AuditTarget,
    ComparisonTable,
    Condition,
    ConditionSummary,
    RewritePolicy,
)

__all__ = [
    # Primary API
    "make_audit_task",
    "PROBE_ONLY_PROFILE",
    # Data loading
    "load_conversations",
    "load_from_eval_log",
    "find_eval_log",
    "all_assistant_turns",
    # Task builders
    "audit_task",
    "audit_tasks",
    "build_audit_task",
    # Types
    "AuditProfile",
    "AuditTarget",
    "ComparisonTable",
    "Condition",
    "ConditionSummary",
    "RewritePolicy",
]
