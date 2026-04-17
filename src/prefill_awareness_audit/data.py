"""Data loading — JSONL conversations, eval log reading, and log discovery."""

from __future__ import annotations

import os
from pathlib import Path

from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

from .types import AuditProfile, AuditTarget, Condition, InterventionConstraints


def load_conversations(path: str | Path) -> list[Sample]:
    """Load conversation samples from a JSONL file.

    Each record should be an Inspect AI Sample (object with ``input``
    containing a list of chat messages, and optional ``target`` / ``id``).
    If a record's metadata lacks ``source_model``, it defaults to
    ``"unknown"``.

    Args:
        path: Path to a JSONL file.

    Returns:
        List of Sample objects.
    """
    dataset = json_dataset(str(path))
    samples = list(dataset)
    for s in samples:
        if s.metadata is None:
            s.metadata = {}
        s.metadata.setdefault("source_model", "unknown")
    return samples


def find_eval_log(task: str, log_dir: str = "") -> str:
    """Find the most recent eval log matching a task name.

    Searches the log directory for eval logs whose task name matches
    the given string.  The log directory is resolved as:

    1. ``log_dir`` parameter (if non-empty)
    2. ``INSPECT_LOG_DIR`` environment variable
    3. ``./logs`` (Inspect AI default)

    Args:
        task: Task name to search for (matched against ``EvalLog.eval.task``).
        log_dir: Optional log directory override.

    Returns:
        File path of the most recent matching log.

    Raises:
        FileNotFoundError: If no matching log is found.
    """
    resolved_dir = log_dir or os.environ.get("INSPECT_LOG_DIR", "./logs")

    all_logs = list_eval_logs(resolved_dir)
    matches = [info for info in all_logs if info.task == task]

    if not matches:
        available = sorted({info.task for info in all_logs})
        if available:
            task_list = ", ".join(available)
            raise FileNotFoundError(
                f"No eval log found for task '{task}' in {resolved_dir}. "
                f"Available tasks: {task_list}"
            )
        raise FileNotFoundError(
            f"No eval logs found in {resolved_dir}. "
            f"Run your benchmark first, or set INSPECT_LOG_DIR / pass log_dir=."
        )

    # list_eval_logs returns descending by mtime — first is most recent
    return matches[0].name


def load_from_eval_log(
    log_file: str | Path,
    limit: int | None = None,
) -> tuple[list[Sample], str | None]:
    """Read conversation samples from an Inspect eval log.

    Extracts the full conversation trace (``messages``) from each sample
    in the log and converts them into Inspect ``Sample`` objects suitable
    for auditing.  Uses ``messages`` rather than ``input`` because
    ``input`` is the original task prompt while ``messages`` is the
    actual conversation that occurred during execution.  Original
    benchmark scores are preserved in ``sample.metadata``.

    Args:
        log_file: Path to an ``.eval`` or ``.json`` log file.
        limit: Maximum number of samples to return.

    Returns:
        Tuple of (samples, model_id) where model_id is the model string
        from the original eval (e.g. ``"anthropic/claude-opus-4-6"``).

    Raises:
        TypeError: If a sample's input is a plain string rather than
            a list of chat messages.
    """
    log = read_eval_log(str(log_file))

    model_id = log.eval.model
    samples: list[Sample] = []

    if not log.samples:
        return samples, model_id

    for s in log.samples:
        if not s.messages:
            raise ValueError(
                f"Sample '{s.id}' has no messages in the log. "
                f"The audit requires a conversation trace."
            )
        samples.append(
            Sample(
                input=s.messages,
                id=str(s.id) if s.id is not None else None,
                metadata={"original_scores": s.scores, "source_model": model_id},
            )
        )

    if limit is not None:
        samples = samples[:limit]

    return samples, model_id


def all_assistant_turns(sample: Sample) -> AuditTarget:
    """Default target_span_selector — selects all assistant messages.

    Suitable for generic audits where no specific target span is defined.
    Walks ``sample.input`` and collects indices of all assistant-role messages.

    Args:
        sample: An Inspect AI Sample with a list of chat messages as input.

    Returns:
        AuditTarget with indices of all assistant messages.
    """
    messages = sample.input
    if not isinstance(messages, list):
        return AuditTarget(sample_id=sample.id or "", message_indices=[], target_kind="all_assistant")

    indices = [i for i, msg in enumerate(messages) if msg.role == "assistant"]
    return AuditTarget(
        sample_id=sample.id or "",
        message_indices=indices,
        target_kind="all_assistant",
    )


def latest_assistant_block(sample: Sample) -> AuditTarget:
    """Target span selector for the latest contiguous assistant/tool block.

    Walks backwards from the end of the conversation, skipping any trailing
    user/tool/system messages, then collects the run of assistant and tool
    messages that precedes them. Useful for agent trajectories where the
    relevant "action" is the most recent assistant turn plus its tool calls.

    Args:
        sample: An Inspect AI Sample with a list of chat messages as input.

    Returns:
        AuditTarget with indices of the latest contiguous assistant/tool block.
    """
    messages = sample.input
    if not isinstance(messages, list):
        return AuditTarget(
            sample_id=sample.id or "", message_indices=[], target_kind="assistant_block"
        )

    n = len(messages)
    i = n - 1

    # Skip trailing user/tool/system messages (the continuation point).
    while i >= 0 and messages[i].role in ("user", "tool", "system"):
        i -= 1

    # Collect the contiguous assistant/tool block going backwards.
    indices: list[int] = []
    while i >= 0 and messages[i].role in ("assistant", "tool"):
        indices.append(i)
        i -= 1

    indices.reverse()
    return AuditTarget(
        sample_id=sample.id or "",
        message_indices=indices,
        target_kind="assistant_block",
    )


PROBE_ONLY_PROFILE = AuditProfile(
    name="probe-only",
    main_score_fields=[],
    target_span_selector=all_assistant_turns,
    allowed_conditions=[Condition.PROBE_ONLY],
    intervention_constraints=InterventionConstraints(),
)
"""Conservative profile — PROBE_ONLY only.

Intervention conditions (LOCAL_DETERMINISTIC, LOCAL_REWRITE, etc.) require
a benchmark-specific AuditProfile with an appropriate target_span_selector
and a custom @task with benchmark re-scoring.
"""
