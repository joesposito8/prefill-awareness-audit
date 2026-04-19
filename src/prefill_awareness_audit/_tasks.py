"""Generic @task for Inspect CLI discovery — three input paths."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from .data import PROBE_ONLY_PROFILE, find_eval_log, load_from_eval_log
from .task_factory import make_audit_task


@task
def prefill_awareness_audit(
    task: str = "",
    log: str = "",
    log_dir: str = "",
    data: str = "",
    condition: str = "PROBE_ONLY",
    probes: str = "",
    limit: int | None = None,
    seed: int = 42,
) -> Task:
    """Audit prefill awareness in transcript-based AI evals.

    Provide exactly one data source:

    - ``task=name`` — find the most recent eval log for the named task
      and extract its conversation histories for auditing.
    - ``log=path`` — read a specific eval log file.
    - ``data=path`` — load conversations from a JSONL file.

    Examples::

        inspect eval prefill_awareness_audit/prefill_awareness_audit -T task=my_benchmark
        inspect eval prefill_awareness_audit/prefill_awareness_audit -T log=logs/my_benchmark.eval
        inspect eval prefill_awareness_audit/prefill_awareness_audit -T data=conversations.jsonl \\
            --model anthropic/claude-sonnet-4-6
        inspect eval prefill_awareness_audit/prefill_awareness_audit -T task=my_benchmark \\
            -T probes=awareness,reflection

    Args:
        task: Task name — discovers the most recent matching eval log.
        log: Explicit path to an eval log file.
        log_dir: Log directory override (default: INSPECT_LOG_DIR or ./logs).
        data: Path to a JSONL file of conversations.
        condition: Audit condition name (PROBE_ONLY, etc.).
        probes: Comma-separated probe names to run — any subset of
            ``awareness``, ``reflection``, ``persona``, ``diagnostic``,
            ``multiple_choice``. Empty (default) runs the first four;
            ``multiple_choice`` is opt-in (the legacy MCQ probe, kept for
            A/B comparison against the free-text probes).
        limit: Maximum number of samples.
        seed: Random seed.
    """
    sources = sum(bool(s) for s in (task, log, data))
    if sources == 0:
        raise ValueError(
            "Provide a data source: task=<name>, log=<path>, or data=<path.jsonl>.\n\n"
            "Examples:\n"
            "  inspect eval prefill_awareness_audit/prefill_awareness_audit -T task=my_benchmark\n"
            "  inspect eval prefill_awareness_audit/prefill_awareness_audit -T log=logs/my_eval.eval\n"
            "  inspect eval prefill_awareness_audit/prefill_awareness_audit -T data=conversations.jsonl "
            "--model anthropic/claude-sonnet-4-6"
        )
    if sources > 1:
        raise ValueError("Provide exactly one of task=, log=, or data=.")

    if log or task:
        # Log-based paths
        if task:
            log_path = find_eval_log(task, log_dir)
        else:
            log_path = log

        samples, model_id = load_from_eval_log(log_path, limit=limit)
        dataset = MemoryDataset(samples, name=f"audit-{task or 'log'}")

        return make_audit_task(
            data=dataset,
            condition=condition,
            profile=PROBE_ONLY_PROFILE,
            solver=[],  # conversations already complete — skip generate()
            probes=probes or None,
            seed=seed,
        )

    # Data path
    return make_audit_task(
        data=data,
        condition=condition,
        profile=PROBE_ONLY_PROFILE,
        probes=probes or None,
        limit=limit,
        seed=seed,
    )
