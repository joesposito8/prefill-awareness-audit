"""Baseline awareness -- self-recognition and false-attribution rates.

Calibration experiment for the prefill-awareness-audit package. Probes
models on Chatbot Arena conversations to measure how well they distinguish
their own writing from another model's writing. See
case_studies/baseline_awareness/README.md for study design and usage.

Quick start (uses committed fixture, no data prep needed):
    inspect eval case_studies/baseline_awareness_audit.py \
        -T condition=PROBE_ONLY \
        --model anthropic/claude-opus-4-20250514

Full dataset (after running prepare_data.py):
    inspect eval case_studies/baseline_awareness_audit.py \
        -T condition=PROBE_ONLY \
        -T data=case_studies/baseline_awareness/data/arena_conversations.jsonl \
        --model anthropic/claude-opus-4-20250514
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task

from prefill_awareness_audit import make_audit_task

_DEFAULT_DATA = str(
    Path(__file__).parent / "baseline_awareness" / "data" / "sample.jsonl"
)


@task
def baseline_awareness_audit(
    condition: str = "PROBE_ONLY",
    data: str = _DEFAULT_DATA,
    limit: int | None = None,
) -> Task:
    """Baseline awareness audit on Chatbot Arena conversations."""
    return make_audit_task(data=data, condition=condition, limit=limit)
