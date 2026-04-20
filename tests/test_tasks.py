"""End-to-end tests for the generic @task in _tasks.py.

Uses Inspect's built-in ``mockllm/model`` provider so evals run with no API
key and no network. These tests exercise the full task-construction path
including ``fork()`` over multiple probes, which is where the multi-probe
CLI crash surfaced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval

from prefill_awareness_audit._tasks import prefill_awareness_audit


def _write_fixture(tmp_path: Path) -> Path:
    jsonl = tmp_path / "sample.jsonl"
    record = {
        "input": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there."},
        ],
        "target": "test",
        "id": "s-001",
        "metadata": {"source_model": "synthetic"},
    }
    jsonl.write_text(json.dumps(record))
    return jsonl


@pytest.mark.parametrize(
    "probes",
    [
        pytest.param(["awareness", "diagnostic"], id="list-from-cli-split"),
        pytest.param("awareness,diagnostic", id="comma-separated-string"),
    ],
)
def test_prefill_awareness_audit_multi_probe_eval_mockllm(
    tmp_path: Path, probes: list[str] | str
) -> None:
    """Multi-probe via the @task function must complete under mockllm.

    Regression for the crash where Inspect's CLI auto-split
    ``-T probes=a,b`` into a list[str] and the downstream fork() raised
    ``ValueError: Object does not have registry info``.
    """
    task = prefill_awareness_audit(data=str(_write_fixture(tmp_path)), probes=probes)
    logs = inspect_eval(task, model="mockllm/model", limit=1, display="none")

    assert len(logs) == 1
    assert logs[0].status == "success"


def test_prefill_awareness_audit_single_probe_eval_mockllm(tmp_path: Path) -> None:
    """Single-probe string path still drives a successful eval (regression guard)."""
    task = prefill_awareness_audit(data=str(_write_fixture(tmp_path)), probes="awareness")
    logs = inspect_eval(task, model="mockllm/model", limit=1, display="none")

    assert len(logs) == 1
    assert logs[0].status == "success"
