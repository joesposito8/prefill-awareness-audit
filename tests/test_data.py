"""Tests for data loading — JSONL, eval log reading, and log discovery."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

from prefill_awareness_audit import Condition, InterventionConstraints
from prefill_awareness_audit.data import (
    PROBE_ONLY_PROFILE,
    all_assistant_turns,
    find_eval_log,
    load_conversations,
    load_from_eval_log,
)


# ---------------------------------------------------------------------------
# load_conversations
# ---------------------------------------------------------------------------


def test_load_conversations(tmp_path: Path) -> None:
    """Load samples from a JSONL file."""
    jsonl = tmp_path / "data.jsonl"
    records = [
        {
            "input": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "target": "test",
            "id": "s-001",
        },
        {
            "input": [
                {"role": "user", "content": "Bye"},
                {"role": "assistant", "content": "Goodbye"},
            ],
            "target": "test",
            "id": "s-002",
        },
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in records))

    samples = load_conversations(jsonl)
    assert len(samples) == 2
    assert samples[0].id == "s-001"
    assert isinstance(samples[0].input, list)
    assert len(samples[0].input) == 2
    # source_model defaults to "unknown" when absent from JSONL
    assert samples[0].metadata["source_model"] == "unknown"
    assert samples[1].metadata["source_model"] == "unknown"


def test_load_conversations_preserves_source_model(tmp_path: Path) -> None:
    """Preserves source_model from JSONL metadata when present."""
    jsonl = tmp_path / "data.jsonl"
    record = {
        "input": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        "id": "s-010",
        "metadata": {"source_model": "anthropic/claude-sonnet-4-6"},
    }
    jsonl.write_text(json.dumps(record))

    samples = load_conversations(jsonl)
    assert samples[0].metadata["source_model"] == "anthropic/claude-sonnet-4-6"


def test_load_conversations_empty_file(tmp_path: Path) -> None:
    """Empty JSONL produces an empty list."""
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("")
    samples = load_conversations(jsonl)
    assert samples == []


# ---------------------------------------------------------------------------
# find_eval_log
# ---------------------------------------------------------------------------


def _make_log_info(task_name: str, path: str) -> MagicMock:
    info = MagicMock()
    info.task = task_name
    info.name = path
    return info


def _make_header_log(task_name: str) -> MagicMock:
    log = MagicMock()
    log.eval.task = task_name
    return log


@patch("prefill_awareness_audit.data.list_eval_logs")
def test_find_eval_log_by_task(mock_list: MagicMock) -> None:
    """Finds the most recent log matching a task name."""
    info = _make_log_info("my_benchmark", "/logs/2024-01-15_my_benchmark.eval")
    mock_list.return_value = [info]

    result = find_eval_log("my_benchmark", log_dir="/logs")
    assert result == "/logs/2024-01-15_my_benchmark.eval"
    mock_list.assert_called_once_with("/logs")


@patch("prefill_awareness_audit.data.list_eval_logs")
def test_find_eval_log_no_match(mock_list: MagicMock) -> None:
    """Raises FileNotFoundError with available task names when no match."""
    mock_list.return_value = [_make_log_info("other_task", "/logs/other.eval")]

    with pytest.raises(FileNotFoundError, match="other_task"):
        find_eval_log("nonexistent", log_dir="/logs")


@patch("prefill_awareness_audit.data.list_eval_logs")
def test_find_eval_log_empty_dir(mock_list: MagicMock) -> None:
    """Raises FileNotFoundError when log dir has no logs at all."""
    mock_list.return_value = []

    with pytest.raises(FileNotFoundError, match="No eval logs found"):
        find_eval_log("any_task", log_dir="/empty")


@patch.dict("os.environ", {"INSPECT_LOG_DIR": "/custom/logs"})
@patch("prefill_awareness_audit.data.list_eval_logs")
def test_find_eval_log_uses_env_var(mock_list: MagicMock) -> None:
    """Falls back to INSPECT_LOG_DIR when log_dir is empty."""
    info = _make_log_info("my_task", "/custom/logs/task.eval")
    mock_list.return_value = [info]

    result = find_eval_log("my_task")
    assert mock_list.call_args[0][0] == "/custom/logs"
    assert result == "/custom/logs/task.eval"


# ---------------------------------------------------------------------------
# load_from_eval_log
# ---------------------------------------------------------------------------


def _make_eval_sample(
    sample_id: str | int,
    messages: list[tuple[str, str]],
) -> MagicMock:
    _role_map = {"user": ChatMessageUser, "assistant": ChatMessageAssistant}
    sample = MagicMock()
    sample.id = sample_id
    sample.messages = [_role_map[role](content=content) for role, content in messages]
    sample.scores = {"aligned": MagicMock(value=1)}
    return sample


def _make_eval_log(
    model: str = "anthropic/claude-opus-4-6",
    samples: list | None = None,
) -> MagicMock:
    log = MagicMock()
    log.eval.model = model
    log.samples = samples
    return log


@patch("prefill_awareness_audit.data.read_eval_log")
def test_load_from_eval_log(mock_read: MagicMock) -> None:
    """Extracts samples and model_id from an eval log."""
    eval_samples = [
        _make_eval_sample("s-001", [("user", "Hi"), ("assistant", "Hello")]),
        _make_eval_sample("s-002", [("user", "Bye"), ("assistant", "Goodbye")]),
    ]
    mock_read.return_value = _make_eval_log(samples=eval_samples)

    samples, model_id = load_from_eval_log("/logs/test.eval")

    assert len(samples) == 2
    assert model_id == "anthropic/claude-opus-4-6"
    assert samples[0].id == "s-001"
    assert isinstance(samples[0].input, list)
    assert samples[0].metadata["original_scores"] is not None
    assert samples[0].metadata["source_model"] == "anthropic/claude-opus-4-6"
    assert samples[1].metadata["source_model"] == "anthropic/claude-opus-4-6"


@patch("prefill_awareness_audit.data.read_eval_log")
def test_load_from_eval_log_with_limit(mock_read: MagicMock) -> None:
    """Respects the limit parameter."""
    eval_samples = [
        _make_eval_sample(i, [("user", "Q"), ("assistant", "A")])
        for i in range(10)
    ]
    mock_read.return_value = _make_eval_log(samples=eval_samples)

    samples, _ = load_from_eval_log("/logs/test.eval", limit=3)
    assert len(samples) == 3


@patch("prefill_awareness_audit.data.read_eval_log")
def test_load_from_eval_log_empty(mock_read: MagicMock) -> None:
    """Returns empty list when log has no samples."""
    mock_read.return_value = _make_eval_log(samples=None)

    samples, model_id = load_from_eval_log("/logs/empty.eval")
    assert samples == []
    assert model_id == "anthropic/claude-opus-4-6"


@patch("prefill_awareness_audit.data.read_eval_log")
def test_load_from_eval_log_empty_messages_raises(mock_read: MagicMock) -> None:
    """Raises ValueError when a sample has no messages."""
    sample = MagicMock()
    sample.id = "bad"
    sample.messages = []
    mock_read.return_value = _make_eval_log(samples=[sample])

    with pytest.raises(ValueError, match="no messages"):
        load_from_eval_log("/logs/bad.eval")


# ---------------------------------------------------------------------------
# all_assistant_turns
# ---------------------------------------------------------------------------


def test_all_assistant_turns_mixed_roles() -> None:
    """Selects only assistant message indices."""
    sample = Sample(
        input=[
            ChatMessageUser(content="Hi"),
            ChatMessageAssistant(content="Hello"),
            ChatMessageUser(content="Question"),
            ChatMessageAssistant(content="Answer"),
            ChatMessageUser(content="Thanks"),
        ],
        target="test",
        id="s-001",
    )
    target = all_assistant_turns(sample)
    assert target.message_indices == [1, 3]
    assert target.target_kind == "all_assistant"
    assert target.sample_id == "s-001"


def test_all_assistant_turns_no_assistant() -> None:
    """Returns empty indices when no assistant messages exist."""
    sample = Sample(
        input=[ChatMessageUser(content="Hi")],
        target="test",
        id="s-002",
    )
    target = all_assistant_turns(sample)
    assert target.message_indices == []


def test_all_assistant_turns_string_input() -> None:
    """Handles string input gracefully."""
    sample = Sample(input="just a string", target="test", id="s-003")
    target = all_assistant_turns(sample)
    assert target.message_indices == []


# ---------------------------------------------------------------------------
# latest_assistant_block
# ---------------------------------------------------------------------------


def test_latest_assistant_block_finds_trailing_run() -> None:
    """Identifies the most recent contiguous assistant/tool block before the tail."""
    from inspect_ai.model import ChatMessageSystem, ChatMessageTool

    from prefill_awareness_audit.data import latest_assistant_block

    sample = Sample(
        input=[
            ChatMessageSystem(content="system"),        # 0
            ChatMessageUser(content="task"),            # 1
            ChatMessageAssistant(content="first"),      # 2
            ChatMessageUser(content="follow-up"),       # 3
            ChatMessageAssistant(content="thinking"),   # 4
            ChatMessageTool(content="result"),          # 5
            ChatMessageAssistant(content="summary"),    # 6
            ChatMessageUser(content="one more thing"),  # 7 — trailing user
        ],
        target="test",
        id="s-blk",
    )
    target = latest_assistant_block(sample)
    # Skip trailing user (7), then collect backward through 6, 5, 4
    assert target.message_indices == [4, 5, 6]
    assert target.target_kind == "assistant_block"
    assert target.sample_id == "s-blk"


def test_latest_assistant_block_no_assistants() -> None:
    """Returns empty indices when there are no assistant turns."""
    from prefill_awareness_audit.data import latest_assistant_block

    sample = Sample(
        input=[ChatMessageUser(content="only user")],
        target="test",
        id="s-no",
    )
    target = latest_assistant_block(sample)
    assert target.message_indices == []


def test_latest_assistant_block_string_input() -> None:
    """Handles string input gracefully."""
    from prefill_awareness_audit.data import latest_assistant_block

    sample = Sample(input="raw string", target="test", id="s-str")
    target = latest_assistant_block(sample)
    assert target.message_indices == []


# ---------------------------------------------------------------------------
# PROBE_ONLY_PROFILE
# ---------------------------------------------------------------------------


def test_default_profile_conditions() -> None:
    """PROBE_ONLY_PROFILE allows only PROBE_ONLY."""
    assert PROBE_ONLY_PROFILE.allowed_conditions == [
        Condition.PROBE_ONLY,
    ]


def test_default_profile_no_main_scores() -> None:
    """PROBE_ONLY_PROFILE has no benchmark score fields."""
    assert PROBE_ONLY_PROFILE.main_score_fields == []


def test_default_profile_default_constraints() -> None:
    """PROBE_ONLY_PROFILE uses default InterventionConstraints."""
    assert PROBE_ONLY_PROFILE.intervention_constraints == InterventionConstraints()


def test_default_profile_selector_is_all_assistant() -> None:
    """PROBE_ONLY_PROFILE uses all_assistant_turns as target selector."""
    assert PROBE_ONLY_PROFILE.target_span_selector is all_assistant_turns
