"""Shared utilities for probe solvers."""

from __future__ import annotations

from inspect_ai.model import ChatMessageTool
from inspect_ai.solver import TaskState


def append_tool_stubs(state: TaskState) -> None:
    """Append ChatMessageTool stubs for any pending tool calls in state.output.

    LLM APIs require tool_result blocks for every tool_use block before the
    next user message. Inspect AI translates ChatMessageTool to the correct
    provider-specific format automatically (Anthropic, OpenAI, Google, etc.).
    When chaining probe solvers, each probe must stub any tool calls from
    the previous generation.
    """
    if (
        state.output
        and state.output.message
        and state.output.message.tool_calls
    ):
        for tc in state.output.message.tool_calls:
            state.messages.append(
                ChatMessageTool(
                    content="[Tool execution skipped for follow-up question]",
                    tool_call_id=tc.id,
                    function=tc.function,
                )
            )
