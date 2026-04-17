"""Shared utilities for probe solvers."""

from __future__ import annotations

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool
from inspect_ai.solver import TaskState


def last_assistant_message(
    messages: list[ChatMessage],
) -> ChatMessageAssistant | None:
    """Return the most recent assistant message, or None if none exists.

    Walks the conversation backwards so trailing tool/user/system messages
    don't obscure the model's final output.  Works whether or not generate()
    has run — after generate, Inspect appends the response to messages;
    before generate (audit-only runs), the last assistant message is the
    pre-existing one from the transcript.
    """
    for msg in reversed(messages):
        if msg.role == "assistant":
            return msg
    return None


def append_tool_stubs(state: TaskState) -> None:
    """Append ChatMessageTool stubs for any pending tool calls in the last turn.

    LLM APIs require tool_result blocks for every tool_use block before the
    next user message. Inspect AI translates ChatMessageTool to the correct
    provider-specific format automatically (Anthropic, OpenAI, Google, etc.).
    When chaining probe solvers, each probe must stub any tool calls from
    the previous generation.
    """
    last = last_assistant_message(state.messages)
    if last and last.tool_calls:
        for tc in last.tool_calls:
            state.messages.append(
                ChatMessageTool(
                    content="[Tool execution skipped for follow-up question]",
                    tool_call_id=tc.id,
                    function=tc.function,
                )
            )
