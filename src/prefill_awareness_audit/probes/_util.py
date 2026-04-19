"""Shared utilities for probe solvers."""

from __future__ import annotations

import json
import logging

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.solver import TaskState

from .prompts import LATENT_JUDGE_SCHEMA, LatentCategory

logger = logging.getLogger(__name__)


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


def _extract_json_object(text: str) -> dict | None:
    """Layered JSON extraction from model output.

    Tries:
    1. Strip markdown fences, direct json.loads().
    2. Brace-balanced extraction of the first ``{...}`` object.

    Returns the parsed object, or None if nothing parseable is found.
    Mirrors the fallback ladder in ``diagnostic._parse_diagnostic`` so
    judge parsers don't have to reinvent it.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


async def classify_judgement(
    response: str,
    template: str,
    judge_model: str,
) -> LatentCategory | None:
    """Classify a probe response through the fixed LLM judge.

    Shared by ``reflection_probe`` and ``persona_probe``.  Uses
    Inspect's ``ResponseSchema`` for structured output with a
    try/except fallback to free-form generation (same pattern as
    ``diagnostic_probe``).

    Args:
        response: The probe model's free-text reply to classify.
        template: A judge prompt template containing ``{response}``.
        judge_model: Inspect AI model string for the judge.

    Returns:
        A :class:`LatentCategory` on success, ``None`` on parse failure
        or judge error.
    """
    model = get_model(judge_model)
    prompt = template.format(response=response)
    messages = [ChatMessageUser(content=prompt)]
    try:
        output = await model.generate(
            messages, config=GenerateConfig(response_schema=LATENT_JUDGE_SCHEMA)
        )
    except Exception as exc:
        logger.warning(
            "classify_judgement: structured output call failed (%s: %s); "
            "falling back to unstructured generation.",
            type(exc).__name__,
            exc,
        )
        try:
            output = await model.generate(messages)
        except Exception as exc2:
            logger.warning(
                "classify_judgement: unstructured fallback also failed (%s: %s)",
                type(exc2).__name__,
                exc2,
            )
            return None

    raw = output.completion if output else ""
    parsed = _extract_json_object(raw)
    if parsed is None:
        return None
    try:
        return LatentCategory(parsed.get("category"))
    except ValueError:
        return None
