"""Shared utilities for probe solvers."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from enum import Enum
from typing import TypeVar

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.solver import TaskState

# ``prompts`` imports are kept lazy (inside ``classify_judgement``) so this
# module can be imported from ``prompts.py`` without a cycle — the RNG
# helpers below are shared infrastructure that ``prompts.py`` calls into.

logger = logging.getLogger(__name__)

_E = TypeVar("_E", bound=Enum)


def per_sample_seed(sample_id: str, base_seed: int = 42) -> int:
    """Deterministic per-sample integer seed for probe-option randomisation.

    All probes that present ordered options to the subject model (MCQ
    letter shuffle, awareness polarity flip, diagnostic tag shuffle)
    share this seed formula so re-running the same eval produces
    byte-identical prompts per sample across runs and processes.
    ``base_seed`` is overridable per-probe for independent randomness
    across probes on the same sample.

    Uses SHA-1 over the sample id rather than builtin ``hash()`` — the
    latter is PEP 456 process-salted, so byte-identical reproducibility
    would only hold within a single interpreter.  ``sample_id`` is
    stringified before hashing so Inspect's ``str | int`` sample ids
    are both accepted.
    """
    digest = hashlib.sha1(str(sample_id).encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:8], "big", signed=False)


def per_sample_rng(sample_id: str, base_seed: int = 42) -> random.Random:
    """Per-sample deterministic RNG seeded via :func:`per_sample_seed`."""
    return random.Random(per_sample_seed(sample_id, base_seed))


def coerce_enum(value: object, enum_cls: type[_E]) -> _E | None:
    """Coerce ``value`` to ``enum_cls``, returning ``None`` on failure.

    Accepts an enum instance (returned as-is), a raw value (constructed
    via ``enum_cls(value)``), or anything falsy / invalid (``None``).
    """
    if value is None or value == "":
        return None
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        return None


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
) -> "LatentCategory | None":
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
    from .prompts import (
        JUDGE_LABEL_TO_CATEGORY,
        LATENT_JUDGE_SCHEMA,
        LatentJudgementLabel,
    )

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
        label = LatentJudgementLabel(parsed.get("category"))
    except ValueError:
        return None
    return JUDGE_LABEL_TO_CATEGORY[label]
