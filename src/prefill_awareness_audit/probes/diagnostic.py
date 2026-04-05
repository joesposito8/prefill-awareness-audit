"""Diagnostic probe -- reason tags and flagged turn identification."""

from __future__ import annotations

import json
import logging

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Solver, TaskState, solver

from ._util import append_tool_stubs
from .prompts import DEFAULT_DIAGNOSTIC_QUESTION, DIAGNOSTIC_SCHEMA, VALID_TAGS

logger = logging.getLogger(__name__)


def _parse_diagnostic(text: str) -> dict:
    """Parse diagnostic JSON with layered fallback.

    Tries:
    1. Direct JSON parse (works when ResponseSchema constrains output)
    2. JSON extraction from markdown code fences
    3. Brace-balanced extraction of JSON object from surrounding text
    """
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    # Try 1: direct parse
    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try 2: brace-balanced extraction (handles nested braces)
        start = cleaned.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(cleaned[start : i + 1])
                        except json.JSONDecodeError:
                            pass
                        break

    if parsed is None:
        return {
            "turn": None,
            "tags": [],
            "reason": text,
            "parse_failed": True,
        }

    # Validate tags
    tags = parsed.get("tags", [])
    validated_tags = [t for t in tags if t in VALID_TAGS]

    return {
        "turn": parsed.get("turn") or parsed.get("turn_number"),
        "tags": validated_tags,
        "reason": parsed.get("reason", ""),
        "parse_failed": False,
    }


@solver
def diagnostic_probe(
    question_template: str = DEFAULT_DIAGNOSTIC_QUESTION,
    metadata_key: str = "diagnostic",
    use_structured_output: bool = True,
) -> Solver:
    async def solve(state: TaskState, generate) -> TaskState:
        append_tool_stubs(state)
        state.messages.append(ChatMessageUser(content=question_template))

        # Use structured output when available (OpenAI, Anthropic support it)
        if use_structured_output:
            try:
                state = await generate(
                    state, response_schema=DIAGNOSTIC_SCHEMA
                )
            except Exception:
                # Model doesn't support structured output -- fall back
                logger.debug("Structured output not supported, falling back")
                state = await generate(state)
        else:
            state = await generate(state)

        raw = state.output.completion if state.output else ""
        result = _parse_diagnostic(raw)

        if state.metadata is None:
            state.metadata = {}
        state.metadata[metadata_key] = result

        return state

    return solve
