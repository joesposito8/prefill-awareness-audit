"""Confidence probe -- 0-100 tampering probability."""

from __future__ import annotations

import re

from inspect_ai._util.text import str_to_float, strip_numeric_punctuation
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Solver, TaskState, solver

from ._util import append_tool_stubs
from .prompts import DEFAULT_CONFIDENCE_QUESTION


def _parse_confidence(text: str) -> int | None:
    """Parse integer 0-100 from model response.

    Uses Inspect AI's str_to_float for robust number parsing, handling
    commas, currency symbols, percentages, and other formatting.
    """
    text = text.strip()

    # Try direct parse with Inspect AI's robust float parser
    try:
        cleaned = strip_numeric_punctuation(text)
        val = str_to_float(cleaned)
        rounded = round(val)
        if 0 <= rounded <= 100:
            return rounded
    except (ValueError, TypeError):
        pass

    # Fallback: extract first number-like token from the text
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        try:
            val = round(float(match.group(1)))
            if 0 <= val <= 100:
                return val
        except ValueError:
            pass

    return None


@solver
def confidence_probe(
    question_template: str = DEFAULT_CONFIDENCE_QUESTION,
    metadata_key: str = "prefill_confidence",
) -> Solver:
    async def solve(state: TaskState, generate) -> TaskState:
        append_tool_stubs(state)
        state.messages.append(ChatMessageUser(content=question_template))
        state = await generate(state)

        raw = state.output.completion if state.output else ""
        value = _parse_confidence(raw)

        if state.metadata is None:
            state.metadata = {}
        state.metadata[metadata_key] = {"value": value, "raw_response": raw}

        return state

    return solve
