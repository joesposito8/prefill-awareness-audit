"""Combined awareness probe -- attribution + confidence in a single prompt."""

from __future__ import annotations

import re

from inspect_ai._util.pattern import ANSWER_PATTERN_LINE
from inspect_ai._util.text import str_to_float, strip_numeric_punctuation
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Solver, TaskState, solver

from ._util import append_tool_stubs
from .prompts import DEFAULT_AWARENESS_QUESTION


# ---------------------------------------------------------------------------
# Attribution parser (moved from attribution.py)
# ---------------------------------------------------------------------------


def _parse_attribution(text: str) -> str | None:
    """Parse three-way attribution from model response.

    Uses Inspect AI's ANSWER_PATTERN_LINE as the primary extractor,
    with bare keyword fallback for non-conforming responses.

    Returns:
        "self", "not_self", "uncertain", or None if unparseable.
    """
    text_lower = text.lower().strip()

    # Primary: Inspect AI's "ANSWER: <rest of line>" pattern
    match = re.search(ANSWER_PATTERN_LINE, text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip().lower()
        if "not" in answer and "self" in answer:
            return "not_self"
        if answer == "self" or answer.startswith("self"):
            return "self"
        if "uncertain" in answer or "unsure" in answer or "not sure" in answer:
            return "uncertain"

    # Fallback: bare keyword search
    if "not self" in text_lower or "not_self" in text_lower:
        return "not_self"
    if "uncertain" in text_lower or "unsure" in text_lower or "not sure" in text_lower:
        return "uncertain"
    if "self" in text_lower:
        return "self"
    return None


# ---------------------------------------------------------------------------
# Confidence parser (moved from confidence.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Combined awareness parser
# ---------------------------------------------------------------------------


def _parse_awareness(text: str) -> tuple[str | None, int | None]:
    """Parse combined attribution + confidence from a single model response.

    Looks for labeled ATTRIBUTION: and CONFIDENCE: markers first, then
    falls back to the standalone parsers.

    Returns:
        (attribution_label, confidence_value) where attribution_label is
        one of "self", "not_self", "uncertain", or None if unparseable,
        and confidence_value is 0-100 or None.
    """
    attribution: str | None = None
    confidence: int | None = None

    # Primary: labeled markers
    attr_match = re.search(r"ATTRIBUTION:\s*(.+)", text, re.IGNORECASE)
    if attr_match:
        attr_text = attr_match.group(1).strip().lower()
        if "not" in attr_text and "self" in attr_text:
            attribution = "not_self"
        elif attr_text.startswith("self") or attr_text == "self":
            attribution = "self"
        elif "uncertain" in attr_text or "unsure" in attr_text:
            attribution = "uncertain"

    conf_match = re.search(r"CONFIDENCE:\s*(\d+)", text, re.IGNORECASE)
    if conf_match:
        val = int(conf_match.group(1))
        if 0 <= val <= 100:
            confidence = val

    # Fallback: use standalone parsers on appropriate text segments
    if attribution is None:
        attribution = _parse_attribution(text)

    if confidence is None:
        # Only parse text after CONFIDENCE marker to avoid picking up
        # numbers from attribution text
        conf_marker = re.search(r"CONFIDENCE:", text, re.IGNORECASE)
        if conf_marker:
            confidence = _parse_confidence(text[conf_marker.start() :])
        else:
            confidence = _parse_confidence(text)

    return attribution, confidence


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@solver
def awareness_probe(
    question_template: str = DEFAULT_AWARENESS_QUESTION,
) -> Solver:
    async def solve(state: TaskState, generate) -> TaskState:
        # Stub any tool calls from the main generation
        append_tool_stubs(state)

        # Ask the combined awareness question
        state.messages.append(ChatMessageUser(content=question_template))
        state = await generate(state)

        # Parse and store in the same metadata keys the old probes used
        raw = state.output.completion if state.output else ""
        attr_label, conf_value = _parse_awareness(raw)

        if state.metadata is None:
            state.metadata = {}
        state.metadata["attribution"] = {"label": attr_label, "raw_response": raw}
        state.metadata["prefill_confidence"] = {
            "value": conf_value,
            "raw_response": raw,
        }

        return state

    return solve
