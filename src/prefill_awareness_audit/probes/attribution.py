"""Attribution probe -- three-way: self / not_self / uncertain."""

from __future__ import annotations

import re

from inspect_ai._util.pattern import ANSWER_PATTERN_LINE
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Solver, TaskState, solver

from ._util import append_tool_stubs
from .prompts import DEFAULT_ATTRIBUTION_QUESTION


def _parse_attribution(text: str) -> str:
    """Parse three-way attribution from model response.

    Uses Inspect AI's ANSWER_PATTERN_LINE as the primary extractor,
    with bare keyword fallback for non-conforming responses.
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
    if "uncertain" in text_lower or "unsure" in text_lower:
        return "uncertain"
    if "self" in text_lower:
        return "self"
    return "uncertain"


@solver
def attribution_probe(
    question_template: str = DEFAULT_ATTRIBUTION_QUESTION,
    metadata_key: str = "attribution",
) -> Solver:
    async def solve(state: TaskState, generate) -> TaskState:
        # Store initial response for the scorer's spontaneous detection
        initial_response = state.output.completion if state.output else ""
        initial_tool_calls = None
        if state.output and state.output.message:
            initial_tool_calls = state.output.message.tool_calls

        if state.metadata is None:
            state.metadata = {}
        state.metadata["initial_response"] = initial_response
        state.metadata["initial_tool_calls"] = initial_tool_calls

        # Stub any tool calls from the main generation
        append_tool_stubs(state)

        # Ask the attribution question
        state.messages.append(ChatMessageUser(content=question_template))
        state = await generate(state)

        # Parse and store
        raw = state.output.completion if state.output else ""
        label = _parse_attribution(raw)
        state.metadata[metadata_key] = {"label": label, "raw_response": raw}

        return state

    return solve
