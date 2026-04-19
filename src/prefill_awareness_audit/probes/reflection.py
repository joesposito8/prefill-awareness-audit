"""Reflection probe — first-person free-text latent awareness elicitation.

The model is asked to describe the prior assistant messages neutrally (no
priming of the authorship axis).  An LLM judge then classifies whether the
reflection expresses doubt that the model itself wrote those messages.

Grounded in AuditBench's finding that scaffolded black-box tooling
outperforms direct probes, combined with Cywiński et al.'s result that
free-text elicitation is stronger than forced-choice menus.  Neutral wording
chosen deliberately over leading phrasings to avoid demand-characteristic
confounds — see ``docs/design-decisions.md §4.9``.
"""

from __future__ import annotations

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Solver, TaskState, solver

from ._util import append_tool_stubs, classify_judgement
from .prompts import (
    LATENT_JUDGE_MODEL,
    LATENT_JUDGE_SCHEMA,
    REFLECTION_JUDGE_TEMPLATE,
    REFLECTION_USER,
)


@solver
def reflection_probe(
    question_template: str = REFLECTION_USER,
    judge_model: str = LATENT_JUDGE_MODEL,
) -> Solver:
    """Latent awareness via first-person neutral reflection + LLM judge.

    Args:
        question_template: The user turn asking the probe model for its
            reflection.  Defaults to the canonical neutral phrasing; use
            ``REFLECTION_USER_CONTINUATION`` for ``continuation_audit``.
        judge_model: Inspect AI model string for the judge.  Defaults to
            ``anthropic/claude-sonnet-4-6``.  Override this (e.g.
            ``"openai/gpt-5-mini"``) if the caller doesn't have an
            Anthropic API key.
    """

    async def solve(state: TaskState, generate) -> TaskState:
        append_tool_stubs(state)
        state.messages.append(ChatMessageUser(content=question_template))
        state = await generate(state)

        raw = state.output.completion if state.output else ""
        category = await classify_judgement(
            response=raw,
            template=REFLECTION_JUDGE_TEMPLATE,
            judge_model=judge_model,
            schema=LATENT_JUDGE_SCHEMA,
        )

        if state.metadata is None:
            state.metadata = {}
        state.metadata["reflection"] = {
            "category": category,
            "raw_response": raw,
        }

        return state

    return solve
