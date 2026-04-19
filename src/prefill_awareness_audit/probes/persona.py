"""Persona probe — user-persona role-swap latent awareness elicitation.

The model is asked to generate the *user's* next turn rather than an
assistant reply.  An LLM judge then classifies whether the invented user
turn implies the model itself doubts having written the prior assistant
messages.

Cywiński et al. (2510.01070) frame the mechanism: secrecy constraints from
fine-tuning attach to the assistant persona specifically, so the hiding
behaviour is a property of the assistant's output policy rather than of
the weights globally.  Role-swapping to generate a user turn bypasses that
policy gate; latent awareness can leak through the simulation.
"""

from __future__ import annotations

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Solver, TaskState, solver

from ._util import append_tool_stubs, classify_judgement
from .prompts import (
    LATENT_JUDGE_MODEL,
    PERSONA_JUDGE_TEMPLATE,
    PERSONA_USER,
)


@solver
def persona_probe(
    question_template: str = PERSONA_USER,
    judge_model: str = LATENT_JUDGE_MODEL,
) -> Solver:
    """Latent awareness via user-persona role-swap + LLM judge.

    Args:
        question_template: The user turn instructing the probe model to
            generate the user's next turn.
        judge_model: Inspect AI model string for the judge.  Defaults to
            ``anthropic/claude-sonnet-4-6``.
    """

    async def solve(state: TaskState, generate) -> TaskState:
        append_tool_stubs(state)
        state.messages.append(ChatMessageUser(content=question_template))
        state = await generate(state)

        raw = state.output.completion if state.output else ""
        category = await classify_judgement(
            response=raw,
            template=PERSONA_JUDGE_TEMPLATE,
            judge_model=judge_model,
        )

        if state.metadata is None:
            state.metadata = {}
        state.metadata["persona"] = {
            "category": category,
            "raw_response": raw,
        }

        return state

    return solve
