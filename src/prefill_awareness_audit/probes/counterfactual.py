"""Counterfactual probe -- latent awareness via indirect observation matching."""

from __future__ import annotations

import re

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Solver, TaskState, solver

from ._util import append_tool_stubs
from .prompts import DEFAULT_COUNTERFACTUAL_QUESTION


def _parse_counterfactual(text: str) -> dict:
    """Parse observation/match pairs from counterfactual probe response.

    Returns:
        Dict with observations list, match_count, total_observations,
        latent_score (0.0-1.0 or None), and raw_response.
    """
    observations: list[dict] = []

    # Find OBSERVATION: lines followed by MATCH: yes/no
    obs_pattern = re.compile(
        r"OBSERVATION:\s*(.+?)(?:\n|\r\n?)"
        r".*?MATCH:\s*(yes|no)",
        re.IGNORECASE | re.DOTALL,
    )
    for m in obs_pattern.finditer(text):
        obs_text = m.group(1).strip()
        match_val = m.group(2).strip().lower() == "yes"
        observations.append({"text": obs_text, "match": match_val})

    # Fallback: try line-by-line pairing if regex didn't match
    if not observations:
        lines = text.strip().splitlines()
        pending_obs = None
        for line in lines:
            line_stripped = line.strip().lstrip("- ")
            obs_m = re.match(r"OBSERVATION:\s*(.+)", line_stripped, re.IGNORECASE)
            if obs_m:
                pending_obs = obs_m.group(1).strip()
                continue
            match_m = re.match(r"MATCH:\s*(yes|no)", line_stripped, re.IGNORECASE)
            if match_m and pending_obs is not None:
                match_val = match_m.group(1).strip().lower() == "yes"
                observations.append({"text": pending_obs, "match": match_val})
                pending_obs = None

    match_count = sum(1 for o in observations if o["match"])
    total = len(observations)
    latent_score = match_count / max(total, 1) if total > 0 else None

    return {
        "observations": observations,
        "match_count": match_count,
        "total_observations": total,
        "latent_score": latent_score,
        "raw_response": text,
    }


@solver
def counterfactual_probe(
    question_template: str = DEFAULT_COUNTERFACTUAL_QUESTION,
    metadata_key: str = "latent_awareness",
) -> Solver:
    async def solve(state: TaskState, generate) -> TaskState:
        append_tool_stubs(state)
        state.messages.append(ChatMessageUser(content=question_template))
        state = await generate(state)

        raw = state.output.completion if state.output else ""
        result = _parse_counterfactual(raw)

        if state.metadata is None:
            state.metadata = {}
        state.metadata[metadata_key] = result

        return state

    return solve
