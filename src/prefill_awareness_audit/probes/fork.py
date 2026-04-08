"""Forked probe execution -- runs probes independently via Inspect's fork()."""

from __future__ import annotations

from inspect_ai.solver import Solver, TaskState, fork, solver


@solver
def forked_probes(probes: list[Solver]) -> Solver:
    async def solve(state: TaskState, generate) -> TaskState:
        # Capture initial response before any probes modify state
        if state.metadata is None:
            state.metadata = {}
        state.metadata["initial_response"] = (
            state.output.completion if state.output else ""
        )
        state.metadata["initial_tool_calls"] = (
            state.output.message.tool_calls
            if state.output and state.output.message
            else None
        )

        # Fork: each probe gets an independent deepcopy, runs in parallel
        forked_states = await fork(state, probes)

        # Merge metadata from all probes back into original state
        for fs in forked_states:
            if fs.metadata:
                for key, value in fs.metadata.items():
                    if key not in ("initial_response", "initial_tool_calls"):
                        state.metadata[key] = value

        return state

    return solve
