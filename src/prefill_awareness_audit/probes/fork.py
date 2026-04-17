"""Forked probe execution -- runs probes independently via Inspect's fork()."""

from __future__ import annotations

from inspect_ai.solver import Solver, TaskState, fork, solver


@solver
def forked_probes(probes: list[Solver]) -> Solver:
    async def solve(state: TaskState, generate) -> TaskState:
        if state.metadata is None:
            state.metadata = {}

        # Fork: each probe gets an independent deepcopy, runs in parallel
        forked_states = await fork(state, probes)

        # Merge probe-produced metadata back into the original state
        for fs in forked_states:
            if fs.metadata:
                state.metadata.update(fs.metadata)

        return state

    return solve
