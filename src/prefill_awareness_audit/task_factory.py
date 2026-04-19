"""Main composition point — builds an Inspect Task for a given audit condition."""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import asdict
from pathlib import Path

from inspect_ai import Task
from inspect_ai.dataset import Dataset, MemoryDataset, json_dataset
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Solver, TaskState, generate, use_tools
from inspect_ai.tool import Tool

from .interventions import apply_intervention
from .probes import (
    PROBE_FACTORIES,
    awareness_probe,
    diagnostic_probe,
    forked_probes,
    latent_probe,
)
from .probes.prompts import CONTINUATION_AWARENESS_QUESTION
from .scoring import audit_scorer
from .types import AuditProfile, Condition

_NO_INTERVENTION = frozenset({Condition.PROBE_ONLY})


def _noop_scorer() -> Scorer:
    """Create a no-op scorer that returns an empty value dict."""

    @scorer(metrics=[])
    def _inner() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return Score(value={}, answer="")

        return score

    return _inner()


def _resolve_probes(probes: list[Solver] | str) -> list[Solver]:
    """Resolve a comma-separated name string to a list of probe solvers."""
    if not isinstance(probes, str):
        return list(probes)
    names = [n.strip() for n in probes.split(",") if n.strip()]
    if not names:
        raise ValueError(
            f"probes string is empty; pass None for defaults or one of "
            f"{sorted(PROBE_FACTORIES)}"
        )
    unknown = [n for n in names if n not in PROBE_FACTORIES]
    if unknown:
        raise ValueError(
            f"Unknown probe name(s) {unknown}; valid names are "
            f"{sorted(PROBE_FACTORIES)}"
        )
    return [PROBE_FACTORIES[n]() for n in names]


def make_audit_task(
    data: str | Path | Dataset,
    condition: str | Condition = "PROBE_ONLY",
    profile: AuditProfile | None = None,
    scorer: Scorer | None = None,
    solver: list[Solver] | None = None,
    probes: list[Solver] | str | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> Task:
    """Build an Inspect Task for one audit condition.

    This is the single entry point for constructing audit tasks.  It resolves
    defaults, handles data loading, validates the condition, applies
    interventions, appends probes, and wraps the scorer.

    Args:
        data: Dataset, path to a JSONL file, or an Inspect Dataset.
        condition: Audit condition name or enum value.
        profile: Benchmark-specific audit configuration.  Defaults to
            ``PROBE_ONLY_PROFILE`` (PROBE_ONLY only).
        scorer: Benchmark scorer.  Defaults to a no-op scorer (probes only).
        solver: Benchmark solver chain.  Defaults to ``[generate()]``.
        probes: Override the probe list appended after the solver chain.
            Defaults to ``[awareness_probe(), latent_probe(), diagnostic_probe()]``.
            Accepts either a list of Solver instances (for programmatic use,
            e.g. a custom awareness prompt) or a comma-separated string of
            probe names drawn from :data:`prefill_awareness_audit.probes.PROBE_FACTORIES`
            — currently ``"awareness"``, ``"latent"``, ``"diagnostic"``. The
            string form is what the generic ``@task`` forwards from the CLI
            ``-T probes=...`` flag.
        limit: Maximum number of samples.
        seed: Random seed (stored in task metadata).

    Returns:
        A configured Inspect Task ready for ``inspect eval``.

    Raises:
        ValueError: If the condition is not in the profile's allowed_conditions,
            or if ``data`` is an unrecognised type.
    """
    # Resolve condition
    if isinstance(condition, str):
        cond = Condition[condition]
    else:
        cond = condition

    # Resolve profile
    if profile is None:
        from .data import PROBE_ONLY_PROFILE

        profile = PROBE_ONLY_PROFILE

    # Validate condition against profile
    if cond not in profile.allowed_conditions:
        raise ValueError(
            f"Condition {cond} is not in profile.allowed_conditions "
            f"({[c.value for c in profile.allowed_conditions]})"
        )

    # Resolve data to a Dataset
    if isinstance(data, (str, Path)):
        dataset: Dataset = json_dataset(str(data))
    elif isinstance(data, Dataset):
        dataset = data
    else:
        raise ValueError(
            f"data must be a path (str/Path) or an Inspect Dataset, got {type(data).__name__}"
        )

    # Apply limit
    if limit is not None:
        samples = list(dataset)[:limit]
        dataset = MemoryDataset(samples, name=getattr(dataset, "name", None))

    # Apply intervention for non-passthrough conditions
    if cond not in _NO_INTERVENTION:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            modified = pool.submit(
                asyncio.run,
                apply_intervention(list(dataset), cond, profile),
            ).result()
        dataset = MemoryDataset(modified, name=getattr(dataset, "name", None))

    # Build solver chain: user solvers + forked probes
    task_solver = list(solver) if solver is not None else [generate()]
    if probes is not None:
        task_probes = _resolve_probes(probes)
    else:
        task_probes = [awareness_probe(), latent_probe(), diagnostic_probe()]
    task_solver.append(forked_probes(task_probes))

    # Wrap scorer
    task_scorer = scorer if scorer is not None else _noop_scorer()
    task_scorer = audit_scorer(task_scorer, profile)

    return Task(
        dataset=dataset,
        solver=task_solver,
        scorer=task_scorer,
        metadata={
            "condition": cond.value,
            "profile": profile.name,
            "intervention_constraints": {
                k: v
                for k, v in asdict(profile.intervention_constraints).items()
                if isinstance(v, bool)
            },
            "custom_probes": probes is not None,
        },
    )


def continuation_audit(
    data: str | Path | Dataset,
    tools: list[Tool] | None = None,
    scorer: Scorer | None = None,
    condition: str | Condition = "PROBE_ONLY",
    profile: AuditProfile | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> Task:
    """Build an audit task where the model continues the conversation first.

    The model generates its own next assistant turn from the end of each
    sample's conversation, then the awareness probe asks about the
    preceding assistant messages using wording anchored on "your last
    message" (which now refers to the just-generated turn).

    Use this when you want the model to be a live participant in the
    session before being probed — for example, when auditing whether
    it recognises prefilled messages as not its own during a continued
    agent run.  For audits of archival transcripts without a fresh
    continuation, use :func:`make_audit_task` instead.

    Args:
        data: Dataset, path to a JSONL file, or an Inspect Dataset.
        tools: Tools to make available during the continuation. Typically
            stub tools from the underlying benchmark so the model can
            emit tool calls that look like it would have during the
            original run. ``tool_calls="none"`` is used, so the tools
            are not actually invoked.
        scorer: Benchmark scorer (wrapped with audit dimensions).
        condition: Audit condition name or enum value.
        profile: Benchmark-specific audit configuration.
        limit: Maximum number of samples.
        seed: Random seed (stored in task metadata).

    Returns:
        A configured Inspect Task ready for ``inspect eval``.
    """
    solver_chain: list[Solver] = []
    if tools:
        solver_chain.append(use_tools(tools))
    solver_chain.append(generate(tool_calls="none"))

    continuation_probes: list[Solver] = [
        awareness_probe(question_template=CONTINUATION_AWARENESS_QUESTION),
        latent_probe(),
        diagnostic_probe(),
    ]

    return make_audit_task(
        data=data,
        condition=condition,
        profile=profile,
        scorer=scorer,
        solver=solver_chain,
        probes=continuation_probes,
        limit=limit,
        seed=seed,
    )
