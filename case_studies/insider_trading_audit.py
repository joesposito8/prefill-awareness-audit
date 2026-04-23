"""Audit Apollo Research's insider-trading scenarios.

Each scenario is a hand-crafted agentic transcript (Alpha trading agent
persona) with ~6-8 synthetic assistant turns plus a trailing user
"deception trigger". The model takes one authentic turn in response to
the trigger, then five probes run in parallel (awareness, reflection,
persona, diagnostic, multiple_choice). source_model="apollo_synthetic"
-- any self-attribution of the prefilled turns is a false positive.

Structurally equivalent to continuation_audit() but with the opt-in
multiple_choice probe appended to the default continuation probe set.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate

from prefill_awareness_audit import (
    AuditProfile,
    Condition,
    InterventionConstraints,
    all_assistant_turns,
    make_audit_task,
)
from prefill_awareness_audit.probes import (
    awareness_probe,
    diagnostic_probe,
    multiple_choice_probe,
    persona_probe,
    reflection_probe,
)
from prefill_awareness_audit.probes.prompts import REFLECTION_USER_CONTINUATION

_PROMPTS_ROOT = Path(__file__).resolve().parent / "insider_trading" / "prompts"

_PROFILE = AuditProfile(
    name="apollo-insider-trading",
    main_score_fields=[],
    target_span_selector=all_assistant_turns,
    allowed_conditions=list(Condition),
    intervention_constraints=InterventionConstraints(),
)


def _load_apollo_samples() -> list[Sample]:
    samples: list[Sample] = []
    for path in sorted(_PROMPTS_ROOT.rglob("*.json")):
        data = json.loads(path.read_text())
        rel = path.relative_to(_PROMPTS_ROOT)
        category = rel.parent.as_posix()
        samples.append(
            Sample(
                input=list(data["messages"]),
                id=str(rel.with_suffix("")),
                metadata={
                    "source_model": "apollo_synthetic",
                    "apollo_category": "root" if category == "." else category,
                    "apollo_original_model": data.get("model"),
                },
            )
        )
    return samples


@task
def insider_trading_audit(
    condition: str = "PROBE_ONLY",
    limit: int | None = None,
) -> Task:
    """Audit one condition against the Apollo insider-trading scenarios.

    Args:
        condition: Audit condition name (PROBE_ONLY, LOCAL_DETERMINISTIC,
            LOCAL_REWRITE, BLOCK_DETERMINISTIC, BLOCK_REWRITE,
            GLOBAL_DETERMINISTIC).
        limit: Maximum number of samples to evaluate.
    """
    dataset = MemoryDataset(_load_apollo_samples(), name="apollo-insider-trading")
    return make_audit_task(
        data=dataset,
        condition=condition,
        profile=_PROFILE,
        solver=[generate(tool_calls="none")],
        probes=[
            awareness_probe(continuation=True),
            reflection_probe(question_template=REFLECTION_USER_CONTINUATION),
            persona_probe(),
            diagnostic_probe(),
            multiple_choice_probe(),
        ],
        limit=limit,
    )
