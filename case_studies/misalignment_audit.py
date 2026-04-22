"""Audit the UKGovernmentBEIS misalignment-continuation benchmark.

Example of how an external team would use the prefill-awareness-audit
package on their own benchmark. The only benchmark-specific pieces are
the dataset loader, benchmark scorer, stub tools, score-field names,
and the ``source_model`` label. Everything else comes from the package.

The probe chain mirrors ``continuation_audit`` inline: a ``use_tools +
generate`` solver pair lets the subject finish the conversation, then
the five-probe forked chain (awareness, reflection, persona, diagnostic,
multiple_choice) runs against the resulting transcript.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The submodule's internal modules use bare ``from evals.X import ...``.
# Add the submodule root to sys.path so those imports resolve.
_SUBMODULE_ROOT = Path(__file__).resolve().parent / "misalignment_continuation"
if str(_SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE_ROOT))

from evals.dataset import (  # noqa: E402
    DEFAULT_ANALYSIS_PATH,
    DEFAULT_DATASET_PATH,
    load_misalignment_dataset,
)
from evals.scorer import misalignment_continuation_scorer  # noqa: E402
from evals.stub_tools import get_stub_tools  # noqa: E402
from inspect_ai import Task, task  # noqa: E402
from inspect_ai.solver import generate, use_tools  # noqa: E402
from prefill_awareness_audit import (  # noqa: E402
    AuditProfile,
    Condition,
    InterventionConstraints,
    latest_assistant_block,
    make_audit_task,
)
from prefill_awareness_audit.probes import (  # noqa: E402
    awareness_probe,
    diagnostic_probe,
    multiple_choice_probe,
    persona_probe,
    reflection_probe,
)
from prefill_awareness_audit.probes.prompts import (  # noqa: E402
    REFLECTION_USER_CONTINUATION,
)

profile = AuditProfile(
    name="misalignment-continuation",
    main_score_fields=["aligned", "confession"],
    target_span_selector=latest_assistant_block,
    allowed_conditions=list(Condition),
    intervention_constraints=InterventionConstraints(),
)


@task
def misalignment_audit(
    condition: str = "PROBE_ONLY",
    limit: int | None = None,
) -> Task:
    """Audit one condition against the misalignment-continuation benchmark.

    Runs the full five-probe continuation chain: awareness + reflection +
    persona + diagnostic + multiple_choice. MC is opt-in in the generic
    @task but enabled by default here because the misalignment transcripts
    benefit from the menu-style elicitation channel (see
    validation-audit-2026-04-21 §"MC probe on the same misalignment
    transcripts").

    Args:
        condition: Audit condition name (PROBE_ONLY, LOCAL_DETERMINISTIC,
            LOCAL_REWRITE, BLOCK_DETERMINISTIC, BLOCK_REWRITE,
            GLOBAL_DETERMINISTIC).
        limit: Maximum number of samples to evaluate.
    """
    dataset = load_misalignment_dataset(
        DEFAULT_DATASET_PATH,
        limit=limit,
        analysis_path=DEFAULT_ANALYSIS_PATH,
        seed=42,
    )

    # Misalignment scenarios are synthetic — no real model generated them.
    for sample in dataset:
        sample.metadata = {**(sample.metadata or {}), "source_model": "synthetic"}

    solver_chain = [use_tools(get_stub_tools()), generate(tool_calls="none")]
    probes = [
        awareness_probe(continuation=True),
        reflection_probe(question_template=REFLECTION_USER_CONTINUATION),
        persona_probe(),
        diagnostic_probe(),
        multiple_choice_probe(),
    ]

    return make_audit_task(
        data=dataset,
        condition=condition,
        profile=profile,
        scorer=misalignment_continuation_scorer(),
        solver=solver_chain,
        probes=probes,
    )
