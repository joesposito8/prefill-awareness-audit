"""Audit the UKGovernmentBEIS misalignment-continuation benchmark.

Example of how an external team would use the prefill-awareness-audit
package on their own benchmark. The only benchmark-specific pieces are
the dataset loader, benchmark scorer, stub tools, score-field names,
and the ``source_model`` label. Everything else comes from the package.
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
from prefill_awareness_audit import (  # noqa: E402
    AuditProfile,
    Condition,
    InterventionConstraints,
    continuation_audit,
    latest_assistant_block,
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

    return continuation_audit(
        data=dataset,
        condition=condition,
        profile=profile,
        scorer=misalignment_continuation_scorer(),
        tools=get_stub_tools(),
    )
