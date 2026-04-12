"""Misalignment-continuation case study — AuditProfile and @task definition.

Integration for the UKGovernmentBEIS/misalignment-continuation benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

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
    AuditTarget,
    Condition,
    RewritePolicy,
    make_audit_task,
)

if TYPE_CHECKING:
    from inspect_ai.dataset import Sample


def get_target_span(sample: Sample) -> AuditTarget:
    """Identify the latest contiguous assistant/tool block for intervention.

    Walks backwards from the end of ``sample.input`` to find the most recent
    block of assistant and tool messages (the last assistant interaction before
    the continuation point).
    """
    messages = sample.input
    if not isinstance(messages, list):
        raise TypeError("Expected sample.input to be a list of ChatMessages")

    n = len(messages)

    # Phase 1: skip trailing non-assistant messages (the continuation point)
    i = n - 1
    while i >= 0 and messages[i].role in ("user", "tool", "system"):
        i -= 1

    # Phase 2: collect contiguous assistant/tool block going backwards
    indices: list[int] = []
    while i >= 0 and messages[i].role in ("assistant", "tool"):
        indices.append(i)
        i -= 1

    indices.reverse()

    return AuditTarget(
        sample_id=sample.id or "",
        message_indices=indices,
        target_kind="assistant_block",
    )


profile = AuditProfile(
    name="misalignment-continuation",
    main_score_fields=["aligned", "confession"],
    target_span_selector=get_target_span,
    allowed_conditions=[
        Condition.BASELINE,
        Condition.PROBE_ONLY,
        Condition.LOCAL_DETERMINISTIC,
        Condition.LOCAL_REWRITE,
        Condition.BLOCK_DETERMINISTIC,
        Condition.BLOCK_REWRITE,
        Condition.GLOBAL_DETERMINISTIC,
    ],
    rewrite_policy=RewritePolicy(
        allow_block_deterministic=True,
        allow_block_rewrite=True,
        allow_global_deterministic=True,
    ),
)


@task
def misalignment_audit(
    condition: str = "BASELINE",
    limit: int | None = None,
) -> Task:
    """Run one audit condition against the misalignment-continuation benchmark.

    Args:
        condition: Audit condition name (e.g. BASELINE, PROBE_ONLY,
            LOCAL_DETERMINISTIC, LOCAL_REWRITE).
        limit: Maximum number of samples to evaluate.
    """
    dataset = load_misalignment_dataset(
        DEFAULT_DATASET_PATH,
        limit=limit,
        analysis_path=DEFAULT_ANALYSIS_PATH,
        seed=42,
    )

    return make_audit_task(
        data=dataset,
        condition=condition,
        profile=profile,
        scorer=misalignment_continuation_scorer(),
        solver=[use_tools(get_stub_tools()), generate(tool_calls="none")],
    )
