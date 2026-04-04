# Prefill Awareness Audit

Audit scaffold for transcript-based AI evals. Determines whether a benchmark's results are driven by what it claims to measure, or by the model detecting that its conversation history has been fabricated. Classifies each benchmark/model pair as **robust**, **artifact-sensitive**, or **structurally prefill-entangled**.

Built on [Inspect AI](https://inspect.ai-safety-institute.org.uk/) -- the audit wraps existing eval components (datasets, solvers, scorers) rather than replacing them.

See [PROJECT_VISION.md](PROJECT_VISION.md) for scope, outcomes, and non-goals.

## Installation

```bash
git clone --recurse-submodules https://github.com/joeyagreco/prefill-awareness-audit.git
cd prefill-awareness-audit
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Quick start

```python
from prefill_awareness_audit import audit_task, AuditProfile, RewritePolicy, Condition

profile = AuditProfile(
    name="my-benchmark",
    main_score_fields=["accuracy"],
    target_span_selector=get_target_span,
    allowed_conditions=[
        Condition.BASELINE, Condition.PROBE_ONLY,
        Condition.LOCAL_DETERMINISTIC, Condition.LOCAL_REWRITE,
    ],
    rewrite_policy=RewritePolicy(),
)

@task
def my_audit(condition: str = "BASELINE") -> Task:
    return audit_task(
        dataset=load_dataset(),
        solver=[use_tools(get_tools()), generate(tool_calls="none")],
        scorer=my_scorer(),
        condition=Condition[condition],
        profile=profile,
    )
```

Run with Inspect:

```bash
inspect eval my_audit.py -T condition=BASELINE
inspect eval my_audit.py -T condition=PROBE_ONLY
inspect eval my_audit.py -T condition=LOCAL_DETERMINISTIC
inspect eval my_audit.py -T condition=LOCAL_REWRITE
```

## Development

```bash
uv sync --group dev
pytest
```
