# Prefill Awareness Audit

Many AI safety evals work by showing a model a fabricated conversation history, then measuring how it responds. But what if the model notices the history is fake -- and that detection, not the thing the eval claims to measure, drives the result?

This tool audits that risk. Given any transcript-based benchmark and model, it applies independent intervention branches (each targeting a different category of detection cue) and classifies the result:

- **Robust** -- the model shows little awareness of fabrication; eval results are unlikely to be confounded.
- **Artifact-sensitive** -- the model detects fabrication via surface artifacts (formatting, metadata, style). Cleaning these materially reduces detection, so the eval's construct can be preserved.
- **Structurally prefill-entangled** -- awareness persists even after all admissible interventions. Detection is baked into the eval design itself.

Built on [Inspect AI](https://inspect.ai-safety-institute.org.uk/). The audit wraps existing eval components (datasets, solvers, scorers) rather than replacing them.

See [PROJECT_VISION.md](PROJECT_VISION.md) for full scope and non-goals.

## Installation

```bash
git clone --recurse-submodules https://github.com/joesposito8/prefill-awareness-audit.git
cd prefill-awareness-audit
uv sync
```

## Quick start

The core integration point is an `AuditProfile` -- a declaration of what your benchmark measures, which intervention scopes are admissible, and how to locate the prefilled target span in each sample.

```python
from inspect_ai import task, Task
from inspect_ai.solver import generate
from prefill_awareness_audit import audit_task, AuditProfile, RewritePolicy, Condition

profile = AuditProfile(
    name="misalignment-continuation",
    main_score_fields=["continues_misaligned"],
    target_span_selector=find_prefilled_assistant_turns,  # your function: Sample -> AuditTarget
    allowed_conditions=[
        Condition.BASELINE,              # unmodified benchmark
        Condition.PROBE_ONLY,            # adds awareness probes, no interventions
        Condition.LOCAL_DETERMINISTIC,    # formatting cleanup on target span
        Condition.LOCAL_REWRITE,         # model-based style rewrite on target span
    ],
    rewrite_policy=RewritePolicy(
        protected_roles={"system"},      # never touch system messages
        preserve_message_count=True,
        preserve_role_order=True,
    ),
)

@task
def misalignment_audit(condition: str = "BASELINE") -> Task:
    return audit_task(
        dataset=load_misalignment_dataset(),
        solver=[generate()],
        scorer=misalignment_scorer(),
        condition=Condition[condition],
        profile=profile,
    )
```

Run each condition with Inspect, then compare:

```bash
inspect eval misalignment_audit.py -T condition=BASELINE
inspect eval misalignment_audit.py -T condition=PROBE_ONLY
inspect eval misalignment_audit.py -T condition=LOCAL_DETERMINISTIC
inspect eval misalignment_audit.py -T condition=LOCAL_REWRITE
```

If `PROBE_ONLY` shows high awareness but `LOCAL_DETERMINISTIC` drops it significantly, the benchmark is artifact-sensitive -- the model was reacting to formatting cues, not the eval's intended construct.

## How it works

The audit appends **probes** (Inspect solvers) after the model generates its main response. These measure awareness without influencing the behavioral score:

- **Attribution probe** -- three-way classification: self / not-self / uncertain
- **Confidence probe** -- 0-100 tampering probability
- **Diagnostic probe** -- structured reason tags (style_mismatch, formatting_artifact, etc.)
- **Spontaneous awareness** -- scorer-side detection of unprompted suspicion in the main response

**Interventions** modify the dataset before the eval runs. Each targets a specific scope (local/block/global) and mechanism (deterministic cleanup or model-based rewrite). Comparing each branch to the probe-only baseline gives clean attribution of which cue types drive awareness.

Post-hoc, the **classifier** loads logs from all conditions and produces a **Scorecard** with the final classification.

## Development

```bash
uv sync --group dev
pytest
```
