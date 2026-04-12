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

Or install from pip:

```bash
pip install prefill-awareness-audit
```

## Quick start

### Audit a prior eval

Run your benchmark first, then audit it by task name:

```bash
# 1. Run your benchmark
inspect eval my_benchmark.py --model anthropic/claude-opus-4-6

# 2. Audit it -- finds the most recent log for "my_benchmark" automatically
inspect eval prefill_awareness_audit -T task=my_benchmark --model anthropic/claude-opus-4-6
```

The audit searches your log directory (defaults to `INSPECT_LOG_DIR` or `./logs`) for the most recent eval matching the task name, extracts the conversation histories, and runs awareness probes against them.

### Audit raw conversation data

Point the audit at a JSONL file of conversations:

```bash
inspect eval prefill_awareness_audit \
    -T data=conversations.jsonl \
    -T condition=PROBE_ONLY \
    --model anthropic/claude-opus-4-6
```

Each JSONL record should be an Inspect AI Sample with an `input` field containing a list of chat messages (`{"role": "...", "content": "..."}`).

### Custom @task for complex benchmarks

For benchmarks with custom data loading, target selection, or scoring, use `make_audit_task()` directly:

```python
from inspect_ai import task, Task
from inspect_ai.solver import generate
from prefill_awareness_audit import make_audit_task, AuditProfile, RewritePolicy, Condition

profile = AuditProfile(
    name="my-benchmark",
    main_score_fields=["continues_misaligned"],
    target_span_selector=find_prefilled_assistant_turns,  # your function: Sample -> AuditTarget
    allowed_conditions=[
        Condition.BASELINE,
        Condition.PROBE_ONLY,
        Condition.LOCAL_DETERMINISTIC,
        Condition.LOCAL_REWRITE,
    ],
    rewrite_policy=RewritePolicy(
        protected_roles=("system",),
        preserve_message_count=True,
        preserve_role_order=True,
    ),
)

@task
def my_benchmark_audit(condition: str = "BASELINE") -> Task:
    return make_audit_task(
        data=load_my_dataset(),
        condition=condition,
        profile=profile,
        solver=[generate()],
        scorer=my_scorer(),
    )
```

Run each condition with Inspect, then compare:

```bash
inspect eval my_benchmark_audit.py -T condition=BASELINE
inspect eval my_benchmark_audit.py -T condition=PROBE_ONLY
inspect eval my_benchmark_audit.py -T condition=LOCAL_DETERMINISTIC
inspect eval my_benchmark_audit.py -T condition=LOCAL_REWRITE
```

If `PROBE_ONLY` shows high awareness but `LOCAL_DETERMINISTIC` drops it significantly, the benchmark is artifact-sensitive -- the model was reacting to formatting cues, not the eval's intended construct.

## How it works

The audit appends **probes** (Inspect solvers) after the model generates its main response. These measure awareness without influencing the behavioral score:

- **Attribution probe** -- three-way classification: self / not-self / uncertain
- **Confidence probe** -- 0-100 tampering probability
- **Diagnostic probe** -- structured reason tags (style_mismatch, formatting_artifact, etc.)
- **Spontaneous awareness** -- scorer-side detection of unprompted suspicion in the main response

**Interventions** modify the dataset before the eval runs. Each targets a specific scope (local/block/global) and mechanism (deterministic cleanup or model-based rewrite). Comparing each branch to the probe-only baseline gives clean attribution of which cue types drive awareness.

Each eval run produces a complete awareness profile via Inspect's `@metric` system -- awareness rates, confidence distributions, coupling statistics, and diagnostic breakdowns all appear in Inspect's log viewer.

Post-hoc, the **compare** tool loads logs from all conditions and generates cross-condition and cross-model comparison tables:

```bash
python -m prefill_awareness_audit compare --log-dir logs/my-experiment/ --figures
```

## Development

```bash
uv sync --group dev
pytest
```
