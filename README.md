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

Two paths, depending on your question:

### Path 1: "How aware is the model?" (generic, zero code)

Run any eval, then audit the log. The audit extracts conversation histories, attaches `source_model` ground truth from the log, and runs awareness probes.

```bash
# 1. Run your benchmark
inspect eval inspect_evals/bfcl -T categories=multi_turn_base \
    --model anthropic/claude-sonnet-4-6 --limit 20

# 2. Audit -- same model (authentic prefill)
inspect eval prefill_awareness_audit/prefill_awareness_audit -T task=bfcl \
    --model anthropic/claude-sonnet-4-6

# 3. Audit -- different model (non-authentic prefill)
inspect eval prefill_awareness_audit/prefill_awareness_audit -T task=bfcl \
    --model openai/gpt-4.1
```

Step 2 measures recognition (does the model know it wrote this?). Step 3 measures detection (does the model know it didn't?). The confusion matrix tells the whole story. See [case_studies/bfcl/](case_studies/bfcl/) for the full walkthrough.

You can also audit raw conversation data:

```bash
inspect eval prefill_awareness_audit/prefill_awareness_audit \
    -T data=conversations.jsonl \
    -T condition=PROBE_ONLY \
    --model anthropic/claude-sonnet-4-6
```

Each JSONL record should be an Inspect AI Sample with `input` (list of chat messages) and optional `metadata.source_model` (defaults to `"unknown"` if absent).

### Path 2: "Is awareness confounding my eval?" (custom @task)

When you need intervention conditions (LOCAL_DETERMINISTIC, LOCAL_REWRITE, etc.) and benchmark re-scoring, write a custom `@task` with `make_audit_task()`:

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

Run each condition, then compare:

```bash
inspect eval my_benchmark_audit.py -T condition=BASELINE
inspect eval my_benchmark_audit.py -T condition=PROBE_ONLY
inspect eval my_benchmark_audit.py -T condition=LOCAL_DETERMINISTIC
inspect eval my_benchmark_audit.py -T condition=LOCAL_REWRITE
```

If `PROBE_ONLY` shows high awareness but `LOCAL_DETERMINISTIC` drops it significantly, the benchmark is artifact-sensitive -- the model was reacting to formatting cues, not the eval's intended construct. See [case_studies/misalignment_audit.py](case_studies/misalignment_audit.py) for the reference example.

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
