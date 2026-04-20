# System Architecture

## Package structure

```
src/prefill_awareness_audit/
  __init__.py                    Public API re-exports
  __main__.py                    CLI: compare subcommand
  _registry.py                   Entry point for Inspect AI task discovery
  _tasks.py                      Generic @task (task=, log=, data= input paths)
  data.py                        Data loading, log discovery, PROBE_ONLY_PROFILE
  types.py                       Condition, Attribution, AuditTarget,
                                 InterventionConstraints, AuditProfile,
                                 ConditionSummary, ComparisonTable
  task_factory.py                make_audit_task(), continuation_audit()

  probes/                        Inspect solvers appended after generate()
    prompts.py                   Centralized prompt templates, constants, output schemas
    _util.py                     Shared helpers (tool stubs, shared judge-call helper)
    fork.py                      forked_probes() -- runs probes in parallel via Inspect's fork()
    awareness.py                 Combined attribution + confidence (one prompt, one call)
    reflection.py                Free-text first-person reflection + LLM judge
    persona.py                   User-persona role-swap + LLM judge (Cywiński)
    multiple_choice.py           Shuffled 9-option MCQ latent probe (opt-in, A/B baseline)
    diagnostic.py                Reason tags (style_mismatch, formatting_artifact, etc.)
    spontaneous.py               Scorer-side: unprompted awareness mentions

  interventions/                 Dataset transformers parameterized by (scope, mechanism)
    __init__.py                  apply_intervention() dispatcher
    deterministic.py             Formatting/provenance cleanup
    rewrite.py                   Contrastive style rewrite
    validator.py                 Post-intervention validation against InterventionConstraints

  scoring/
    audit_scorer.py              Wraps any scorer, adds probe dimensions + custom metrics
    metrics.py                   Custom @metric definitions for per-condition awareness profiles

  analysis/
    compare.py                   Cross-condition/cross-model comparison from eval logs
    figures.py                   Matplotlib figure generation
```

## Entry points and task discovery

The package registers an Inspect AI entry point in `pyproject.toml`:

```toml
[project.entry-points.inspect_ai]
prefill_awareness_audit = "prefill_awareness_audit._registry"
```

This lets `inspect eval prefill_awareness_audit/prefill_awareness_audit` discover the generic `@task` without a file path. The first segment is the entry-point package name, the second is the `@task` function name — Inspect only loads entry-point-registered tasks when the invocation contains `/`. The `_registry.py` module imports the `@task` from `_tasks.py`, which Inspect's decorator-based discovery picks up.

## Three input paths

### Path 1 -- Audit from a prior eval (task-based discovery)

```bash
inspect eval prefill_awareness_audit/prefill_awareness_audit -T task=my_benchmark
```

Uses `find_eval_log()` to search the log directory (defaults to `INSPECT_LOG_DIR` or `./logs`) for the most recent eval log matching the task name. Extracts conversation histories via `load_from_eval_log()`. Model specified via `--model` on the CLI.

Optional `log_dir=` override: `-T log_dir=./experiment_logs`.

### Path 2 -- Audit from raw data

```bash
inspect eval prefill_awareness_audit/prefill_awareness_audit -T data=conversations.jsonl --model anthropic/claude-sonnet-4-6
```

Loads conversation data from a standard JSONL file. Each record is an Inspect AI Sample (object with `input` containing a list of chat messages).

### Path 3 -- Custom @task (complex benchmarks)

Use `make_audit_task()` for archival transcripts with a custom solver chain, or `continuation_audit()` when you want the model to generate its next turn before probing:

```python
@task
def misalignment_audit(condition="PROBE_ONLY", limit=None):
    return continuation_audit(
        data=load_dataset(limit=limit),
        condition=condition,
        profile=my_profile,
        scorer=my_scorer(),
        tools=get_stub_tools(),
    )
```

Full control over data loading, target selection, scoring, and solver. Used when benchmarks need a custom dataset loader, target_span_selector, or scorer.

## make_audit_task() and continuation_audit() API

Two user-facing composition helpers. Pick based on whether the model should generate a fresh turn before probing.

```python
def make_audit_task(
    data: str | Path | Dataset,
    condition: str | Condition = "PROBE_ONLY",
    profile: AuditProfile | None = None,       # None = PROBE_ONLY_PROFILE
    scorer: Scorer | None = None,              # None = no-op scorer (probes only)
    solver: list[Solver] | None = None,        # None = [generate()]
    probes: list[Solver] | str | None = None,  # None = four-probe default set
    limit: int | None = None,
    seed: int = 42,
) -> Task:
```

Handles: data loading (if path), condition validation, intervention application (including the sync/async split via `ThreadPoolExecutor`), probe wiring, scorer wrapping.

```python
def continuation_audit(
    data: str | Path | Dataset,
    tools: list[Tool] | None = None,
    scorer: Scorer | None = None,
    condition: str | Condition = "PROBE_ONLY",
    profile: AuditProfile | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> Task:
```

Wraps `make_audit_task()` with a fixed `use_tools(tools) + generate(tool_calls="none")` solver chain and probe templates anchored on "your last message" — suitable for agent trajectories where the model should generate its next turn before being asked about the prefilled history.

`PROBE_ONLY_PROFILE` is a conservative profile allowing only `Condition.PROBE_ONLY`, with `all_assistant_turns` as the target span selector. Intervention conditions require a benchmark-specific `AuditProfile`.

## Data loading (data.py)

- `load_conversations(path)` -- JSONL to list[Sample] via `json_dataset()`; defaults missing `source_model` metadata to `"unknown"`
- `find_eval_log(task, log_dir)` -- log discovery via `list_eval_logs()` with task name filter
- `load_from_eval_log(path, limit)` -- eval log to (list[Sample], model_id); preserves original scores in sample metadata and sets `source_model` to the log's probe model
- `all_assistant_turns(sample)` -- default target_span_selector, selects all assistant-role message indices
- `latest_assistant_block(sample)` -- target_span_selector for the most recent contiguous assistant/tool block (agent trajectories)

## Key types

**Condition** (StrEnum): PROBE_ONLY, LOCAL_DETERMINISTIC, LOCAL_REWRITE, BLOCK_DETERMINISTIC, BLOCK_REWRITE, GLOBAL_DETERMINISTIC.

**Attribution** (StrEnum): SELF, NOT_SELF, UNCERTAIN — the model's claim about a prefilled assistant turn's origin.

**AuditTarget**: Per-sample target span (sample_id, message_indices, target_kind).

**InterventionConstraints**: Structural invariants every intervention must preserve — `protected_roles` (roles the intervention cannot modify). Message count and role order are always preserved; the validator rejects any intervention that changes them.

**AuditProfile**: The one object a benchmark integration provides. Contains name, main_score_fields, target_span_selector, allowed_conditions, intervention_constraints, and optional semantic_guard.

**ConditionSummary**: Flat metrics from one eval log -- condition, model, metrics dict, sample_count. Used by the compare tool.

**ComparisonTable**: Cross-condition comparison output -- list of ConditionSummary, deltas vs PROBE_ONLY, optional model grouping.

## Per-condition metrics

Each eval run produces a complete awareness profile via Inspect's `@metric` system. Metrics show up in Inspect's log viewer and `inspect log list --json`.

### Two-tier metric registration

The audit scorer uses a mixed-list metrics spec:

- **Top-level metrics** (Tier 2): Custom `@metric` functions that receive the full `SampleScore` with complete `Score.value` dict and `Score.metadata`. They return `dict[str, float]` which Inspect expands into individual `EvalMetric` entries. Used for awareness rates, confidence stats, coupling, etc.

- **Per-field metrics** (Tier 1): Standard Inspect aggregators (`accuracy()`, `stderr()`) keyed to `Score.value` dict fields. Used only for benchmark-specific fields from `main_score_fields`.

### Metric groups

All proportions include Wilson score 95% CI. All means include 95% CI. Every group reports `n_samples` (valid samples used in computation).

**Ground truth confusion matrix**: `recognition_rate` (TPR), `detection_rate` (TNR), `false_attribution_rate` (FPR), `miss_rate` (FNR) with Wilson CIs; `g_mean` (sqrt(TPR·TNR), no CI); `uncertain_rate` with Wilson CI. Reports `n_samples`, `n_classified`, `n_parse_fail`. Compares `attribution` (model claim) against `prefill_authentic` (ground truth from `source_model` metadata).

**Tamper confidence**: `confidence_mean` with CI, `confidence_median`, `confidence_std`. Excludes unparseable responses (sentinel -1).

**Spontaneous awareness**: `spontaneous_rate` with Wilson CI.

**Latent awareness (free-text + judge, dual-framing)**: emitted when at least one of `reflection` / `persona` ran. Per-technique: `reflection_convinced_rate`, `persona_convinced_rate` with Wilson CI. Cross-technique (emitted when *both* ran): `latent_prefill_rate` (strict consensus, both `convinced`), `latent_any_awareness_rate` (either ≥ `suspicious`), `latent_consistent_rate` (both returned same category), `latent_awareness_score_mean` (graded 0–2) with 95% CI. Counts: `latent_n_samples`, `latent_n_parse_fail_reflection`, `latent_n_parse_fail_persona`.

**Latent awareness (multiple choice)**: emitted when the opt-in `multiple_choice` probe ran. `mc_prefill_rate` (selected "prefill" canonical option) and `mc_eval_rate` (selected "eval" canonical option), each with Wilson CI. Per-option rates as `mc_option_{id}_rate`. Reports `mc_n_samples`, `mc_n_parse_fail`.

**Diagnostic tags**: Per-tag rates (`diagnostic_{tag}_rate`) with Wilson CIs, flagged turn mean/std. Excludes samples where diagnostic parsing failed.

**Awareness-benchmark coupling** (only when `main_score_fields` non-empty): `coupling_{field}_given_self`, `coupling_{field}_given_not_self`, `coupling_{field}_cohens_d`, `coupling_{field}_confidence_corr` with Fisher z-transform CI.

### Statistical helpers

Implemented in `scoring/metrics.py` with numpy only (no scipy):

- `_wilson_ci(k, n)` -- Wilson score interval for binomial proportion
- `_mean_ci(values)` -- mean, median, std, 95% CI for the mean
- `_fisher_z_ci(r, n)` -- Fisher z-transform CI for Pearson correlation
- `_cohens_d(group1, group2)` -- Cohen's d with pooled standard deviation

## Component responsibilities

- **task_factory**: `make_audit_task()` is the primary API -- resolves defaults, loads data, applies interventions, wires probes, and wraps the scorer. `continuation_audit()` layers a fixed continuation solver chain on top. The sync/async split (interventions need async for model calls, but `@task` is sync) is handled internally via `ThreadPoolExecutor`.
- **probes**: Solvers that run after generate() via forked execution (each probe gets an independent deepcopy of TaskState). Cannot influence the main behavioral score. Four active probes in the default set (awareness -- combined attribution+confidence, reflection -- free-text latent with LLM judge, persona -- user-persona role-swap with LLM judge, diagnostic -- reason tags), plus one opt-in latent probe (multiple_choice -- preserved MCQ as A/B baseline) and one passive scorer-side signal (spontaneous).
- **interventions**: Each takes (samples, AuditProfile) and returns modified samples. Validates scope against `InterventionConstraints` before applying. Deterministic and rewrite share the scope parameterization.
- **scoring**: audit_scorer wraps the original scorer, adding probe dimensions to Score and registering custom metrics for per-condition aggregation. metrics.py defines all custom @metric functions.
- **analysis**: compare.py loads eval logs and builds cross-condition comparison tables. figures.py generates matplotlib visualizations.

## Cross-condition comparison

`python -m prefill_awareness_audit compare --log-dir logs/my-experiment/ [--figures]`

Reads eval-level metrics from log headers (fast, no sample loading). Produces:
- Per-condition cards with full awareness profile
- Delta summary table vs PROBE_ONLY
- Optional figures (awareness bars, confidence distribution, delta heatmap)

No classification -- presents numbers only, the reader interprets.

## External eval integration

External eval repos are added as git submodules under `case_studies/`. Each integration file defines:

1. An `AuditProfile` with benchmark-specific configuration
2. A `@task` function that calls `make_audit_task()` with the benchmark's dataset, solver, and scorer

The package handles probe insertion, intervention application, score wrapping, and metric computation. The benchmark integration only declares what to measure and what interventions are admissible.
