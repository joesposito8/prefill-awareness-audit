# System Architecture

## Package structure

```
src/prefill_awareness_audit/
  __init__.py                    Public API re-exports
  __main__.py                    CLI: compare subcommand
  types.py                      Condition, AuditTarget, RewritePolicy, AuditProfile,
                                 ConditionSummary, ComparisonTable
  task_factory.py                audit_task() -- single composition entry point

  probes/                        Inspect solvers appended after generate()
    prompts.py                   Centralized prompt templates, constants, output schemas
    _util.py                     Shared helpers (tool call stub insertion)
    fork.py                      forked_probes() -- runs probes in parallel via Inspect's fork()
    awareness.py                 Combined attribution + confidence (one prompt, one call)
    counterfactual.py            Latent awareness via indirect observation matching
    diagnostic.py                Reason tags (style_mismatch, formatting_artifact, etc.)
    spontaneous.py               Scorer-side: unprompted awareness mentions

  interventions/                 Dataset transformers parameterized by (scope, mechanism)
    __init__.py                  apply_intervention() dispatcher
    deterministic.py             Formatting/provenance cleanup
    rewrite.py                   Contrastive style rewrite
    validator.py                 Post-intervention validation against RewritePolicy

  scoring/
    audit_scorer.py              Wraps any scorer, adds probe dimensions + custom metrics
    metrics.py                   Custom @metric definitions for per-condition awareness profiles

  analysis/
    compare.py                   Cross-condition/cross-model comparison from eval logs
    figures.py                   Matplotlib figure generation
```

## Key types

**Condition** (StrEnum): BASELINE, PROBE_ONLY, LOCAL_DETERMINISTIC, LOCAL_REWRITE, BLOCK_DETERMINISTIC, BLOCK_REWRITE, GLOBAL_DETERMINISTIC.

**AuditTarget**: Per-sample target span (sample_id, message_indices, target_kind).

**RewritePolicy**: Permission-based -- booleans for each scope/mechanism pair, plus structural guards (protected_roles, preserve_message_count, preserve_role_order).

**AuditProfile**: The one object a benchmark integration provides. Contains name, main_score_fields, target_span_selector, allowed_conditions, rewrite_policy, and optional semantic_guard and baseline_control_loader.

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

**Awareness attribution**: `attribution_self_rate`, `attribution_not_self_rate`, `attribution_uncertain_rate` with Wilson CIs. Excludes parse failures (sentinel -2).

**Tamper confidence**: `confidence_mean` with CI, `confidence_median`, `confidence_std`. Excludes unparseable responses (sentinel -1).

**Spontaneous awareness**: `spontaneous_rate` with Wilson CI.

**Latent awareness**: `latent_awareness_mean` with CI, `latent_awareness_median`, `latent_awareness_std`. Excludes sentinels.

**Diagnostic tags**: Per-tag rates (`diagnostic_{tag}_rate`) with Wilson CIs, flagged turn mean/std. Excludes samples where diagnostic parsing failed.

**Awareness-benchmark coupling** (only when `main_score_fields` non-empty): `coupling_{field}_given_self`, `coupling_{field}_given_not_self`, `coupling_{field}_cohens_d`, `coupling_{field}_confidence_corr` with Fisher z-transform CI.

### Statistical helpers

Implemented in `scoring/metrics.py` with numpy only (no scipy):

- `_wilson_ci(k, n)` -- Wilson score interval for binomial proportion
- `_mean_ci(values)` -- mean, median, std, 95% CI for the mean
- `_fisher_z_ci(r, n)` -- Fisher z-transform CI for Pearson correlation
- `_cohens_d(group1, group2)` -- Cohen's d with pooled standard deviation

## Component responsibilities

- **task_factory**: Takes decomposed Inspect components + AuditProfile + Condition, returns a Task. Applies intervention to dataset, appends probes, wraps scorer.
- **probes**: Solvers that run after generate() via forked execution (each probe gets an independent deepcopy of TaskState). Cannot influence the main behavioral score. Three active probes (awareness -- combined attribution+confidence, counterfactual -- latent awareness, diagnostic -- reason tags), one passive scorer-side (spontaneous).
- **interventions**: Each takes (samples, AuditProfile) and returns modified samples. Validates scope against RewritePolicy before applying. Deterministic and rewrite share the scope parameterization.
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
2. A `@task` function that calls `audit_task()` with the benchmark's dataset, solver, and scorer

The package handles probe insertion, intervention application, score wrapping, and metric computation. The benchmark integration only declares what to measure and what interventions are admissible.
