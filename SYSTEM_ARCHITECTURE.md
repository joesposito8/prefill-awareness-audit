# System Architecture

## Package structure

```
src/prefill_awareness_audit/
  __init__.py                    Public API re-exports
  types.py                      Condition, AuditTarget, RewritePolicy, AuditProfile, Scorecard
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
    audit_scorer.py              Wraps any scorer, adds probe dimensions
    classifier.py                Post-hoc: load logs, compute scorecard, classify

  analysis/
    report.py                    Scorecard tables and figures
```

## Key types

**Condition** (StrEnum): BASELINE, PROBE_ONLY, LOCAL_DETERMINISTIC, LOCAL_REWRITE, BLOCK_DETERMINISTIC, BLOCK_REWRITE, GLOBAL_DETERMINISTIC.

**AuditTarget**: Per-sample target span (sample_id, message_indices, target_kind).

**RewritePolicy**: Permission-based -- booleans for each scope/mechanism pair, plus structural guards (protected_roles, preserve_message_count, preserve_role_order).

**AuditProfile**: The one object a benchmark integration provides. Contains name, main_score_fields, target_span_selector, allowed_conditions, rewrite_policy, and optional semantic_guard and baseline_control_loader.

**Scorecard**: Computed post-hoc from all condition logs. Contains all score families and the classification label.

## Component responsibilities

- **task_factory**: Takes decomposed Inspect components + AuditProfile + Condition, returns a Task. Applies intervention to dataset, appends probes, wraps scorer.
- **probes**: Solvers that run after generate() via forked execution (each probe gets an independent deepcopy of TaskState). Cannot influence the main behavioral score. Three active probes (awareness -- combined attribution+confidence, counterfactual -- latent awareness, diagnostic -- reason tags), one passive scorer-side (spontaneous).
- **interventions**: Each takes (samples, AuditProfile) and returns modified samples. Validates scope against RewritePolicy before applying. Deterministic and rewrite share the scope parameterization.
- **scoring**: audit_scorer wraps the original scorer, adding probe dimensions to Score. classifier loads logs from all conditions and computes the Scorecard.
- **analysis**: Generates summary tables and figures from the Scorecard.

## External eval integration

External eval repos are added as git submodules under `case_studies/`. Each integration file defines:

1. An `AuditProfile` with benchmark-specific configuration
2. A `@task` function that calls `audit_task()` with the benchmark's dataset, solver, and scorer

The package handles probe insertion, intervention application, score wrapping, and post-hoc classification. The benchmark integration only declares what to measure and what interventions are admissible.
