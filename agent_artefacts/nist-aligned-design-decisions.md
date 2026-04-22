# NIST-Aligned Design Decisions — Prefill Awareness Audit

A set of design decisions grounding PAA's experimental design in NIST
(AI RMF, TN 1297) and NIST-adjacent metrology (JCGM GUM, JCGM VIM,
ISO/IEC 17025) measurement-science guidance. Each decision either
**ratifies** an existing PAA choice in NIST vocabulary, **adopts** a new
practice, or **opens** a question for Joey.

Companion to `docs/design-decisions.md`. Format matches that document:
Decision / Alternatives considered / Why, with additional PAA-artifact
and NIST-basis fields.

**Terminology note** (consulted before reading §5). NIST does not use
"construct validity" as a term of art. The VIM equivalent is
*validation of a measurement procedure* (VIM 2.8); the AI RMF
equivalent is *"valid and reliable"* under MEASURE 2.5. The
construct-validity term is reserved below for references to the
AI-measurement literature (Jacobs & Wallach 2021) where PAA's thesis
naturally lives.

**Status vocabulary**: *ratify* (existing PAA choice, ratified in NIST
language), *adopt* (new practice), *open* (question for Joey).

---

## Executive summary

Sixteen decisions, one bullet each, plus five open questions.

1. **Publish the PAA metrological model with a Validation section** as a
   single repo doc (`docs/metrology.md`), covering measurands,
   measurement equations, instruments, reference-material chain,
   influence quantities, and the method-validation ledger. (§1.1)
2. **Write explicit measurand specifications** (VIM 2.3) for each of the
   six PAA measurands, including population generalisation limits. (§1.2)
3. **Ratify `source_model` metadata as a VIM 2.41 traceability artifact**
   and document the chain-of-custody failure modes. (§1.3)
4. **Ratify independent intervention branches as a factor-at-a-time
   experimental design** (NIST/SEMATECH e-Handbook Ch. 5.3). (§2.1)
5. **Adopt NIST reproducibility vocabulary** in `compare.py` output and
   in the LessWrong post: cross-condition deltas are *reproducibility
   under changed conditions* (VIM 2.24); cross-run same-condition is
   *intermediate precision* (VIM 2.22); within-run fork is
   *repeatability* (VIM 2.20). (§2.2)
6. **Declare the influence-quantity register** (VIM 2.52) and commit to
   a *standard conditions of measurement* statement that includes
   provider-default sampling parameters. (§2.3)
7. **Declare a Type A / Type B uncertainty budget** (TN 1297 §2;
   GUM §4.2, §4.3) per metric family, with a worked example and an
   explicit position on expanded uncertainty U = k·u_c. (§3.1)
8. **Adopt a GUM §5.1 closed-form combined uncertainty for g-mean**,
   conditional on the realised confusion-matrix margins. (§3.2)
9. **Add a Cohen's d confidence interval** using the Hedges & Olkin
   large-sample variance, computed numpy-only. (§3.3)
10. **Defer multiplicity control** for coupling statistics to an open
    question until a multi-field benchmark ships; current
    single-field operation is a no-op. (§3.4)
11. **Adopt a small PAA check-standard reference set** (VIM 5.13;
    ISO 17025 §7.7). Sized for qualitative drift detection, not power
    estimation. (§4.1)
12. **Calibrate parsers against the check-standard set** (VIM 2.39) and
    record per-model parser accuracy. (§4.2)
13. **Require per-scorecard environment capture in task metadata**:
    exact fields enumerated; no post-hoc reconstruction. (§4.3)
14. **Commit to a `compare.py` delta-reading convention** — deltas are
    signal only when their absolute value exceeds the larger of the two
    endpoints' combined-standard-uncertainty halfwidths. (§4.4)
15. **Position PAA against AI RMF MEASURE 2.13 and MEASURE 2.5** —
    meta-measurement and "valid and reliable" — with a cadence tied to
    §4.1's reference-set rerun. State the GenAI Profile risk mapping by
    name. (§5.1)
16. **Document NIST coverage gaps explicitly** (LLM-as-judge uncertainty,
    capability reference materials, compound-metric uncertainty
    reporting, prompt-as-procedure versioning) and publish PAA's own
    conventions for each. (§6.1)

Open questions deferred to Joey (§7): three-outcome classifier rule;
reference-set directory location; inter-judge variance study; ISO 17043
PT positioning; multiplicity control sequencing.

---

## 1. Measurement model and traceability

### 1.1 Publish the PAA metrological model, with a Validation section, as a single repo document

**Decision:** Create `docs/metrology.md` recording, in VIM/GUM-aligned
language, the six PAA measurands (M1–M6), the measurement equations
where closed-form compounds exist (g-mean, latent graded mean, Cohen's
d, Pearson r), the measurement procedure, the instrument inventory (six
probes + the spontaneous scorer), the reference-material chain, and the
influence-quantity register. **Include a Validation section** (per
ISO 17025 §7.2.2) that lists each non-standard method with intended
use, validation test set, pass/fail criteria, outcome, date, and known
limitations. ISO 17025 treats metrology reference and validation record
as distinct documents; for a solo-author project we merge them and
flag the boundary in the doc.

**PAA artifact:** new file `docs/metrology.md`; cross-references to
`src/prefill_awareness_audit/scoring/metrics.py`,
`src/prefill_awareness_audit/probes/`, `src/prefill_awareness_audit/data.py`,
`src/prefill_awareness_audit/scoring/audit_scorer.py`.

**NIST basis:** VIM 2.3, 2.6, 2.8; GUM §3, §8; AI RMF MEASURE 2.1;
ISO 17025 §7.2.2.

**Alternatives considered:**
- Keep the model implicit in docstrings and architecture docs —
  rejected because decisions below need a stable target, and implicit
  models drift silently.
- Embed metrological definitions in `docs/design-decisions.md` —
  rejected because that document is a development-phase narrative, not
  a measurement reference.
- Two separate documents (`docs/metrology.md` +
  `docs/method-validation-record.md`) per ISO 17025 convention —
  rejected for a solo-author project; keeping them in sync is a
  foreseeable maintenance failure.

**Tradeoff:** committing to a published measurand model raises the cost
of future redesigns. Changing a measurand mid-project forces re-running
validation and revising any public claim (LessWrong post, case-study
scorecards) that cited the old definition. The metrology doc becomes a
contract.

**Status:** adopt.

### 1.2 Explicit measurand specifications (VIM 2.3)

**Decision:** Each measurand M1–M6 gets a one-paragraph specification
stating (a) the quantity, (b) the kind of quantity (binomial proportion
/ bounded real / ordinal score / effect size), (c) the state of the
phenomenon under which it is defined (probe model, transcript population,
probe question template, judge model where applicable), and (d) the
population to which it generalises. The direct attribution rate (M1),
for example, is defined only with respect to a specified probe model,
probe question template, and transcript population; it does not
generalise to unseen probe templates without further evidence.

**PAA artifact:** `docs/metrology.md §measurands`; docstrings on
`ground_truth_metrics`, `confidence_stats`, `spontaneous_rate_metric`,
`latent_rates`, `multiple_choice_rates`, `coupling_metrics` in
`src/prefill_awareness_audit/scoring/metrics.py`.

**NIST basis:** VIM 2.3; VIM 2.3 Note 2 (specification requires state
of the phenomenon).

**Alternatives considered:**
- Let docstrings serve — rejected: docstrings describe implementation,
  not measurand semantics.
- One monolithic "PAA measures prefill awareness" statement — rejected:
  six distinct measurands with different populations cannot share a
  specification without muddling inference.

**Tradeoff:** some specifications will surface scope disagreements
(e.g. `spontaneous_rate` almost certainly does not generalise across
domains with a fixed phrase list — see §4.2 calibration). Publishing the
specification makes the disagreement visible rather than latent.

**Status:** adopt.

### 1.3 Ratify `source_model` as a VIM 2.41 traceability artifact

**Decision:** `source_model` is PAA's traceability artifact — the
"unbroken documented chain" from a measurement result back to the
actual provenance of the prefilled turns. Ratify the existing design,
add NIST vocabulary, and document the chain's limits: for eval-log
inputs the chain terminates at Inspect's `eval.model` canonicalisation;
for JSONL inputs it terminates in the benchmark integrator's
attestation; `"unknown"` samples are explicitly outside the traceability
chain and correctly excluded from the confusion matrix.

**PAA artifact:** `src/prefill_awareness_audit/data.py:28-34`
(defaulting); `src/prefill_awareness_audit/data.py:129`
(`load_from_eval_log` setting `source_model` from `log.eval.model`);
`src/prefill_awareness_audit/scoring/audit_scorer.py:93-95`
(`prefill_authentic = source == probe`);
`src/prefill_awareness_audit/scoring/metrics.py:174-176` (unknown
exclusion from confusion matrix denominators).

**NIST basis:** VIM 2.41 (metrological traceability); ISO 17025 §6.5.

**Alternatives considered:**
- Treat attribution as self-consistent without external ground truth —
  rejected when the confusion-matrix design was adopted (context
  `d07d2a5f`).
- Require a cryptographic attestation on `source_model` — rejected as
  disproportionate for research use.

**Tradeoff:** the chain is only as strong as the benchmark integrator's
attestation. For synthetic datasets where a human authored the
transcript, there is no way to distinguish "human wrote this" from
"human paraphrased a model's output." The Validation section of
`docs/metrology.md` (§1.1) treats this as a documented limitation, not
a defect to patch — which means downstream claims about
`false_attribution_rate` on human-authored transcripts must name the
attestation assumption explicitly.

**Status:** ratify.

---

## 2. Experimental design vocabulary

### 2.1 Ratify independent intervention branches as a factor-at-a-time design

**Decision:** `docs/design-decisions.md §1.2` (independent intervention
branches, not cumulative ladder) is ratified in NIST/SEMATECH terms as
a *factor-at-a-time experimental design*: each condition varies one
factor (scope × mechanism cell) against the shared PROBE_ONLY baseline,
enabling clean attribution. Interaction effects are deliberately not
estimated; the design trades the ability to detect interactions for
unambiguous main-effect attribution.

**PAA artifact:** `src/prefill_awareness_audit/types.py` (Condition
StrEnum); `docs/design-decisions.md §1.2`;
`src/prefill_awareness_audit/analysis/compare.py` (per-condition vs
PROBE_ONLY deltas).

**NIST basis:** NIST/SEMATECH e-Handbook of Statistical Methods,
Chapter 5.3 (Choosing an experimental design); specifically the
one-factor-at-a-time (OFAT) family and its tradeoffs with fractional
factorials.

**Alternatives considered:**
- Cumulative / stacked interventions — rejected in `docs/design-decisions.md §1.2` because deltas would
  confound with previously-applied interventions.
- Fractional factorial design over (scope × mechanism) — rejected
  because the conditions are not statistically symmetric (LOCAL is
  admissible by every profile; BLOCK and GLOBAL are profile-specific)
  and because interaction detection is not worth the extra runs for the
  probe-awareness question.

**Tradeoff:** interaction effects (e.g. "does LOCAL_REWRITE + BLOCK
interact?") are unmeasurable by construction. This is an accepted
limitation — PAA's thesis is about main-effect attribution of
awareness to cue types, not about interaction geometry.

**Status:** ratify.

### 2.2 NIST reproducibility vocabulary

**Decision:** `compare.py` output and the LessWrong post use NIST
precision vocabulary explicitly:
- Same run, forked probes on the same sample → **repeatability
  conditions** (VIM 2.20); variance within is instrument sampling
  noise.
- Same pipeline, same model version, different run / time →
  **intermediate precision** (VIM 2.22); captures API stochasticity and
  any drift.
- Different `Condition` (intervention), same model → **reproducibility
  under changed conditions** (VIM 2.24); each intervention changes the
  measurement system.
- Different probe model, same transcript → **reproducibility under
  changed operator/instrument** (VIM 2.24).

The 42-run audit matrix finding (context `9eb0786f`: model identity
dominates over intervention condition) is re-framed as an
intermediate-precision observation: PAA's measurement reproducibility
across probe models is lower than its reproducibility across
interventions. This is a property of the population (models differ more
from each other than interventions differ from each other), not a PAA
defect.

**PAA artifact:** `src/prefill_awareness_audit/analysis/compare.py`
(output formatting); public docs; LessWrong post.

**NIST basis:** VIM 2.20 / 2.21 (repeatability); VIM 2.22 / 2.23
(intermediate precision); VIM 2.24 / 2.25 (reproducibility).

**Alternatives considered:**
- Keep ad-hoc "intervention delta" vocabulary — rejected because it
  conflates three precision conditions into one.
- Adopt a single umbrella term like "between-group variance" — rejected
  because the VIM three-way distinction separates nuisance variance
  from designed variance from operator variance.

**Tradeoff:** the LessWrong audience is statistics-literate but not
metrology-literate. VIM vocabulary is dense and risks reading as
jargon. Mitigation: the post's body uses plain-language analogues
("different model runs," "different interventions") with a footnote
to the VIM clause on first use. Readers who want the NIST vocabulary
get it; readers who don't are unobstructed.

**Status:** adopt.

### 2.3 Influence-quantity register and standard conditions

**Decision:** `docs/metrology.md` publishes a table of every influence
quantity affecting PAA measurements, with (a) status (controlled /
captured / uncontrolled), (b) current handling, (c) order-of-magnitude
likely effect, (d) whether a near-term study is planned. Quantities
include: probe model identity, judge model identity, judge prompt
template, probe question template, random seed, intervention condition,
sample size, parser fallback behaviour, spontaneous-phrase list, tier-2
context window, tool-stub insertion, between-run stochasticity, and
sampling parameters (temperature / top-p / max-tokens). Additionally
publish a **standard conditions of measurement** statement (VIM 2.24
Note) declaring that PAA accepts provider-default sampling parameters
unless overridden and requires the active parameters to be recorded per
scorecard; setting them explicitly is not required, but logging them
*is*.

**PAA artifact:** `docs/metrology.md`; cross-references to
`src/prefill_awareness_audit/probes/prompts.py:135`
(`LATENT_JUDGE_MODEL`), `src/prefill_awareness_audit/probes/spontaneous.py:19`
(tier-2 context window 300 chars), `src/prefill_awareness_audit/_tasks.py:21`
and `src/prefill_awareness_audit/task_factory.py:74`, `:192` (seed=42
default), Inspect task metadata (provider defaults).

**NIST basis:** VIM 2.52 (influence quantity); VIM 2.24 Note (standard
conditions of measurement); GUM §3.

**Alternatives considered:**
- List only controlled factors — rejected: uncontrolled influences are
  where measurement claims get falsified.
- Require temperature=0 for all probes — rejected: forces provider
  behaviour that several providers implement as "near-deterministic"
  rather than "deterministic," and strips the natural stochasticity the
  auditor may want to characterise. Recording the value is sufficient;
  forcing it is overreach.
- Put this inside the uncertainty budget — rejected: the register is
  consumed by readers diagnosing failure modes; the budget is consumed
  by readers quantifying uncertainty. Different audiences.

**Tradeoff:** publishing the list of uncontrolled influences is
*adversarial* — it gives reviewers a checklist. That's the right cost
to pay; NIST measurement culture treats disclosed uncertainty as
stronger than hidden certainty. Separately, the sampling-parameter
logging requires a small addition to task metadata (§4.3).

**Status:** adopt.

---

## 3. Uncertainty budget

### 3.1 Type A / Type B uncertainty budget per metric family

**Decision:** Each metric family gets a one-page uncertainty budget in
`docs/metrology.md §uncertainty-budget` listing input quantities, Type
A or Type B classification (TN 1297 §2), and combined standard
uncertainty u_c(y) where a closed form exists. Type B components are
reported as bounds rather than measured values where empirical
characterisation is out of scope. Worked example for the ground-truth
confusion matrix family, to set the template:

| Input quantity | Source | Type | Standard uncertainty u(x) |
|---|---|---|---|
| per-rate sampling | Wilson CI on the rate | A | Wilson halfwidth / 1.96 (conservative) |
| judge-model bias | fixed judge, unmeasured variation across judge choices | B | bounded by prior judge-agreement literature (e.g. Zheng et al. 2023); order-of-magnitude only |
| prompt-template variation | fixed default prompts | B | bounded by ad-hoc single-model sweeps; order-of-magnitude only |
| parser fallback error | heuristic multi-tier parse chain | B | estimated from `n_parse_fail` counts plus a calibration sweep (§4.2) |
| between-run stochasticity | provider-default sampling parameters | A (in principle) | not measured; bound from §4.1 check-standard reruns |

Expanded uncertainty U = k·u_c with k=2 (≈95% coverage for
approximately normal distributions) is **not** reported alongside
Wilson by default. Wilson already approximates 95% coverage for
proportions; adding U columns doubles reporting volume for marginal
benefit. The budget supports k=2 reporting on demand — if a downstream
consumer requires it, the machinery is in place.

**PAA artifact:** `docs/metrology.md §uncertainty-budget`; helper
comments near each metric function in
`src/prefill_awareness_audit/scoring/metrics.py`.

**NIST basis:** TN 1297 §2 (Type A / Type B); TN 1297 §6 (expanded
uncertainty, coverage factor k); GUM §4.2, §4.3, §6; ISO 17025 §7.6.

**Alternatives considered:**
- Report only Type A (sampling) — rejected: judge variance, prompt
  variance, and parser error are real but hidden.
- Empirically characterise every Type B component up-front — rejected:
  disproportionate for a research tool; TN 1297 explicitly permits
  Type B bounds from "prior data, experience, judgment."
- Default to expanded uncertainty reporting (U = k·u_c) alongside
  Wilson — rejected here (see above); retained as an on-demand option.

**Tradeoff:** many Type B entries start as order-of-magnitude bounds
rather than measured values. The budget is revisable rather than
authoritative — honest, but readers will want to know which rows are
measured and which are bounded. The table's Type column is the tell.

**Status:** adopt.

### 3.2 GUM §5.1 closed-form u_c for g-mean

**Decision:** Extend `ground_truth_metrics` to compute combined standard
uncertainty for g-mean using GUM §5.1 for uncorrelated input quantities:

    Var(g) = (p₂ / 2g)² · Var(p₁) + (p₁ / 2g)² · Var(p₂)
    u_c(g) = √Var(g)

where p₁ = detection_rate = TN/(TN+FP), p₂ = recognition_rate =
TP/(TP+FN), g = √(p₁·p₂), and u(pᵢ) are Wilson-derived standard
uncertainties (half-width / 1.96, taking the larger side of the
asymmetric Wilson interval). The two rates are computed on **disjoint
subsamples** — p₁ over non-authentic samples, p₂ over authentic samples
— so conditional on the realised (n_non_authentic, n_authentic)
margins the two Wilson estimators are independent and the uncorrelated
formula holds. The conditioning is explicit in the metrology doc.
Report `g_mean_u_c` alongside `g_mean`.

**PAA artifact:**
`src/prefill_awareness_audit/scoring/metrics.py:139-248`
(`ground_truth_metrics`).

**NIST basis:** GUM §5.1 (combined standard uncertainty, uncorrelated
input quantities); TN 1297 §5.

**Alternatives considered:**
- Bootstrap u_c — rejected: the underlying Bernoulli sampling is cheap
  to propagate analytically, and the closed form is more transparent.
- Wilson-style direct interval on g-mean — rejected: no closed-form
  Wilson exists for products of proportions.

**Tradeoff:** small code addition. The u_c value is only as good as the
independence assumption — which is exact conditional on the margins but
not unconditionally. The metrology-doc entry states the conditioning.

**Status:** adopt (implementation in a follow-up task).

### 3.3 Cohen's d confidence interval

**Decision:** Add a CI to Cohen's d in `coupling_metrics` using the
large-sample Hedges & Olkin variance approximation, computed numpy-only:

    Var(d) = (n₁ + n₂) / (n₁ · n₂) + d² / (2 · (n₁ + n₂ − 2))
    SE(d) = √Var(d)
    CI₉₅(d) = d ± 1.96 · SE(d)

Report `coupling_{field}_cohens_d_ci_lo` and `_ci_hi` alongside the
point estimate. The large-sample approximation is used rather than a
noncentral-t CI because (a) it is numpy-only, matching PAA's "no scipy
in metrics" constraint, and (b) PAA's typical per-group n ≥ 20 in the
coupling numerator, where the approximation tracks the noncentral-t
interval to within the second decimal. Document this choice and its
breakdown for small n in the uncertainty-budget entry.

**PAA artifact:** `src/prefill_awareness_audit/scoring/metrics.py:637`
(Cohen's d currently returned without CI); `_cohens_d_ci` helper to
add in `metrics.py` statistical-helpers block.

**NIST basis:** ISO 17025 §7.8.3 (reports include uncertainty where
relevant); GUM §8; Hedges, L. V. & Olkin, I. (1985), Ch. 5.

**Alternatives considered:**
- Cohen's d without CI — rejected: effect-size decisions shouldn't
  be made without dispersion; currently a silent gap.
- Noncentral-t CI (Cumming & Finch 2001) — rejected for the scipy
  dependency; may revisit if scipy enters the dependency tree.
- Bootstrap CI — rejected: adds per-sample compute and is less
  transparent than the closed-form approximation.

**Tradeoff:** the approximation degrades for small n per group. For the
current misalignment case study (n_samples ≥ 20 per condition), the
approximation is adequate. Smaller benchmarks are possible — the
metrology doc must flag "small-n CI unreliable" once n per group
drops below ~10.

**Status:** adopt (implementation in a follow-up task).

### 3.4 Defer multiplicity control to open question

**Decision:** No FDR / Bonferroni correction is adopted now for
coupling across `main_score_fields`. The current first case study
(misalignment-continuation) declares a single main score field, so
multiplicity is a no-op. When the first multi-field benchmark ships,
the operational question ("adjust CI halfwidths? compute p-values and
adjust those?") can be made concretely. Treat this as §7.5 open.

**PAA artifact:** `src/prefill_awareness_audit/scoring/metrics.py:580-680`
(`coupling_metrics`); no change now; documented as a known gap in
`docs/metrology.md §uncertainty-budget`.

**NIST basis:** GUM §8 (reporting assumptions); ISO 17025 §7.8.3.

**Alternatives considered:**
- Adopt BH FDR per family per condition now — rejected: current output
  is CI-only; FDR on CIs requires either computing p-values or adopting
  simultaneous-inference intervals, and neither pays off for the
  single-field case.
- Adopt Bonferroni — rejected: overly conservative for correlated
  benchmark scores.

**Tradeoff:** the first multi-field benchmark ships without multiplicity
protection and may display nominal coverage where the simultaneous
coverage is lower. The metrology doc warns of this. The decision is
to pay that cost once, in exchange for not adopting a framework that
may be wrong for the eventual shape of the data.

**Status:** open (deferred; §7.5).

---

## 4. Reference material, calibration, records

### 4.1 A PAA check-standard reference set

**Decision:** Create a small curated set of transcripts with adjudicated
authenticity labels, stored under `case_studies/reference/` (pending
§7.2). Target size on first population: **20 authentic + 20 synthetic +
10 mixed**, across at least two domains. This is a *qualitative drift
check*, not a power study: with n=20 per class the Wilson CI halfwidth
at p=0.5 is ±0.22, which is too wide to detect sub-5-point drift but
adequate to catch catastrophic regressions (e.g. parser breakage,
judge schema mis-calibration on a new provider). Accept the wide CI
intentionally; power-study sizing is out of scope. Every scorecard
records the reference-set version used for the most recent validation.

**PAA artifact:** new directory `case_studies/reference/` with a JSONL
dataset plus a README documenting label provenance; version constant
referenced from task metadata.

**NIST basis:** VIM 5.13 (reference material); ISO 17025 §7.7 (ensuring
validity of results via replicates / retests); NIST/SEMATECH Ch.
2.2–2.3.

**Alternatives considered:**
- No reference set, rely on case-study-specific validation — rejected:
  case-study data changes; a drifting reference defeats the purpose.
- Aim for n=100+ per class for power — rejected: labelling cost for a
  solo-author project, and the goal is drift detection, not power.
- Host externally (HuggingFace, etc.) — rejected: in-repo keeps the
  artefact version-locked to the code that consumes it.

**Tradeoff:** n=20 per class is a qualitative check; a 3-point
detection_rate drift will not trigger it. The validation record names
this explicitly so the check-standard is not oversold.

**Status:** adopt; directory location is an open question (§7.2).

### 4.2 Parser calibration against the reference set

**Decision:** Calibrate the parser fallback chains
(`_parse_attribution`, `_parse_confidence`, `_parse_diagnostic`,
`_extract_json_object`, and the spontaneous-phrase tier-2 matcher)
against §4.1's reference set. For each parser: record per-model
accuracy (tier-1 hit rate, tier-2 hit rate, fallback invocation rate,
outright failure rate). This gives the uncertainty budget a measured
Type B component for parser error rather than a bound.

**PAA artifact:** `docs/metrology.md §parser-calibration`;
`src/prefill_awareness_audit/probes/awareness.py` (parsers);
`src/prefill_awareness_audit/probes/spontaneous.py:19`
(tier-2 window); `src/prefill_awareness_audit/probes/_util.py`
(`_extract_json_object`).

**NIST basis:** VIM 2.39 (calibration — establishing the relation
between indication and measurement result with uncertainty);
ISO 17025 §7.2.2 (method validation, instrument specifically).

**Alternatives considered:**
- Skip calibration, keep parsers as "trusted infrastructure" — rejected:
  the Explore report flagged parsers as heuristic and untested; silent
  per-model regressions are the likeliest failure mode.
- Unit-test parsers on synthetic inputs only — rejected: synthetic
  inputs do not exhibit the distribution of real-model outputs,
  especially across providers.

**Tradeoff:** calibration results are provider-dependent and will need
refresh when a major model generation ships. The metrology doc
captures the last-calibrated date per parser per provider.

**Status:** adopt.

### 4.3 Per-scorecard environment capture

**Decision:** Every scorecard records the following fields in Inspect
`task_metadata` (the canonical record per ISO 17025 §7.5). Fields:

- `probe_model` — provider/model as canonicalised by Inspect
  (`str(state.model)`).
- `probe_model_version` — provider-reported version hash where
  available, else the `provider/model` string plus date of run (dates
  are a weak proxy for version when hashes are unavailable).
- `judge_model` — provider/model (currently pinned;
  `probes/prompts.py:135`).
- `inspect_ai_version`, `paa_version` — from package metadata.
- `seed` — the integer passed to the `@task` (default 42 at
  `_tasks.py:21`, `task_factory.py:74`, `:192`).
- `run_utc_date` — ISO-8601.
- `active_probe_set` — explicit list (not derived post-hoc from metric
  keys).
- `intervention_condition` — the `Condition` enum value.
- `sampling_parameters` — dict of temperature / top-p / max-tokens as
  configured (may be empty, indicating provider defaults).
- `reference_set_version` — for calibration runs only.

"Active probe set" is captured explicitly rather than reconstructed
from metric output, because a reviewer seeking to repeat the
measurement needs the run recipe, not the output.

**PAA artifact:**
`src/prefill_awareness_audit/scoring/audit_scorer.py`
(score.metadata); `src/prefill_awareness_audit/_tasks.py`,
`src/prefill_awareness_audit/task_factory.py` (task metadata population).

**NIST basis:** ISO 17025 §6.4 (equipment / versioning), §7.5
(technical records sufficient to permit repetition); AI RMF MEASURE 2.1.

**Alternatives considered:**
- Infer model version from the `provider/model` string alone — rejected:
  provider aliases shift silently; explicit version hashes survive.
- Record to a sidecar JSON — rejected: Inspect's task/eval metadata is
  the canonical record and is already preserved per run.
- Derive `active_probe_set` from the presence of per-probe metric keys
  — rejected: reconstruction post-hoc is incomplete (e.g. a probe that
  failed every sample leaves no metric keys) and violates the "record
  the recipe, not just the result" principle.

**Tradeoff:** the reference-set-version field couples §4.3 to §4.1 —
§4.3 cannot be fully implemented until §4.1 ships. Sequencing: §4.3
can land the core fields (model, versions, seed, date, probe set,
condition, sampling parameters) independently of §4.1.

**Status:** partial-ratify + adopt (core fields already Inspect-native;
`active_probe_set`, `sampling_parameters`, `paa_version`,
`reference_set_version` are new).

### 4.4 `compare.py` delta-reading convention

**Decision:** A delta between a condition metric and the PROBE_ONLY
baseline is treated as signal only when its absolute value exceeds the
larger of the two endpoints' combined-standard-uncertainty halfwidths.
Smaller deltas are reported in `compare.py` output but marked
"within measurement uncertainty" rather than "no effect." The
convention is purely a reading rule — it does not change the
computation — and is documented in the metrology doc and in the
`compare.py` docstring.

**PAA artifact:**
`src/prefill_awareness_audit/analysis/compare.py:86-104`
(delta computation);
`src/prefill_awareness_audit/analysis/compare.py:236-244` (delta
display).

**NIST basis:** GUM §8 (reporting procedure); ISO 17025 §7.8.3
(reports include uncertainty where relevant); VIM 2.26 (measurement
uncertainty).

**Alternatives considered:**
- Pick a fixed threshold (e.g. |delta| > 0.05) — rejected: ignores the
  underlying uncertainty, which varies by N and by condition.
- Full hypothesis test per delta — rejected: premature for a
  post-hoc comparison of two Bernoulli rates at small N, and the test
  multiplicity across ~30 per-condition metrics is worse than helpful.
- Defer the rule to the three-outcome classifier (§7.1) — rejected:
  the human reader needs a reading rule before a classifier exists.

**Tradeoff:** the rule is conservative — deltas inside the combined
uncertainty are not called signal. Some real effects will be missed in
early case studies with small N. The metrology doc frames this as a
deliberate preference for specificity over sensitivity at the reading
level, matching PAA's broader "structurally entangled is a valid
finding" posture.

**Status:** adopt.

---

## 5. Positioning against NIST AI RMF and GenAI Profile

### 5.1 PAA as MEASURE 2.13 meta-measurement; "valid and reliable" is the RMF-native construct-validity frame; GenAI Profile mapping by name

**Decision:** PAA's purpose in AI RMF terms is *MEASURE 2.13 —
effectiveness of measurement methods is evaluated*. Declare this
explicitly: PAA is a meta-measurement instrument whose measurand is a
property of other evaluation instruments (whether their reported result
is driven by the declared construct or by prefill-detection). Tie
PAA's review cadence to the reference-set rerun in §4.1 — reviewed
before each major release, not calendar-based, to avoid committing a
research artefact to a schedule it cannot keep.

In public materials (LessWrong post, README), state the
construct-validity thesis using AI RMF MEASURE 2.5 ("valid and
reliable") as the NIST-native frame, and cite Jacobs & Wallach 2021
("Measurement and Fairness") as the bridge to the AI-measurement
literature's "construct validity" term.

For GenAI Profile alignment, map PAA by **name**, not by §2 risk
number (AI 600-1's risk numbering is not stable across drafts). Primary
alignment: *Human–AI Configuration* — PAA addresses one mechanism by
which a benchmark can mis-characterise human–AI interactions, namely
when the model reads the evaluation context rather than the intended
task. Adjacent: *Confabulation* — PAA measures behaviour in the same
family (model generates attributions about its own prior turns, some of
which may be confabulated).

**PAA artifact:** `README.md`, `PROJECT_VISION.md`;
`docs/metrology.md §positioning`.

**NIST basis:** AI RMF MEASURE 2.13 (effectiveness of measurement
methods); MEASURE 4.1 (measurement approaches evaluated and updated);
MEASURE 2.5 (valid and reliable); Jacobs & Wallach 2021; NIST AI 600-1
GenAI Profile §2 (risk list, cited by name).

**Alternatives considered:**
- Position PAA as a measurement of a specific AI RMF risk (e.g.
  confabulation directly) — rejected: PAA does not measure the risk
  directly; it measures a confound in other measurements of the risk.
- Avoid AI RMF positioning — rejected: MEASURE 2.13 is exactly where
  PAA sits and the framework gives it an established home.
- Exclusive use of "construct validity" — rejected: NIST-adjacent
  readers will lose the thread; the AI-measurement literature retains
  it for their audience.
- Exclusive use of AI RMF "valid and reliable" — rejected: AI-measurement
  readers expect the construct-validity term.
- Cite AI 600-1 risks by §2 number — rejected: the numbering has
  shifted across drafts; naming is stable, numbering is not.

**Tradeoff:** committing to a pre-release review cadence ties future
maintenance effort to code releases. The cadence is framed as a
trigger ("before each major release"), not a calendar commitment, so
if PAA enters dormancy the cadence pauses with it — avoiding a stale
commitment becoming a footgun. Publishing a vocabulary bridge between
NIST and the AI-measurement literature adds one glossary entry and
one footnote in the post.

**Status:** adopt.

---

## 6. NIST coverage gaps

### 6.1 Publish PAA's own conventions for gaps NIST does not yet address

**Decision:** Where NIST and NIST-adjacent guidance is thin for
LLM-evaluation contexts, PAA publishes its own convention and labels
it as such. Gaps addressed:

- **LLM-as-judge uncertainty.** No NIST worked example for an
  instrument whose readout is itself a stochastic model. *PAA's
  convention*: treat the judge as a Type B influence quantity (§2.3);
  pin the judge to avoid conflict of interest (ratifying
  `docs/design-decisions.md §4.10` at `probes/prompts.py:135`); leave
  inter-judge variance as an open question (§7.3).
- **Capability reference materials.** No NIST CRM for "transcript a
  capable model should recognise as fabricated." *PAA's convention*:
  the check-standard set (§4.1) is a non-certified reference material,
  versioned, with explicit "not a VIM 5.14 CRM" language.
- **Compound-metric uncertainty reporting.** No AI-specific convention
  for u_c on g-mean, Cohen's d, etc. *PAA's convention*: GUM §5.1
  closed form where one exists (g-mean, §3.2); Hedges & Olkin
  approximation where it does not (Cohen's d, §3.3); bootstrap as the
  last-resort escape hatch. All named in the scorecard glossary.
- **Prompt-as-procedure versioning.** ISO 17025 §7.2 assumes a
  procedure document; a prompt *is* a procedure. *PAA's convention*:
  `probes/prompts.py` is under git version control as the canonical
  procedure record; changes to it record (a) the file's git SHA at
  run time in task metadata, and (b) an optional module-level
  `PROMPT_SCHEMA_VERSION` constant that is bumped when output-shape
  changes break downstream parsing. The constant is new; until it
  lands, the git SHA is sufficient.

**PAA artifact:** `docs/metrology.md §coverage-gaps`;
`src/prefill_awareness_audit/probes/prompts.py` (future
`PROMPT_SCHEMA_VERSION` constant).

**NIST basis:** documents the inverse of the cited clauses — TN 1297,
AI RMF MEASURE 2.13, VIM 2.6, ISO 17025 §7.2 — as places where NIST
defers to the measurement community.

**Alternatives considered:**
- Silently adopt ad-hoc conventions — rejected: reviewers find them
  anyway; better to name them.
- Wait for NIST guidance — rejected: PAA cannot pause; the post cannot
  wait.
- Invent a new formalism — rejected: PAA adopts existing conventions
  (GUM, Hedges-Olkin, git) and names them; it does not push methodology.

**Tradeoff:** labelling conventions as "PAA convention" is honest but
limits their authority — reviewers may substitute their own. That is
the correct posture for a single-author research tool.

**Status:** adopt.

---

## 7. Open questions for Joey

Five decisions deferred with real tradeoffs.

### 7.1 Three-outcome classifier rule

`PROJECT_VISION.md` describes three outcomes (robust / artifact-sensitive
/ structurally-entangled); the classifier is unimplemented. Options:
(a) commit to a transparent threshold rule now (e.g. use §4.4's
delta-reading convention at the per-metric level plus a per-condition
roll-up); (b) declare the classifier scope-deferred and document the
decision surface the reader should apply. Default: **(b)** — scope-defer
until a multi-case-study dataset exists to tune against.

### 7.2 Reference-set directory location

§4.1 targets `case_studies/reference/`. Alternatives: `agent_artefacts/`
(ephemeral), a new top-level `references/` (clearer semantics, but one
more root directory). Default: **`case_studies/reference/`**.

### 7.3 Inter-judge variance study

Default: **no, stipulate Type B bound** (§3.1 and §6.1). A full sweep
(Claude vs GPT vs Gemini) costs real money and attention for a Type B
contribution we can bound from prior judge-agreement literature. If
the LessWrong post wants to claim judge-independence empirically, we
need evidence.

### 7.4 ISO 17043 proficiency-testing positioning

Default: **no, out of scope**. PAA is an auditing tool, not a
multi-evaluator PT scheme. If future work positions multiple audit
implementations side-by-side, ISO 17043 §5 is the right frame.

### 7.5 Multiplicity control for coupling across `main_score_fields`

Deferred per §3.4. When the first multi-field benchmark integration
ships, choose between: (a) compute p-values alongside CIs and apply
BH FDR; (b) adopt simultaneous-inference CIs; (c) widen CIs by a
Bonferroni factor. Default: **(a)** when data exists; no decision now.

---

## References

- NIST AI RMF 1.0 (AI 100-1) — <https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf>
- NIST AI 600-1 Generative AI Profile — <https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf>
- NIST Technical Note 1297 (1994) — <https://www.nist.gov/pml/nist-technical-note-1297>
- JCGM 100:2008 (GUM) — <https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf>
- JCGM 200:2012 (VIM 3rd ed.) — <https://www.bipm.org/documents/20126/2071204/JCGM_200_2012.pdf>
- NIST/SEMATECH e-Handbook of Statistical Methods — <https://www.itl.nist.gov/div898/handbook/>
- ISO/IEC 17025:2017 — <https://www.iso.org/standard/66912.html> (paywalled)
- ISO/IEC 17043:2023 — <https://www.iso.org/standard/80864.html> (paywalled)
- Jacobs, A. Z. & Wallach, H., "Measurement and Fairness," FAccT 2021 —
  <https://dl.acm.org/doi/10.1145/3442188.3445901>
- Hedges, L. V. & Olkin, I., *Statistical Methods for Meta-Analysis*,
  Academic Press 1985 (Cohen's d variance, Ch. 5).
- Borenstein, M. et al., *Introduction to Meta-Analysis*, Wiley 2009,
  Ch. 4 (effect-size CIs).
- Cumming, G. & Finch, S., "A Primer on the Understanding, Use, and
  Calculation of Confidence Intervals that are Based on Central and
  Noncentral Distributions," Educ. Psychol. Meas. 61(4), 2001.
- Brown, L. D., Cai, T. T. & DasGupta, A., "Interval Estimation for a
  Binomial Proportion," Statistical Science 16(2), 2001.
- Zheng, L. et al., "Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena," NeurIPS 2023.
- PAA context items: `d07d2a5f` (ground truth), `822c98bd` (latent
  probe design), `9eb0786f` (42-run audit matrix).
