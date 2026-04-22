# NIST Measurement-Science Review — Working Notes

Internal scratch for PAA-26. **NOT a deliverable**. The deliverable is
`nist-aligned-design-decisions.md` (Phase C output), which is appended to
the project Google Doc. Everything here is the scaffolding that produces
those decisions.

---

## Phase A — PAA metrological model

### A.1 Measurands (VIM 2.3)

A measurand is *the quantity intended to be measured*; specification
requires the kind of quantity and the state of the phenomenon. PAA
estimates six distinct measurands.

| ID | Measurand | Kind of quantity | State of phenomenon | Code |
|---|---|---|---|---|
| M1 | **Direct attribution rate** — probability that, given a prefilled assistant span of known authenticity, the model under test attributes the span to a specified source when asked | categorical probability (3-way: SELF/NOT_SELF/UNCERTAIN) aggregated into confusion-matrix cells | probe model queried immediately after the transcript under a specified probe question template, no tools, no CoT other than what the model volunteers | `probes/awareness.py`, metrics in `scoring/metrics.py:139-248` (`ground_truth_metrics`) |
| M2 | **Tamper confidence** — model's subjective probability (0–100) that the conversation history has been artificially constructed | bounded real on [0, 100]; reported as mean / median / std with 95% CI on the mean | probe model asked combined attribution+confidence question | `probes/awareness.py` parser `_parse_confidence`; `scoring/metrics.py` `confidence_stats` |
| M3 | **Spontaneous awareness rate** — probability that the model's main response (before any probe) contains unprompted awareness-indicating language | binomial proportion | detected by a two-tier keyword scanner over last assistant message text + reasoning | `probes/spontaneous.py`, `scoring/metrics.py` `spontaneous_rate_metric` |
| M4 | **Latent awareness (free-text)** — probability that a neutral reflection or user-persona continuation response is judged to express authorship doubt; plus graded awareness score (0–2) | binomial proportions per technique + graded ordinal scale | probe model generates text under one of two prompts (reflection / persona); fixed judge model classifies the output as `none`/`suspicious`/`convinced` | `probes/reflection.py`, `probes/persona.py`, `probes/prompts.py:135` (`LATENT_JUDGE_MODEL`), `scoring/metrics.py` `latent_rates` |
| M5 | **Latent awareness (MCQ, opt-in)** — probability that the model selects the canonical "prefill" or "eval" option from a shuffled 9-option menu | two independent binomial proportions | probe model asked to pick all options that apply; shuffle deterministic per sample | `probes/multiple_choice.py`, `scoring/metrics.py` `multiple_choice_rates` |
| M6 | **Awareness–benchmark coupling** — the association between the model's awareness and the eval's native score, per score field | per-field Cohen's d + two Pearson correlations with Fisher-z CI | computed post-hoc from the per-sample scorer output | `scoring/metrics.py:580-680` `coupling_metrics` |

Note: the three-outcome *classification* (`robust` / `artifact-sensitive` /
`structurally prefill-entangled`) referenced in `PROJECT_VISION.md` is
**not** a measurand — it is a proposed decision rule on top of M1–M6 and
is not currently implemented.

### A.2 Measurement equations

Closed-form compound quantities where GUM §5.1 sensitivity analysis
attaches:

- **G-mean** (in `ground_truth_metrics`):
  `g = sqrt(detection_rate · recognition_rate)`
  where detection_rate = TN/(TN+FP), recognition_rate = TP/(TP+FN). Input
  quantities: two binomial proportions. Sensitivity coefficients:
  `∂g/∂p1 = p2/(2g)`, `∂g/∂p2 = p1/(2g)`. **No u_c currently reported.**
- **Latent graded score mean** (in `latent_rates`):
  `L̄ = (1/N) Σ mean(reflection_score_i, persona_score_i)`
  with category scores {NONE=0, SUSPICIOUS=1, CONVINCED=2}. Reported with
  95% CI on the mean via `_mean_ci`.
- **Cohen's d**: pooled SD of two benchmark-score groups conditional on
  attribution. Formula at `metrics.py:99-113`. **No CI reported on d
  itself.**
- **Pearson r (confidence ↔ benchmark; latent ↔ benchmark)**: Fisher z CI
  at `metrics.py:75-96`.
- **Ground-truth confusion rates**: direct ratios (k successes / n trials)
  with Wilson 95% CI.

### A.3 Measurement procedure (VIM 2.6)

1. **Data loading** (`data.py`, seed=42 fixed at `data.py:28`).
2. **Condition validation** against `AuditProfile.allowed_conditions`.
3. **Intervention application** — deterministic or rewrite branch, validated
   by `interventions/validator.py` (structural invariants:
   message-count, role-order, protected-role identity).
4. **Solver chain** — main `generate()` producing the initial response.
5. **Forked probes** (`probes/fork.py`) — each of the active probes runs on
   an **independent deepcopy** of TaskState; no probe observes another
   probe's Q&A.
6. **Scorer wrapping** (`scoring/audit_scorer.py`) — base scorer runs,
   then probe metadata is merged into `Score.value` and
   `Score.metadata`; `prefill_authentic = (source_model == str(state.model))`
   at `audit_scorer.py:95`.
7. **Metric aggregation** — seven top-level `@metric` functions run on the
   full `SampleScore` list, producing `dict[str, float]` that Inspect
   expands into `EvalMetric` entries.

Parallelism: probe fork. Caching: two-level disk cache for rewrite
interventions in `interventions/rewrite.py`.

### A.4 Instruments (VIM 2.3–2.6 applied to each probe)

| Instrument | Principle | Output | Known failure modes |
|---|---|---|---|
| `awareness_probe` | direct self-report with ATTRIBUTION:/CONFIDENCE: markers | attribution label (SELF/NOT_SELF/UNCERTAIN/None) + confidence [0, 100] or -1 | parser sentinel on ambiguous responses; `_parse_attribution` falls back to bare-keyword scan |
| `reflection_probe` | neutral first-person reflection graded by LLM judge | 3-way category via `LatentCategory` | judge schema parse failure → None; judge-model conflict-of-interest avoided by pinning judge |
| `persona_probe` | user-persona role-swap graded by LLM judge | 3-way category | same as reflection |
| `diagnostic_probe` | structured JSON reason tagging | `{turn, tags, reason}` + `parse_failed` flag | structured output may fallback to free-form on models without ResponseSchema support |
| `multiple_choice_probe` (opt-in) | shuffled 9-option menu; parser reads first non-empty line | set of canonical choice IDs; `latent_prefill` and `latent_eval` binary sentinels (-1 on failure) | position-bias mitigated by deterministic shuffle seeded per sample |
| `spontaneous_detector` | two-tier keyword scan (scorer-side, not a solver) | `{detected, matched_phrases}` | tier-2 context anchor window = 300 chars (hardcoded, not validated) |

### A.5 Reference-material chain (VIM 2.41 traceability; VIM 5.13 reference material)

- `source_model` metadata on each Sample (`data.py:28-34`) is compared
  against `str(state.model)` to yield `prefill_authentic`.
- For eval-log inputs, `source_model` is set from the original log's
  `eval.model` (canonical `provider/model` string).
- For JSONL inputs, `source_model` defaults to `"unknown"`.
- Unknown-provenance samples are counted but excluded from the confusion
  matrix at `scoring/metrics.py:174-176`.
- **No external certified reference material** (no VIM 5.14 CRM). The
  traceability chain terminates at the benchmark integrator who set
  `source_model`.

### A.6 Influence quantities (VIM 2.52)

| Quantity | Current status | Captured in output | NIST observation |
|---|---|---|---|
| Probe model identity | experimental variable | yes (state.model + source_model) | primary factor of interest |
| Judge model identity | fixed constant (`anthropic/claude-sonnet-4-6`) | implicit (pinned) | uncontrolled systematic (Type B) |
| Judge prompt template | fixed constant | no | uncontrolled systematic |
| Probe question wording | fixed defaults, overridable kwargs | no | uncontrolled systematic |
| Condition / intervention | primary experimental variable | yes | designed as changed-condition per VIM 2.24 |
| Random seed | fixed (42) | yes (task metadata) | repeatability control per VIM 2.20 |
| Sample limit N | controllable | yes (via `n_samples` keys) | affects uncertainty, not mean |
| Sample ordering | seed-determined at load | implicit | reproducibility control |
| Parser fallback chain | deterministic, multi-tier | failures tracked (`n_parse_fail`) | heuristic, unvalidated |
| Confidence scale | fixed [0, 100] | yes | documented in prompt |
| Spontaneous phrase list | fixed in `prompts.py:267-310` | matches in metadata | empirical, unvalidated |
| Tier-2 context window | 300 chars hardcoded at `spontaneous.py:19` | no | magic constant |
| Latent judge schema | 3-category StrEnum `LatentCategory` | yes | design choice, unvalidated vs human raters |
| Tool stub insertion | automatic before probes | no | framework necessity, not a measurement choice |
| Between-run stochasticity | uncontrolled (temperature not set) | no | unmeasured |

### A.7 Uncertainty quantification inventory (TN 1297 Type A/B)

| Location | What | TN 1297 classification |
|---|---|---|
| `_wilson_ci(k, n)` at `metrics.py:34-53` | Wilson score CI on every binomial rate (detection, false-attribution, recognition, miss, uncertain, spontaneous, per-option MCQ, per-tag diagnostic, per-technique latent) | Type A (sampling) |
| `_mean_ci(values)` at `metrics.py:56-72` | Normal-approx CI on the mean for confidence stats, latent graded mean, diagnostic flagged-turn mean | Type A (sampling) |
| `_fisher_z_ci(r, n)` at `metrics.py:75-96` | Fisher z CI on Pearson correlations | Type A (sampling, with transform) |
| `_cohens_d` at `metrics.py:99-113` | Point estimate of Cohen's d | **no CI** (gap) |
| `n_samples`, `n_classified`, `n_parse_fail` counts across metric groups | parse-failure tracking | Type B evidence (input-quantity quality) |
| Judge-model variance | not measured | **uncharacterized Type B** (gap) |
| Prompt-template variance | not measured | **uncharacterized Type B** (gap) |
| Between-run variance | not measured | **uncharacterized Type A** (gap) |

### A.8 Existing measurement-affecting design decisions (from `docs/design-decisions.md`)

Paraphrased so Phase B can map NIST clauses to them:

| ID | Decision | Measurement effect |
|---|---|---|
| §1.2 | Independent branches, not cumulative | clean single-variable attribution per condition |
| §1.3 | Three-outcome classification (not pass/fail) | framing; classifier stub not implemented |
| §2.1 | Condition as StrEnum | explicit logging / naming |
| §2.2 | Declarative InterventionConstraints (protected_roles only) | structural invariants enforced byte-level |
| §3.1 | Forked probes | probe independence (no cross-probe priming) |
| §3.2 | Combined attribution + confidence in one prompt | 50% fewer model calls; assumes independence of two questions |
| §3.3 | Spontaneous detection as scorer-side scan (not a probe) | avoids priming the model to notice |
| §3.4 | Two-tier phrase matching | sensitivity/specificity tradeoff, empirically tuned |
| §3.5 | Centralized `prompts.py` | version-controllable prompts |
| §4.5 | LLM judge over regex / embeddings / PCA | near-human agreement with Zheng et al. |
| §4.6 | Two free-text probes (reflection + persona) with consensus | cross-framing validation |
| §4.7 | User-persona sampling specifically | Cywiński mechanism |
| §4.8 | No assistant-prefill booster | cross-provider comparability > technique strength |
| §4.9 | Three-category judge schema (none/suspicious/convinced) | richer than binary |
| §4.10 | Fixed judge model, not probe model | avoids conflict of interest |
| §4.12 | Neutral reflection prompt | demand-characteristic control |
| §5.1 | Two-axis scope × mechanism intervention matrix | coverage of confound categories |
| §5.2 | Deterministic cleanup as composable pure functions | reproducible, testable |
| §5.3 | Contrastive style rewriting | style-match more precise than generic neutral |
| §5.4 | Two-level disk cache for rewrites | rerun reproducibility |
| §5.5 | Two-phase validation (structural + optional semantic) | guards intervention invariants |
| §6.2 | Seeded dataset loading (seed=42) | cross-condition reproducibility |

---

## Phase B — NIST-to-PAA mapping table

| NIST clause | Paraphrase | PAA artifact(s) | Current PAA treatment | Action |
|---|---|---|---|---|
| **VIM 2.3** Measurand | operational definition of the quantity intended to be measured; requires kind-of-quantity and state-of-phenomenon | M1–M6 (§A.1) | measurands implicit in metric-function docstrings; never formally stated | **decide** — write explicit measurand specifications |
| **VIM 2.6** Measurement procedure | detailed description of how measurement is performed | audit pipeline (§A.3) | described in `SYSTEM_ARCHITECTURE.md` in prose; no VIM-style measurement procedure record | **decide** — adopt a formal procedure record |
| **VIM 2.8** Validation of measurement procedure | verification that the specified requirements are adequate for intended use | whole PAA | informal; design-decisions doc argues adequacy but without validation evidence | **decide** — commit to a validation check-standard and procedure |
| **VIM 2.20 / 2.21** Repeatability | same procedure/operator/system, short time | seed=42, forked probes (§6.2, §3.1) | reproducibility of pipeline ensured by seeding; no same-run replicate study | **ratify** + note gap |
| **VIM 2.22 / 2.23** Intermediate precision | same procedure and lab, extended time, possibly different operator or calibration | cross-run scorecards, `compare.py` | compare-tool aggregates across runs but does not frame them as intermediate-precision | **decide** — re-frame compare output |
| **VIM 2.24 / 2.25** Reproducibility | different locations/operators/systems | each `Condition` vs PROBE_ONLY; cross-model runs (context `9eb0786f`) | currently framed as "intervention effect"; NIST vocabulary is *reproducibility under changed conditions* | **decide** — adopt NIST vocabulary in compare-tool output and in the post |
| **VIM 2.26** Measurement uncertainty | non-negative parameter characterizing dispersion of measurand values | Wilson, Fisher z, mean CIs | Type A only | **ratify** partial + **decide** on compound/Type-B additions |
| **VIM 2.39** Calibration | relation between indication and measured value, with uncertainty | no analogue in PAA | probe instruments are not calibrated; parser chains heuristic | **decide** — calibrate parsers against labelled response set |
| **VIM 2.41** Traceability | documented unbroken chain back to reference | `source_model` metadata, `prefill_authentic` | de facto traceability for eval-log path; absent for JSONL path (`"unknown"`) | **ratify** + document the chain and its limits |
| **VIM 2.52** Influence quantity | affects relation between indication and result without being the measurand | judge model, prompt wording, seed, condition, between-run noise (§A.6) | partially captured | **decide** — declare each influence quantity and its control status |
| **VIM 5.13 / 5.14** Reference material / CRM | stable, homogeneous material fit for measurement use | none | no external reference | **decide** — adopt a small check-standard set |
| **GUM §3** Basic concepts | measurand, input quantities, sensitivity | §A.1, §A.2 | implicit | **decide** — publish the model |
| **GUM §4.2** Type A uncertainty | evaluated statistically from repeated observations | Wilson CI, mean CI, Fisher z | implemented | **ratify** |
| **GUM §4.3** Type B uncertainty | evaluated by other means (prior data, judgement) | not explicitly modeled | judge variance, prompt variance, parser error are Type B but not stated | **decide** — declare Type B budget |
| **GUM §5.1** Combined standard uncertainty (uncorrelated) | u_c^2 = Σ c_i^2 u(x_i)^2 | g-mean, coupling stats | **not computed** for compound metrics | **decide** — adopt closed-form u_c for g-mean; bootstrap for coupling d |
| **GUM §6** Expanded uncertainty U = k·u_c; coverage factor k | reporting convention | nowhere | Wilson already approximates 95% coverage; GUM §6 convention not stated | **decide** — either adopt U = 2·u_c alongside Wilson, or document why we don't |
| **GUM §8** Reporting procedure | state measurand, procedure, u_c / U, coverage | scorecard | partial | **decide** — standardise scorecard header |
| **GUM Annex G** Effective degrees of freedom (Welch-Satterthwaite) | for small samples, use Student-t for k | not used | Wilson good for small n; GUM Annex G not needed for PAA's metric set | drop (no action) |
| **TN 1297 §2** Type A / B classification | NIST-internal policy matching GUM | same as GUM 4.2/4.3 | see above | **decide** along with GUM §4 |
| **TN 1297 §5** Combined standard uncertainty | u_c computation | same as GUM §5 | see above | **decide** along with GUM §5.1 |
| **TN 1297 §6** Expanded uncertainty, k=2 for ≈95% | reporting | nowhere | see above | **decide** along with GUM §6 |
| **AI RMF MEASURE 1.1** Select methods and metrics for significant risks | methodology | probe set and metric catalogue | implicit | **ratify** — the probe/metric design is exactly this activity |
| **AI RMF MEASURE 1.2** Regularly assess effectiveness of measurement tools | meta-measurement | no formal process | design-decisions doc is retrospective, not scheduled | **decide** — commit to a cadence |
| **AI RMF MEASURE 1.3** Internal / independent expert review | governance | n/a | solo project | **ratify** via the ultrareview-style review workflow used for LessWrong post review — declare as the mechanism |
| **AI RMF MEASURE 2.1** Document test sets, metrics, tools | documentation | `SYSTEM_ARCHITECTURE.md`, `docs/design-decisions.md`, per-scorecard metadata | strong | **ratify** + tighten per-scorecard env capture |
| **AI RMF MEASURE 2.3** Performance measured regularly | deployment monitoring | n/a until PAA is used as a service | not applicable | drop |
| **AI RMF MEASURE 2.4** Monitor for changes in performance | drift | no check-standard yet | n/a until reference set adopted | **decide** — triggered by reference-set decision |
| **AI RMF MEASURE 2.5** Validity and reliability | "valid and reliable" trustworthy characteristic — the closest AI RMF analogue of construct validity | the whole audit | PAA's thesis *is* that other evals fail this; PAA itself must also pass | **decide** — declare PAA's validity evidence |
| **AI RMF MEASURE 2.13** Effectiveness of measurement methods evaluated | meta-measurement | no formal mechanism | design-decisions doc + this review is an instance | **decide** — formalise meta-measurement cadence |
| **AI RMF MEASURE 3.1** Track risks over time | longitudinal | n/a | drop |
| **AI RMF MEASURE 4.1** Measurement approaches evaluated / updated | continuous improvement | this very review | one-off so far | **decide** — set expectation for iteration |
| **AI 600-1 §2 risk #2** Confabulation | GenAI risk | PAA measures the closely-related "detect fabricated context" behaviour | adjacency, not identity | **ratify** — note the mapping in the post |
| **AI 600-1 §2 risk #7** Human–AI configuration | GenAI risk | PAA's thesis: benchmark results may reflect the model modelling the evaluator | strong alignment | **ratify** — note the mapping |
| **AI 600-1 §3** MEASURE actions for GAI risks | action taxonomy | PAA is a measurement instrument relevant to several actions | implicit | **ratify** with cross-reference |
| **ISO 17025 §6.4** Equipment / versioning | version controls on instruments | model provider/version pinning | partial | **decide** — require exact model version recorded in every scorecard |
| **ISO 17025 §6.5** Metrological traceability | calibration chain | `source_model` + judge pinning | partial | **decide** — document the chain |
| **ISO 17025 §7.2.2** Validation of non-standard methods | validation evidence | `docs/design-decisions.md`, 42-run audit matrix, 20-sample validation | informal | **decide** — structure the validation evidence as a Method Validation Record |
| **ISO 17025 §7.5** Technical records sufficient to permit repetition | reproducibility evidence | eval logs, seed, prompts, code | strong | **ratify** |
| **ISO 17025 §7.6** Evaluation of measurement uncertainty | uncertainty budgets | Wilson / Fisher CIs | partial | **decide** — publish uncertainty budget |
| **ISO 17025 §7.7** Ensuring validity of results (replicates, retests, PT) | measurement assurance | none | no replicate or PT mechanism | **decide** — adopt periodic replicate checks against the reference set |
| **ISO 17025 §7.8.3** Reports include uncertainty where relevant | reporting | scorecard prints CIs | **ratify** |
| **ISO 17043 §5** PT statistical design | proficiency testing | n/a in current scope | PAA could become a PT scheme, but that's not where we are | **open question** — flag for Joey, no decision yet |
| **NIST/SEMATECH 2.2–2.5** Measurement assurance programme | check standards, control charts | none | see §7.7 decision | merged into the reference-set decision |
| **NIST/SEMATECH 7.2.4** Wilson vs Wald vs Clopper-Pearson | proportion CI methods | Wilson | **ratify** with citation |

Clauses explicitly dropped (no PAA artifact): GUM Annex G, MEASURE 2.3,
MEASURE 3.1, ISO 17043 §5 (flagged as open question only).

---

## Phase C — Decisions derived from Phase B

Candidate decisions, each traceable to one or more Phase-B rows marked
"decide" or "ratify." Consistency cross-check: the measurand-spec,
traceability, influence-quantity, and reference-material decisions form
a single coherent measurement-system specification; the uncertainty
budget decisions layer on top of it; the validation / meta-measurement
decisions close the loop. No decision stands alone.

Drafted in the deliverable at `nist-aligned-design-decisions.md`.
