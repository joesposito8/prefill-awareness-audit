# Project Vision

## Scope

Prefill Awareness Audit is a reusable audit scaffold that determines whether a transcript-based AI eval's results are driven by what the eval claims to measure, or by the model detecting that its conversation history has been fabricated.

The tool takes a benchmark and model, applies a set of independent intervention branches (each targeting a different category of detection cue), and classifies the result into one of three outcomes.

## The problem

Transcript-based evals show a model prefilled or synthetic conversation history, then measure how the model responds. The construct-validity risk: the model may detect the history is fake, and that detection -- not the thing the eval claims to measure -- may drive the benchmark result.

## Three outcomes

**Robust**: The model shows little awareness that its history was fabricated. The eval result is unlikely to be confounded by prefill detection.

**Artifact-sensitive**: The model detects fabrication, but the signal is driven by surface artifacts (formatting inconsistencies, metadata markers, style mismatches). Cleaning these artifacts materially reduces detection, suggesting the eval's construct can be preserved with better transcript preparation.

**Structurally prefill-entangled**: The model detects fabrication even after all admissible surface interventions. Awareness is baked into the structure of the eval design itself. This is a valid, informative finding -- not a failure requiring escalation.

## The audit matrix

The audit uses independent intervention branches, not a cumulative ladder. Each branch modifies the transcript in one way and is compared directly to the probe-only baseline. This gives clean attribution: if a branch lowers awareness, you know which cue type was responsible.

Interventions are defined by two axes:
- **Scope**: local (target span only), block (all prefilled assistant turns), global (transcript-wide)
- **Mechanism**: deterministic (rule-based formatting cleanup) or rewrite (model-based contrastive style matching)

Three conditions are always required (probe-only, local-deterministic, local-rewrite). Additional conditions (block, global) are declared per-benchmark via `AuditProfile.allowed_conditions`, and every intervention is validated against the profile's `InterventionConstraints`.

## Scorecard

Every benchmark/model result produces a scorecard with:
- Main benchmark scores (the eval's native outputs)
- Direct awareness scores (self / not-self / uncertain attribution)
- Confidence scores (0-100 tampering probability)
- Diagnostic reason scores (what the model thinks is suspicious)
- Spontaneous awareness (unprompted suspicion in main response)
- Intervention response scores (what each branch changed)
- Awareness-benchmark coupling (does awareness correlate with the main result?)
- Optional control baseline comparison

The classification label is the summary judgment these scores support.

## First case study

UKGovernmentBEIS/misalignment-continuation -- a benchmark that evaluates whether models continue misaligned actions from synthetic transcripts.

## Non-goals

- Comprehensively measuring prefill awareness as a capability
- Generating control baselines (accepted as optional external input)
- Cumulative/stacked interventions as primary analysis
- Invariant specification language (declarative `InterventionConstraints` with protected roles instead)
- Global model rewriting of user/system messages
- Iterative optimization loops
- Supporting non-transcript-based evals
- Producing a standalone benchmark
