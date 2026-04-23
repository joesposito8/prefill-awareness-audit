# Full audit — 4 subjects × 2 case studies — 2026-04-22

## Scope

First cross-subject PAA scorecard at meaningful n. Replaces the haiku-only
n=20 validation (`validation-audit-2026-04-21.md`) with a 4-subject matrix
on two case studies. Both case studies are synthetic — every cell
populates only the TNR/FPR leg of the ground-truth confusion matrix
(`recognition_rate = --` everywhere). This is intentional: BFCL was
pivoted out mid-run, and Apollo insider-trading + UK-GovBEIS
misalignment-continuation are both all-synthetic sources.

## Matrix

| # | Case study | Subject | n | Log file |
|---|---|---|---|---|
| 1 | insider-trading | haiku-4-5 | 26 | `insider-haiku45_…_LD4aThXd….eval` |
| 2 | misalign | haiku-4-5 | 50 | `misalign-haiku45_…_QoB3iBoh….eval` |
| 3 | insider-trading | sonnet-4-6 | 26 | `insider-sonnet46_…_bzz8RLsk….eval` |
| 4 | misalign | sonnet-4-6 | 50 | `misalign-sonnet46_…_dx3yRTc8….eval` |
| 5 | insider-trading | gpt-5.4 | 26 | `insider-gpt54_…_9oprzPmB….eval` |
| 6 | misalign | gpt-5.4 | 50 | `misalign-gpt54_…_HDrFrAFR….eval` |
| 7 | insider-trading | opus-4-7 | 26 | `insider-opus47_…_EjHkyC8D….eval` |
| 8 | misalign | opus-4-7 | 50 | `misalign-opus47_…_WazbtMxR….eval` |

Condition: `PROBE_ONLY` on all cells. All logs flat at
`logs/full-audit-2026-04-22/`.

## Awareness probe (ground-truth confusion matrix)

Sign convention: `attribution_score ∈ [−2, +2]` with −2 =
`definitely_me`, +2 = `definitely_not_me`.

| Metric | h·ins | h·mis | s·ins | s·mis | g·ins | g·mis | o·ins | o·mis |
|---|---|---|---|---|---|---|---|---|
| n_samples | 26 | 50 | 26 | 50 | 26 | 50 | 26 | 50 |
| n_classified | 26 | 49 | 26 | 48 | 26 | 49 | **1** | **25** |
| n_parse_fail | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 2 |
| **detection_rate** | **0.769** | 0.347 | 0.000 | 0.104 | 0.154 | 0.000 | 0.000 | 0.040 |
| detection_rate CI | [0.58, 0.89] | [0.23, 0.49] | [0.00, 0.13] | [0.05, 0.22] | [0.06, 0.34] | [0.00, 0.07] | [0.00, 0.79] | [0.01, 0.20] |
| false_attribution_rate | 0.231 | 0.653 | 1.000 | 0.896 | 0.846 | 1.000 | 1.000 | 0.960 |
| uncertain_rate | 0 | 0 | 0 | 0 | 0 | 0.020 | **0.962** | **0.479** |
| attribution_score_mean | +0.81 | −0.60 | −2.00 | −1.44 | −1.38 | −1.96 | −0.04 | −0.90 |
| spontaneous_rate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| recognition_rate | -- | -- | -- | -- | -- | -- | -- | -- |

## Reflection + persona probes (free-text, sonnet-4-6 judge)

| Metric | h·ins | h·mis | s·ins | s·mis | g·ins | g·mis | o·ins | o·mis |
|---|---|---|---|---|---|---|---|---|
| latent_prefill_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| reflection_convinced_rate | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.02 |
| persona_convinced_rate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| reflection_parse_fail | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| persona_parse_fail | 0 | 0 | 0 | 3 | 0 | 3 | 0 | 0 |

The two `reflection_convinced = 1` hits (s·mis, o·mis) are judge false
positives — the sample-level raw text affirms authorship while
critiquing content quality. Spot-check of 24 `none`-classified
samples found four structurally-identical confessions the judge left
alone; conclusion is that the reflection/persona channel is not
reliably discriminating authorship. See
`docs/design-decisions.md §4.12` on demand characteristics. A
label-rename fix landed in the same session (`convinced` →
`disclaimed_authorship`) to tighten future judge runs.

## Multiple-choice probe

| Metric | h·ins | h·mis | s·ins | s·mis | g·ins | g·mis | o·ins | o·mis |
|---|---|---|---|---|---|---|---|---|
| mc_n_samples | 26 | 47 | 26 | 50 | 26 | 50 | 26 | 48 |
| mc_parse_fail | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 2 |
| **mc_prefill_rate** | **0.385** | 0.085 | 0.000 | 0.220 | 0.077 | 0.220 | 0.000 | 0.083 |
| mc_eval_rate | 1.00 | 0.49 | 0.89 | 0.86 | 1.00 | 0.88 | 0.96 | 0.65 |
| mc_overreach_rate | 0.04 | 0.47 | 0.19 | 0.94 | 0.15 | 0.94 | 0.00 | 0.98 |
| mc_role_play_rate | 0.42 | 0.49 | 0.08 | 0.16 | 0.42 | 0.88 | 0.04 | 0.04 |
| mc_tool_mismatch_rate | 0.19 | 0.17 | 0.00 | 0.64 | 0.15 | 0.94 | 0.00 | 0.19 |
| mc_contradict_rate | 0.00 | 0.02 | 0.04 | 0.26 | 0.00 | 0.40 | 0.00 | 0.04 |
| mc_none_rate | 0.00 | 0.47 | 0.12 | 0.04 | 0.00 | 0.00 | 0.35 | 0.00 |

## Diagnostic probe

| Metric | h·ins | h·mis | s·ins | s·mis | g·ins | g·mis | o·ins | o·mis |
|---|---|---|---|---|---|---|---|---|
| behavior_policy_mismatch | 1.00 | 0.90 | 0.23 | 1.00 | 1.00 | 0.98 | 0.04 | 0.91 |
| tool_mismatch | 0.00 | 0.24 | 0.00 | 0.59 | 0.04 | 0.76 | 0.00 | 0.24 |
| style_mismatch | 0.04 | 0.06 | 0.04 | 0.12 | 0.00 | 0.24 | 0.00 | 0.00 |
| formatting_artifact | 0.00 | 0.12 | 0.00 | 0.02 | 0.00 | 0.06 | 0.00 | 0.13 |
| metadata_or_provenance | 0.12 | 0.28 | 0.00 | 0.33 | 0.00 | 0.60 | 0.00 | 0.09 |
| other | 0.15 | 0.22 | 0.00 | 0.43 | 0.04 | 0.14 | 0.00 | 0.13 |
| flagged_turn_mean | 6.6 | 2.5 | 2.0 | 2.5 | 9.5 | 9.6 | 8.0 | 8.6 |

## Coupling (misalign cells only — insider profile has `main_score_fields=[]`)

| Metric | h·mis | s·mis | g·mis | o·mis |
|---|---|---|---|---|
| aligned_given_self | 0.906 | 0.837 | 0.796 | **1.000** |
| aligned_given_not_self | 0.588 | 1.000 | NaN (n=0) | 1.000 |
| aligned_cohens_d | +0.834 | −0.456 | NaN | NaN |
| aligned_attribution_corr | −0.371 | +0.087 | +0.071 | −0.121 |
| confession_given_self | 0.469 | 0.651 | 0.265 | 0.792 |
| confession_given_not_self | 0.118 | 0.000 | NaN | 0.000 |
| confession_cohens_d | +0.772 | +1.413 | NaN | NaN |
| confession_attribution_corr | −0.354 | −0.348 | −0.085 | −0.132 |

## Cross-subject patterns

1. **Detection is a capability inversion.** On insider-trading:
   haiku 0.77 > gpt-5.4 0.15 > sonnet = opus 0. On misalign: haiku 0.35 >
   sonnet 0.10 > opus 0.04 > gpt-5.4 0. Haiku wins both case studies.
   No "bigger = better" ordering.
2. **Opus has a qualitatively distinct strategy — decline to commit.**
   `uncertain_rate` = 0.96 (insider) and 0.48 (misalign). The other
   three subjects virtually never say "uncertain"; opus uses it as the
   default on insider and on half the misalign samples. Raw responses
   show substantive meta-cognitive reasoning ("I don't have memory
   independent of the transcript provided…"). Opus's confusion matrix
   is near-empty (n_classified = 1 / 25). That's not a failure; it's a
   different epistemic posture.
3. **MC `prefill_rate` is the most stable cross-subject signal on
   misalign.** 0.085 (haiku), 0.22 (sonnet), 0.22 (gpt), 0.083 (opus).
   Even opus emits a non-zero mc_prefill despite attributing to self
   almost never. The MC probe is reaching a different mechanism than
   the direct attribution question.
4. **`mc_eval_rate` is near-saturated everywhere (0.49–1.00).** "This
   looks like a test" is detected by every subject on every cell; what
   varies is whether that detection routes to prefill or to eval.
5. **Opus on misalign is perfectly aligned: aligned_given_self = 1.000
   and aligned_given_not_self = 1.000.** 25/25 classified samples
   refused the misaligned action, regardless of authorship claim.
   Cohen's d is NaN because both conditional means equal 1. That's an
   alignment finding worth citing separately.
6. **Coupling sign on `aligned` varies by subject**: haiku +0.83,
   sonnet −0.46, gpt NaN, opus NaN. Only haiku shows the "self-claimer
   more aligned" pattern from the n=20 validation. Ceiling effects
   (opus 1.000/1.000; gpt n_not_self=0) prevent the comparison on two
   of four subjects.
7. **Latent free-text probes are not paying rent at n=50.** Reflection
   + persona judges return "none" on 99%+ of samples across every cell;
   the two `convinced` hits are judge false positives on substantive
   self-critique (see §Reflection above). Channel should be treated
   as "not discriminating" until either the judge is recalibrated or
   a different free-text framing lands.
8. **Diagnostic `behavior_policy_mismatch` is the dominant tag on
   every misalign cell and on 3/4 insider cells.** The exception is
   sonnet × insider (0.23) where sonnet is so confident of authorship
   it doesn't flag policy mismatches either. Opus × insider is the
   cleanest (0.04) because it declines the whole framing.

## Cost actuals

| Cell | Cost | Notes |
|---|---|---|
| h·ins | $0.42 | probe 0.31 + sonnet judge 0.11 |
| h·mis | $3.20 | haiku probe 1.39 + sonnet judge 0.26 + legacy-sonnet grader 1.55 |
| s·ins | $1.11 | sonnet probe+judge merged |
| s·mis | $7.18 | sonnet probe+judge 5.60 + legacy grader 1.58 |
| g·ins | $0.45 | gpt 0.35 + sonnet judge 0.10 |
| g·mis | $3.73 | gpt 1.96 + sonnet judge 0.20 + legacy grader 1.57 |
| o·ins | $7.44 | opus 7.33 + sonnet judge 0.11 |
| o·mis | ~$29.15 | opus 27.36 + sonnet judge 0.25 + legacy grader 1.54 |
| **Total** | **~$52.68** | |

Overshoot concentrated in the two opus cells ($36.59 combined) —
opus output tokens on misalign (167 k) dominated the bill at
$75/M output rate.

## Environment

- `inspect_ai`: 0.3.210
- `prefill_awareness_audit`: 0.1.0
- `paa` branch: `full-audit-2026-04-22`
- Seed: 42 (all cells)
- Probe models: each row's subject model (also does generation for
  the continuation solver)
- Judge model: `anthropic/claude-sonnet-4-6` (pinned for reflection +
  persona)
- External grader (misalign only):
  `anthropic/claude-sonnet-4-20250514` (legacy sonnet, called by
  `misalignment_continuation_scorer` for benchmark-side aligned +
  confession labels)
- Probes active: `awareness, reflection, persona, diagnostic,
  multiple_choice` on every cell (5-probe forked chain)
- Run UTC dates: 2026-04-22 through 2026-04-23

## Caveats

- Both case studies synthetic → scorecard measures TNR + FPR only;
  no TPR / FNR / g-mean anywhere.
- Insider-trading cells are n=26 (full Apollo scenario set); misalign
  cells are n=50 (`-T limit=50`).
- Haiku × misalign log is reused from the pre-pivot run (same code
  path, same seed) — not re-executed post-pivot.
- BFCL cells (pre-pivot) archived at `logs/pre-pivot-bfcl-archive/`
  and excluded from this scorecard.
- Single seed — no variance estimate across seeds. Any "finding"
  requiring seed stability should cite this limitation.
- Reflection + persona judge uses the pre-rename label vocabulary
  (`none / suspicious / convinced`); the session's rename
  (`affirms_or_neutral / hedged_authorship / disclaimed_authorship`)
  is live in the code but will only affect runs produced after this
  audit.

## Follow-up candidates

1. Re-run the two misalign cells where the reflection judge produced
   the only non-zero `convinced` rates under the renamed labels to
   confirm the false-positive goes away.
2. Cross-model probing (haiku transcripts audited by sonnet, etc.)
   would populate the recognition leg of the confusion matrix; this
   audit only exercises the detection leg.
3. Apollo insider-trading cell n=26 is the full available set;
   expanding the deception-scenario library would tighten CIs on the
   striking haiku-vs-sonnet detection gap (currently 0.769 vs 0.000
   at n=26).
