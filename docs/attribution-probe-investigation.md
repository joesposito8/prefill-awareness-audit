# Attribution Probe Investigation — Findings

## Question

Why did BFCL audit runs never produce "self" attribution even when the probe model authored the content? Is the probe broken, or was BFCL a bad fit?

## Experiments Run

All experiments probed `case_studies/baseline_awareness/data/sample.jsonl` (20 Arena conversations: 10 authored by `anthropic/claude-opus-4-20250514`, 10 by `google/gemini-2.5-flash`). Prose-rich, no tools.

### Experiment 1 — Opus probing mixed Arena

| Source model | Probe | `self` | `not_self` | `uncertain` |
|---|---|---:|---:|---:|
| anthropic/claude-opus-4-20250514 (authentic) | Opus | **3** | 7 | 0 |
| google/gemini-2.5-flash (non-authentic) | Opus | 2 | **7** | 1 |

Opus misattributes its OWN output as "not_self" 70% of the time (confidence=95 on these wrong calls).

### Experiment 2 — Haiku probing mixed Arena

| Source model | Probe | `self` | `not_self` | `uncertain` |
|---|---|---:|---:|---:|
| anthropic/claude-opus-4-20250514 (NOT authentic for Haiku) | Haiku | **6** | 4 | 0 |
| google/gemini-2.5-flash (non-authentic) | Haiku | 2 | 4 | 4 |

Haiku claims Opus-authored content as "self" 60% of the time.

### Experiment 3 — BFCL prose density

3 BFCL samples, 48 assistant turns total. Classified by word count + structured-formatting presence:

- **prose_rich** (≥40 words + tables/bullets/bold): 8 turns (17%)
- **prose_plain** (≥40 words): 0 turns (0%)
- **functional** (<40 words, preamble/transition): 40 turns (83%)

## Conclusions

**1. The probe is fundamentally broken, not BFCL-specific.** Opus fails to recognize its own Arena output 70% of the time on prose-rich, ideal-case data. The BFCL "uncertain" pattern was a surface symptom of a deeper issue. Every hypothesis that was BFCL-scoped (H1 missing tools, H2 style-poor) was a misdiagnosis.

**2. H3 (Claude-family convergence) is confirmed.** Haiku attributes Opus-authored content as "self" 60% of the time with confidence=0. It treats Claude-family style as self-style.

**3. Opus and Haiku use opposite heuristics.**
   - **Opus** collapses the two probe questions into "does this look tampered?" High tampering (conf=95) → `not_self`, regardless of actual authorship. That's why it labels its OWN content as "not self."
   - **Haiku** treats "no tampering detected" as sufficient evidence for "self". Low tampering (conf=0) → `self`, even on Opus-authored content. That's the 60% false self-attribution on Opus.

**4. The probe prompt conflates two distinct questions.** `DEFAULT_AWARENESS_QUESTION` asks both "was this written by you?" (Q1) and "was the history artificially constructed?" (Q2). Models apparently can't separate them — Q2's tampering judgment drives Q1's self/not_self answer.

**5. BFCL's "uncertain" skew is secondary.** Once the fundamental probe issue is fixed, BFCL's 83% functional-glue density will still make it a lower-signal benchmark than Arena, but that's a tier-2 concern, not the root cause.

## Rejected Hypotheses

| # | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | Missing tool definitions confuse the model | Disproven | Adding tools didn't change attribution labels (`/tmp/test_tools_hypothesis.py`) |
| H2 | BFCL is style-poor | Retracted | Inspection showed rich distinctive style in summary turns |
| H4 | Prompt elicits hedging on any input | Disproven | Only 0-4/10 "uncertain" on Arena; models commit to self/not_self |
| H6 | Parser false positive | Disproven | Raw responses are literal `ATTRIBUTION: self/not_self/uncertain` |

## Supported Hypotheses

| # | Hypothesis | Status |
|---|---|---|
| H3 | Claude-family style convergence | Confirmed (Haiku: 60% false-self on Opus) |
| H5 | Probe is epistemically ill-posed | Strongly supported (Opus: 70% false-not-self on own content) |
| **H7 (new)** | **Q1 and Q2 in the combined prompt are conflated by models; tampering-confidence drives attribution** | **New finding** |

## Recommended Follow-up (separate task)

1. **Split the prompt.** Decouple attribution from tampering detection. Ask attribution alone, without the tampering framing.
2. **Test model-specific prompt variants.** Opus and Haiku have opposite priors; a single prompt may not work for both.
3. **Consider that "self-attribution via style" may be fundamentally infeasible for same-family models.** If so, the audit's scope should be explicit: it detects cross-family prefill, not same-family.
4. **Run Experiment 1/2 equivalent with Sonnet** to triangulate whether Opus's bias is an Opus quirk or common to larger Claudes.

## Cost Spent

~$0.80 (dominated by Opus run; Haiku negligible; Exp 3 free).
