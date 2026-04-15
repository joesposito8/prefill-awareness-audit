# BFCL Demo -- Log-Based Awareness Audit

Demonstrates the log-based audit path using the Berkeley Function Calling Leaderboard (BFCL). BFCL's multi-turn category uses prefilled assistant turns containing function calls, making it a natural target for awareness probing.

This demo uses the generic `prefill_awareness_audit` task with zero custom code. No custom scorer, solver, or profile needed.

## Prerequisites

Install `inspect_evals` (provides the BFCL task):

```bash
pip install inspect_evals[bfcl]
```

## Workflow

### 1. Run the BFCL benchmark

```bash
inspect eval inspect_evals/bfcl -T categories=multi_turn_base \
    --model anthropic/claude-sonnet-4-6 --limit 20
```

This produces a log in your `INSPECT_LOG_DIR` (default: `./logs`).

### 2. Audit with the same model (authentic prefill)

```bash
inspect eval prefill_awareness_audit -T task=bfcl \
    --model anthropic/claude-sonnet-4-6
```

The audit discovers the most recent BFCL log, extracts its conversation histories (with `source_model` set from the log), and runs awareness probes. Since the same model produced the original responses, `prefill_authentic=1` for all samples. High `recognition_rate` (TPR) means the model recognizes its own writing.

### 3. Audit with a different model (non-authentic prefill)

```bash
inspect eval prefill_awareness_audit -T task=bfcl \
    --model openai/gpt-4.1
```

Now GPT-4.1 is probed on Claude's conversation histories. Since `source_model` (Claude) differs from the probe model (GPT-4.1), `prefill_authentic=0` for all samples. High `detection_rate` (TNR) means the model correctly identifies the text as not its own.

### 4. Compare results

```bash
python -m prefill_awareness_audit compare --log-dir logs/
```

## What the metrics mean

The audit produces a ground truth confusion matrix:

| | Claims self | Claims not-self |
|---|---|---|
| **Authentic** (same model) | TP (correct recognition) | FN (missed own writing) |
| **Non-authentic** (different model) | FP (false attribution) | TN (correct detection) |

Key derived rates:
- **detection_rate** (TNR): How well does the model detect text it didn't write?
- **false_attribution_rate** (FPR): How often does the model wrongly claim foreign text?
- **recognition_rate** (TPR): How well does the model recognize its own writing?
- **g_mean**: Geometric mean of TPR and TNR -- overall discriminative ability.

Step 2 (same model) populates the TP/FN row. Step 3 (different model) populates the FP/TN row. Together they give the full confusion matrix.
