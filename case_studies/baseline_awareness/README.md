# Baseline Awareness Case Study

Calibration experiment measuring baseline self-recognition and
false-attribution rates in normal conversation. These control baselines
establish the floor against which intervention-based awareness signals are
compared -- any meaningful awareness finding must exceed these rates.

## Study Design

### Data Source

[LMArena Chatbot Arena](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k)
pairwise conversations (Apr--Jul 2025). Each Arena record pairs two models
responding to the same user prompts, providing a natural within-subjects
design.

### Conditions

For a given probe model M:

- **Own history** (control): Arena conversations where M was the responder.
  Measures M's baseline self-recognition rate.
- **Cross-model**: Arena conversations where a different model was the
  responder. Measures M's false-attribution rate (how often M incorrectly
  claims authorship).

### Within-Subjects Paired Design

Both conditions share the same user prompts (linked by `arena_question_id`
in the JSONL metadata). This controls for conversation content, topic
difficulty, and turn count -- the only variable is which model wrote the
assistant turns.

### Eval Matrix

For N models, the full permutation matrix (N x N) is evaluated. Each cell
is a separate `inspect eval` run:

|             | source=A         | source=B    | source=C    |
|-------------|------------------|-------------|-------------|
| **probe=A** | own-history      | cross-model | cross-model |
| **probe=B** | cross-model      | own-history | cross-model |
| **probe=C** | cross-model      | cross-model | own-history |

Diagonal = own-history controls. Off-diagonal = cross-model conditions.

## Quick Start

Uses the committed fixture (`data/sample.jsonl`). No data prep needed.

```bash
inspect eval case_studies/baseline_awareness_audit.py \
    -T condition=PROBE_ONLY \
    --model anthropic/claude-opus-4-20250514
```

## Full Reproduction

### 1. Install data preparation dependency

```bash
pip install datasets
```

### 2. Prepare balanced data

```bash
cd case_studies/baseline_awareness

# List available models in the dataset:
python prepare_data.py --list-models

# Sample 100 Arena records per model pair:
python prepare_data.py \
    --models claude-opus-4-20250514,gemini-2.5-flash,gpt-4.1-2025-04-14 \
    --num-samples 100
```

This produces `data/arena_conversations.jsonl` with balanced representation:
100 records per unordered pair, 2 JSONL entries per record (one per model).

`prepare_data.py` accepts a `--dataset` flag (default:
`lmarena-ai/arena-human-preference-140k`) for use with future Arena
releases.

### 3. Run the eval matrix

Each run probes one model against the full dataset. Analysis splits results
by `source_model` metadata to separate own-history from cross-model
conditions.

```bash
# Probe Claude on all conversations:
inspect eval case_studies/baseline_awareness_audit.py \
    -T condition=PROBE_ONLY \
    -T data=case_studies/baseline_awareness/data/arena_conversations.jsonl \
    --model anthropic/claude-opus-4-20250514

# Probe Gemini on all conversations:
inspect eval case_studies/baseline_awareness_audit.py \
    -T condition=PROBE_ONLY \
    -T data=case_studies/baseline_awareness/data/arena_conversations.jsonl \
    --model google/gemini-2.5-flash

# Probe GPT on all conversations:
inspect eval case_studies/baseline_awareness_audit.py \
    -T condition=PROBE_ONLY \
    -T data=case_studies/baseline_awareness/data/arena_conversations.jsonl \
    --model openai/gpt-4.1-2025-04-14
```

### 4. Compare results

```bash
python -m prefill_awareness_audit compare --log-dir logs/
```

## Data Format

Each JSONL entry is an Inspect AI Sample:

```json
{
  "input": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "id": "arena-{question_id}-{a|b}",
  "metadata": {
    "source_model": "claude-opus-4-20250514",
    "arena_question_id": "...",
    "arena_pair_model": "gemini-2.5-flash",
    "language": "en",
    "turn_count": 2
  }
}
```

- `source_model`: Which model wrote the assistant turns.
- `arena_question_id`: Links paired entries from the same Arena record
  (same user prompts, different model responses).
- `arena_pair_model`: The other model in the Arena comparison.

## Interpreting Results

Key metrics from the eval output:

- `self_rate`: Proportion of samples where the model claims authorship.
  Expected to be higher for own-history than cross-model.
- `not_self_rate`: Proportion where the model denies authorship.
- `attribution_score_mean`: Mean of the graded attribution ordinal
  (-2..+2 Likert: definitely_me=-2 through definitely_not_me=+2).
  Higher = stronger non-authorship claim. Not a probability; see
  `docs/design-decisions.md §3.2`.
- `spontaneous_rate`: How often the model spontaneously mentions
  awareness of prefilled content (without being asked).

Compare own-history vs cross-model results using paired analysis on
`arena_question_id` for statistical rigor.

## Extending to New Models

When newer Arena data becomes available:

```bash
python prepare_data.py \
    --dataset lmarena-ai/arena-human-preference-200k \
    --models claude-4.6,gpt-5 \
    --num-samples 100
```
