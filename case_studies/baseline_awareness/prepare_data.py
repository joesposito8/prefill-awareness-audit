"""Download LMArena Chatbot Arena data and convert to Inspect AI JSONL.

Produces balanced samples for the baseline awareness case study. Each Arena
record is a pairwise comparison (two models responding to the same user
prompts), yielding two JSONL entries linked by arena_question_id for paired
analysis.

Usage:
    pip install datasets
    python prepare_data.py \
        --models claude-3-7-sonnet-20250219,gpt-4.1-2025-04-14 \
        --num-samples 100

    # Use a different Arena release:
    python prepare_data.py \
        --dataset lmarena-ai/arena-human-preference-200k \
        --models ...
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path


# Match the package's central seed convention (make_audit_task default).
SEED = 42

# Arena model names → Inspect AI provider prefix.  Inspect requires
# "provider/model" format for source_model so the scorer can compare
# it against str(state.model).  Add entries here when new providers
# appear in the Arena data.
_PROVIDER_PREFIX: dict[str, str] = {
    "claude": "anthropic",
    "gemini": "google",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "deepseek": "deepseek",
    "llama": "together",
    "mistral": "mistral",
    "command": "cohere",
}


def _canonicalize_model(arena_name: str) -> str:
    """Convert a bare Arena model name to Inspect's 'provider/model' format.

    Raises ValueError if the provider cannot be inferred.
    """
    if "/" in arena_name:
        return arena_name  # already prefixed
    prefix = arena_name.split("-")[0]
    provider = _PROVIDER_PREFIX.get(prefix)
    if provider is None:
        raise ValueError(
            f"Cannot infer provider for Arena model {arena_name!r}. "
            f"Add an entry to _PROVIDER_PREFIX in {__file__}."
        )
    return f"{provider}/{arena_name}"


def _extract_text(content: list[dict] | str) -> str:
    """Extract plain text from Arena content blocks.

    Arena conversation messages store content as a list of typed blocks:
    [{"type": "text", "text": "...", "image": None, "mimeType": None}].
    This function concatenates all text blocks into a single string.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
    return "\n\n".join(parts)


def _has_images(content: list[dict] | str) -> bool:
    """Check whether any content block contains an image."""
    if isinstance(content, str):
        return False
    return any(block.get("image") is not None for block in content)


def _convert_conversation(messages: list[dict]) -> list[dict] | None:
    """Convert Arena conversation messages to Inspect AI chat format.

    Returns None if the conversation contains images (text-only audit).
    """
    result = []
    for msg in messages:
        if _has_images(msg["content"]):
            return None
        result.append({
            "role": msg["role"],
            "content": _extract_text(msg["content"]),
        })
    return result


def _make_pair_key(model_a: str, model_b: str) -> tuple[str, str]:
    """Create a canonical unordered pair key (sorted alphabetically)."""
    return tuple(sorted([model_a, model_b]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Arena data and convert to Inspect AI JSONL.",
    )
    parser.add_argument(
        "--dataset",
        default="lmarena-ai/arena-human-preference-140k",
        help="HuggingFace dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to include (required)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Arena records per unordered model pair (default: %(default)s)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language filter (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/ relative to this script)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models in the dataset and exit",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "Error: the 'datasets' package is required.\n"
            "Install it with: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} ...")
    ds = load_dataset(args.dataset, split="train")
    print(f"  {len(ds)} records loaded.")

    # Index records by unordered model pair.
    pair_index: dict[tuple[str, str], list[int]] = defaultdict(list)
    all_models: set[str] = set()

    for idx, record in enumerate(ds):
        all_models.add(record["model_a"])
        all_models.add(record["model_b"])
        pair_key = _make_pair_key(record["model_a"], record["model_b"])
        pair_index[pair_key].append(idx)

    if args.list_models:
        print(f"\nAvailable models ({len(all_models)}):")
        for m in sorted(all_models):
            print(f"  {m}")
        # Also show pair counts for top pairs
        print(f"\nModel pairs with most records (top 20):")
        sorted_pairs = sorted(pair_index.items(), key=lambda x: len(x[1]), reverse=True)
        for pair, indices in sorted_pairs[:20]:
            print(f"  {pair[0]} / {pair[1]}: {len(indices)} records")
        return

    if not args.models:
        print(
            "\nError: --models is required. Available models:",
            file=sys.stderr,
        )
        for m in sorted(all_models):
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    requested_models = [m.strip() for m in args.models.split(",")]

    # Validate model names.
    for m in requested_models:
        if m not in all_models:
            print(f"Error: model '{m}' not found in dataset.", file=sys.stderr)
            print(f"Available models:", file=sys.stderr)
            for am in sorted(all_models):
                print(f"  {am}", file=sys.stderr)
            sys.exit(1)

    # Generate all unordered pairs from the requested models.
    requested_pairs = list(combinations(sorted(requested_models), 2))
    if not requested_pairs:
        print(
            "Error: need at least 2 models to form pairs.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nRequested models: {requested_models}")
    print(f"Unordered pairs to sample: {len(requested_pairs)}")
    print(f"Records per pair: {args.num_samples}")

    # Sample records for each pair.
    rng = random.Random(SEED)
    selected_indices: list[int] = []
    pair_counts: dict[tuple[str, str], int] = {}

    for pair in requested_pairs:
        available = pair_index.get(pair, [])
        if not available:
            print(f"  WARNING: no records for pair {pair[0]} / {pair[1]}", file=sys.stderr)
            pair_counts[pair] = 0
            continue

        # Filter by language.
        filtered = []
        for idx in available:
            record = ds[idx]
            if record["language"] == args.language:
                filtered.append(idx)

        if len(filtered) < args.num_samples:
            print(
                f"  WARNING: pair {pair[0]} / {pair[1]} has {len(filtered)} "
                f"{args.language} records (requested {args.num_samples})",
                file=sys.stderr,
            )

        shuffled = list(filtered)
        rng.shuffle(shuffled)
        sampled = shuffled[: args.num_samples]
        selected_indices.extend(sampled)
        pair_counts[pair] = len(sampled)
        print(f"  {pair[0]} / {pair[1]}: {len(sampled)} records sampled (from {len(filtered)} available)")

    # Convert selected records to JSONL entries.
    entries: list[dict] = []
    skipped_images = 0

    for idx in selected_indices:
        record = ds[idx]
        question_id = record["id"]
        model_a = record["model_a"]
        model_b = record["model_b"]
        language = record["language"]

        # Convert conversation_a.
        conv_a = _convert_conversation(record["conversation_a"])
        if conv_a is None:
            skipped_images += 1
            continue

        turn_count_a = sum(1 for m in conv_a if m["role"] == "user")
        entries.append({
            "input": conv_a,
            "id": f"arena-{question_id}-a",
            "metadata": {
                "source_model": _canonicalize_model(model_a),
                "arena_question_id": question_id,
                "arena_pair_model": model_b,
                "language": language,
                "turn_count": turn_count_a,
            },
        })

        # Convert conversation_b.
        conv_b = _convert_conversation(record["conversation_b"])
        if conv_b is None:
            skipped_images += 1
            continue

        turn_count_b = sum(1 for m in conv_b if m["role"] == "user")
        entries.append({
            "input": conv_b,
            "id": f"arena-{question_id}-b",
            "metadata": {
                "source_model": _canonicalize_model(model_b),
                "arena_question_id": question_id,
                "arena_pair_model": model_a,
                "language": language,
                "turn_count": turn_count_b,
            },
        })

    # Write output.
    output_path = output_dir / "arena_conversations.jsonl"
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Summary.
    model_counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        model_counts[entry["metadata"]["source_model"]] += 1

    print(f"\nOutput: {output_path}")
    print(f"  Total JSONL entries: {len(entries)}")
    if skipped_images:
        print(f"  Skipped (images): {skipped_images}")
    print(f"  Per source model:")
    for m in sorted(model_counts):
        print(f"    {m}: {model_counts[m]}")
    print(f"  Per pair:")
    for pair, count in pair_counts.items():
        print(f"    {pair[0]} / {pair[1]}: {count} records -> {count * 2} entries")


if __name__ == "__main__":
    main()
