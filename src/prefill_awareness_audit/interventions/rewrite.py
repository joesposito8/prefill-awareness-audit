"""Model-based contrastive style rewrite intervention.

Uses Inspect AI's ``get_model().generate()`` to rewrite assistant messages
so they match the style of surrounding context while preserving factual content.
Two-level disk cache avoids redundant model calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from inspect_ai.model import ChatMessageUser, get_model

from prefill_awareness_audit.interventions.prompts import (
    REWRITE_PROMPT,
    STYLE_CARD_PROMPT,
)

if TYPE_CHECKING:
    from inspect_ai.dataset import Sample
    from inspect_ai.model import Model

    from prefill_awareness_audit.types import RewritePolicy

logger = logging.getLogger(__name__)

DEFAULT_REWRITER_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_CACHE_DIR = Path(".paa_cache")


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------


def _cache_key(*components: str) -> str:
    """SHA-256 hash of joined components."""
    return hashlib.sha256(":".join(components).encode()).hexdigest()


def _cache_get(key: str, cache_dir: Path, subdirectory: str) -> str | None:
    """Read a cached value from disk. Returns None on miss."""
    path = cache_dir / subdirectory / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("value")
    except (json.JSONDecodeError, OSError):
        return None


def _cache_put(
    key: str,
    value: str,
    cache_dir: Path,
    subdirectory: str,
    metadata: dict | None = None,
) -> None:
    """Write a value to the disk cache."""
    directory = cache_dir / subdirectory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.json"
    data = {"value": value}
    if metadata:
        data["metadata"] = metadata
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Style context and card
# ---------------------------------------------------------------------------


def _collect_style_context(
    messages: list,
    target_indices: list[int],
) -> tuple[list[str], list[int]]:
    """Collect 2-3 assistant messages NOT in target_indices as style exemplars.

    Walks backward from the end of the message list to find the most recent
    non-target assistant messages.

    Returns:
        (context_texts, context_indices) for cache keying and style analysis.
    """
    target_set = set(target_indices)
    context_texts: list[str] = []
    context_indices: list[int] = []

    for i in range(len(messages) - 1, -1, -1):
        if i in target_set:
            continue
        msg = messages[i]
        if msg.role == "assistant":
            text = msg.text
            if text and text.strip():
                context_texts.append(text)
                context_indices.append(i)
            if len(context_texts) >= 3:
                break

    # Reverse to maintain chronological order
    context_texts.reverse()
    context_indices.reverse()
    return context_texts, context_indices


async def _build_style_card(
    context_texts: list[str],
    model: Model,
) -> str:
    """Prompt the rewriter model to extract a style card from context messages.

    If no context is available, returns a generic neutral style card.
    """
    if not context_texts:
        return (
            "Formality: neutral\n"
            "Sentence structure: mixed\n"
            "Formatting: minimal markdown\n"
            "Vocabulary: standard\n"
            "Distinctive patterns: none identified"
        )

    numbered = "\n\n".join(
        f"--- Message {i + 1} ---\n{text}" for i, text in enumerate(context_texts)
    )
    prompt_text = STYLE_CARD_PROMPT.format(context_messages=numbered)

    output = await model.generate([ChatMessageUser(content=prompt_text)])
    return output.completion.strip()


# ---------------------------------------------------------------------------
# Single-message rewrite
# ---------------------------------------------------------------------------


async def _rewrite_single_message(
    text: str,
    style_card: str,
    model: Model,
) -> str:
    """Rewrite a single message to match the style card.

    One retry on empty or malformed output (truncated, refusal).
    Returns the original text if both attempts fail.
    """
    prompt_text = REWRITE_PROMPT.format(style_card=style_card, original_text=text)

    for attempt in range(2):
        try:
            output = await model.generate([ChatMessageUser(content=prompt_text)])
            result = output.completion.strip()
            if result:
                return result
            if attempt == 0:
                logger.warning("Rewrite returned empty, retrying (attempt 1/2)")
        except Exception:
            if attempt == 0:
                logger.warning("Rewrite model call failed, retrying (attempt 1/2)")
            else:
                logger.warning("Rewrite model call failed on retry, keeping original")

    logger.warning("Rewrite failed after 2 attempts, keeping original text")
    return text


# ---------------------------------------------------------------------------
# Sample-level entry point
# ---------------------------------------------------------------------------


async def rewrite_intervention(
    sample: Sample,
    indices: list[int],
    policy: RewritePolicy,
    rewriter_model: str = DEFAULT_REWRITER_MODEL,
    cache_dir: Path | None = None,
) -> Sample:
    """Apply contrastive style rewrite to a deep-copied sample.

    Args:
        sample: Original sample (not modified).
        indices: Message indices to rewrite.
        policy: Controls which roles are protected.
        rewriter_model: Inspect AI model string for the rewriter.
        cache_dir: Directory for disk caches. Defaults to ``.paa_cache``.

    Returns:
        Deep copy of sample with rewritten messages.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    modified = sample.model_copy(deep=True)
    messages = modified.input
    if not isinstance(messages, list):
        return modified

    model = get_model(rewriter_model)
    sample_id = str(sample.id or "unknown")

    # Collect style context and build/cache the style card
    context_texts, context_indices = _collect_style_context(messages, indices)
    sorted_ctx = sorted(context_indices)
    style_card_cache_key = _cache_key(
        sample_id, str(sorted_ctx), rewriter_model
    )

    style_card = _cache_get(style_card_cache_key, cache_dir, "style_cards")
    if style_card is None:
        style_card = await _build_style_card(context_texts, model)
        _cache_put(
            style_card_cache_key,
            style_card,
            cache_dir,
            "style_cards",
            metadata={
                "sample_id": sample_id,
                "context_indices": sorted_ctx,
                "model": rewriter_model,
            },
        )

    style_card_hash = hashlib.sha256(style_card.encode()).hexdigest()[:16]

    # Rewrite each target message
    for idx in indices:
        if idx >= len(messages):
            continue
        msg = messages[idx]
        if msg.role in policy.protected_roles:
            continue

        original_text = msg.text
        if not original_text or not original_text.strip():
            continue

        rewrite_cache_key = _cache_key(
            sample_id, str(idx), style_card_hash, rewriter_model
        )
        cached = _cache_get(rewrite_cache_key, cache_dir, "rewrites")
        if cached is not None:
            msg.text = cached
            continue

        rewritten = await _rewrite_single_message(original_text, style_card, model)
        msg.text = rewritten
        _cache_put(
            rewrite_cache_key,
            rewritten,
            cache_dir,
            "rewrites",
            metadata={
                "sample_id": sample_id,
                "message_index": idx,
                "style_card_hash": style_card_hash,
                "model": rewriter_model,
            },
        )

    return modified
