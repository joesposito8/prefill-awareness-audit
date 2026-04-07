"""Deterministic cleanup operations for intervention transformers.

All functions are pure str -> str transforms composed into a pipeline.
The module entry point ``deterministic_intervention`` deep-copies a Sample
and applies the pipeline to messages at the specified indices.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inspect_ai.dataset import Sample

    from prefill_awareness_audit.types import RewritePolicy


# ---------------------------------------------------------------------------
# Individual cleanup operations (str -> str)
# ---------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Strip trailing spaces per line, normalize line endings, collapse blanks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def normalize_markdown(text: str) -> str:
    """Normalize header spacing, list markers, and code fence style."""
    # Headers: ensure exactly one space after #
    text = re.sub(
        r"^(#{1,6})[ \t]*(?=\S)", lambda m: m.group(1) + " ", text, flags=re.MULTILINE
    )
    # List markers: normalize * and + to -
    text = re.sub(r"^(\s*)[*+](\s)", lambda m: m.group(1) + "-" + m.group(2), text, flags=re.MULTILINE)
    # Code fences: normalize ~~~ to ```
    text = re.sub(r"^~{3,}", "```", text, flags=re.MULTILINE)
    return text


def normalize_thinking_tags(text: str) -> str:
    """Normalize all thinking tag variants to <thinking>/</thinking>."""
    # Opening tags: <THINKING>, <Thinking>, <antThinking>, etc.
    text = re.sub(r"<(?:ant)?[Tt][Hh]?[Ii]?[Nn]?[Kk]?[Ii]?[Nn]?[Gg][^>]*>", "<thinking>", text)
    # More precise: match common variants
    text = re.sub(r"<(?:ant)?(?:thinking|THINKING|Thinking)[^>]*>", "<thinking>", text)
    # Closing tags
    text = re.sub(r"</(?:ant)?(?:thinking|THINKING|Thinking)[^>]*>", "</thinking>", text)
    return text


def strip_provenance_markers(text: str) -> str:
    """Remove message IDs, ISO timestamps, and model metadata."""
    # Message/response/call IDs: resp_abc123, call_abc123, msg_abc123
    text = re.sub(r"(?:resp_|call_|msg_)[a-f0-9]{16,32}", "", text)
    # ISO timestamps: 2024-01-15T10:30:00Z or with offset
    text = re.sub(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})",
        "",
        text,
    )
    # Model metadata: "model": "gpt-4" or "model": "claude-3-opus"
    text = re.sub(r'"model"\s*:\s*"[^"]*"', "", text)
    return text


def fix_serialization_artifacts(text: str) -> str:
    """Fix double-escaped characters and encoding issues."""
    # Double-escaped newlines (literal \\n in text, not actual newlines)
    text = text.replace("\\\\n", "\n")
    # Escaped forward slashes
    text = text.replace("\\/", "/")
    # Common unicode escapes
    text = re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda m: chr(int(m.group(1), 16)),
        text,
    )
    return text


# ---------------------------------------------------------------------------
# Composed pipelines
# ---------------------------------------------------------------------------


def apply_deterministic_cleanup(text: str) -> str:
    """Apply all cleanup operations in fixed order."""
    text = normalize_whitespace(text)
    text = normalize_markdown(text)
    text = normalize_thinking_tags(text)
    text = strip_provenance_markers(text)
    text = fix_serialization_artifacts(text)
    return text


def strip_provenance_only(text: str) -> str:
    """Apply only provenance marker removal (for protected roles in global scope)."""
    return strip_provenance_markers(text)


# ---------------------------------------------------------------------------
# Sample-level entry point
# ---------------------------------------------------------------------------


def deterministic_intervention(
    sample: Sample,
    indices: list[int],
    policy: RewritePolicy,
    provenance_only_indices: list[int] | None = None,
) -> Sample:
    """Apply deterministic cleanup to a deep-copied sample at specified indices.

    Args:
        sample: Original sample (not modified).
        indices: Message indices to apply full cleanup to.
        policy: Controls which roles are protected.
        provenance_only_indices: Indices that get only provenance stripping
            (used for protected roles in global scope).

    Returns:
        Deep copy of sample with cleaned messages.
    """
    modified = sample.model_copy(deep=True)
    messages = modified.input
    if not isinstance(messages, list):
        return modified

    provenance_only = set(provenance_only_indices or [])

    for idx in indices:
        if idx >= len(messages):
            continue
        msg = messages[idx]
        if msg.role in policy.protected_roles:
            continue
        msg.text = apply_deterministic_cleanup(msg.text)

    for idx in provenance_only:
        if idx >= len(messages):
            continue
        msg = messages[idx]
        msg.text = strip_provenance_only(msg.text)

    return modified
