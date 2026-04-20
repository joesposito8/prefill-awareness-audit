"""Post-intervention structural validation.

Validates that a modified Sample conforms to the InterventionConstraints
by comparing it against the original. Returns (passed, violations) where
violations is a list of human-readable descriptions of what went wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inspect_ai.dataset import Sample

    from prefill_awareness_audit.types import InterventionConstraints


def _content_to_text(content: Any) -> str:
    """Convert message content to a comparable text string.

    Handles both ``str`` content and ``list[Content]`` (concatenates text items).
    Mirrors the ``ChatMessageBase.text`` getter logic so comparisons are
    consistent with what the intervention code reads and writes.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if hasattr(item, "text"):
                parts.append(item.text)
        return "\n".join(parts)
    return str(content)


def validate_intervention(
    original: Sample,
    modified: Sample,
    target_indices: list[int],
    constraints: InterventionConstraints,
) -> tuple[bool, list[str]]:
    """Validate a modified sample against the original and constraints.

    Args:
        original: The unmodified sample.
        modified: The sample after intervention.
        target_indices: Indices that were targeted for modification.
        constraints: Structural invariants the modification must preserve.

    Returns:
        A tuple of (passed, violations). ``passed`` is True if all checks
        succeed. ``violations`` lists human-readable descriptions of failures.
    """
    violations: list[str] = []
    orig_messages = original.input if isinstance(original.input, list) else []
    mod_messages = modified.input if isinstance(modified.input, list) else []

    # 1. Message count preserved
    if len(orig_messages) != len(mod_messages):
        violations.append(
            f"Message count changed: {len(orig_messages)} -> {len(mod_messages)}"
        )
        # Further index-based checks are unreliable if counts differ
        return (False, violations)

    # 2. Role order preserved
    orig_roles = [m.role for m in orig_messages]
    mod_roles = [m.role for m in mod_messages]
    if orig_roles != mod_roles:
        violations.append(f"Role order changed: {orig_roles} -> {mod_roles}")

    target_set = set(target_indices)

    for i, (orig_msg, mod_msg) in enumerate(zip(orig_messages, mod_messages)):
        orig_text = _content_to_text(orig_msg.content)
        mod_text = _content_to_text(mod_msg.content)

        # 3. Protected roles untouched
        if orig_msg.role in constraints.protected_roles and orig_text != mod_text:
            violations.append(
                f"Protected role '{orig_msg.role}' at index {i} was modified"
            )

        # 4. Target-only changes: non-target messages must be byte-identical
        if i not in target_set and orig_text != mod_text:
            violations.append(
                f"Non-target message at index {i} (role={orig_msg.role}) was modified"
            )

        # 5. Tool outputs intact
        if orig_msg.role == "tool":
            if orig_text != mod_text:
                violations.append(
                    f"Tool message content at index {i} was modified"
                )
            if getattr(orig_msg, "tool_call_id", None) != getattr(
                mod_msg, "tool_call_id", None
            ):
                violations.append(
                    f"Tool message tool_call_id at index {i} was modified"
                )
            if getattr(orig_msg, "function", None) != getattr(
                mod_msg, "function", None
            ):
                violations.append(
                    f"Tool message function at index {i} was modified"
                )

        # 6. No empty messages (only if intervention caused it)
        if not mod_text.strip() and orig_text.strip():
            # Assistant messages with tool calls may have empty text content
            has_tool_calls = bool(getattr(mod_msg, "tool_calls", None))
            if not has_tool_calls:
                violations.append(f"Modified message at index {i} is empty")

    return (len(violations) == 0, violations)
