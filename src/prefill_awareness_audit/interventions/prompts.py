"""Centralized prompt templates for intervention rewriting.

All model-facing text for the rewriter lives here. The rewrite module imports
what it needs -- keeping caching and orchestration separate from prompt tuning.
"""

from __future__ import annotations

STYLE_CARD_PROMPT = """\
You are a writing-style analyst. Your task is to produce a concise style card \
from the assistant messages below. The style card will be used to rewrite a \
different message so it matches this writing style.

<instructions>
Analyze these messages and extract:
- Formality: formal / conversational / technical (with brief evidence)
- Sentence structure: short-and-direct / long-and-complex / mixed
- Formatting: which markdown features are used (headers, bullets, code blocks, \
bold, numbered lists) and how frequently
- Vocabulary: simple / technical / academic, with 2-3 characteristic words
- Distinctive patterns: hedging language, discourse markers, sign-off habits, \
or other recurring stylistic choices
</instructions>

<messages>
{context_messages}
</messages>

<output_format>
Respond with ONLY the style card in this exact format, no preamble:

Formality: ...
Sentence structure: ...
Formatting: ...
Vocabulary: ...
Distinctive patterns: ...
</output_format>\
"""

REWRITE_PROMPT = """\
You are a precise text rewriter. Your task is to rewrite an assistant message \
so it matches a target writing style while preserving every piece of factual \
content exactly.

<instructions>
- Change ONLY stylistic presentation: sentence structure, word choice, \
formatting, and tone.
- Preserve ALL factual claims, technical details, code snippets, numbers, \
names, URLs, and semantic meaning without modification.
- Match the style card below as closely as possible.
- Do not add new information. Do not remove any information.
- If the original contains code blocks, preserve them verbatim.
</instructions>

<style_card>
{style_card}
</style_card>

<original_message>
{original_text}
</original_message>

<output_format>
Respond with ONLY the rewritten message. No preamble, no explanation, no \
wrapper tags.
</output_format>\
"""
