# prefill_awareness_audit zone

## Language + runtime
- Python >=3.11 -- f-string improvements, StrEnum, tomllib; Inspect AI requires 3.11+

## Framework + key libs
- Inspect AI -- native solver/scorer/task composition; the audit wraps existing evals rather than replacing them
- anthropic SDK -- direct API access for contrastive rewrites (model-based interventions)
- numpy + matplotlib -- scorecard computation and report figures

## File layout
- src/ layout -- clean install semantics, no ambient imports from project root
- Subpackages by responsibility: probes/, interventions/, scoring/, analysis/
- types.py at package root -- all user-facing types in one place
- task_factory.py -- single composition entry point (audit_task)

## Probes architecture
- probes/prompts.py -- centralized prompt templates, domain constants (DIAGNOSTIC_TAGS + VALID_TAGS, AWARENESS_PHRASES, LATENT_CANONICAL_OPTIONS, AttributionLabel + ATTRIBUTION_ORDINAL_SCORE + ATTRIBUTION_TO_THREE_WAY), output schemas (DIAGNOSTIC_SCHEMA, LATENT_JUDGE_SCHEMA), and the two per-sample prompt builders -- build_awareness_prompt() (flips the 5-point option polarity per sample_id) and build_diagnostic_prompt() (shuffles the 6 diagnostic tags per sample_id). All model-facing text lives here.
- Forked execution via fork.py: forked_probes() uses Inspect AI's fork() to run each probe on an independent deepcopy of TaskState in parallel. No probe sees another's Q&A.
- awareness.py -- graded 5-point authorship-certainty probe (single ATTRIBUTION: marker, no separate confidence). Defaults to the per-sample polarity flip via build_awareness_prompt(); passing question_template= explicitly pins a single template and skips randomisation. Stores reversed_order in metadata. Parsers: _parse_awareness (primary, returns Attribution three-way + AttributionLabel graded), _parse_attribution (legacy three-way fallback for ANSWER: self style logs). The scorer derives attribution_score (-2..+2 ordinal) from AttributionLabel via ATTRIBUTION_ORDINAL_SCORE -- see docs/design-decisions.md §3.2.
- reflection.py -- free-text first-person reflection probe. Neutral prompt, LLM-judge classification into none/suspicious/convinced via shared _util.classify_judgement() helper.
- persona.py -- user-persona role-swap probe (Cywiński technique). Model writes the user's next turn; same 3-way judge schema.
- multiple_choice.py -- opt-in MCQ latent probe (preserved as A/B baseline). Per-sample letter shuffle over LATENT_CANONICAL_OPTIONS via the shared per_sample_seed(); deterministic parser.
- diagnostic.py -- suspicious turn identification with reason tags (JSON output). Defaults to per-sample tag-order shuffle via build_diagnostic_prompt() (base_seed=43 so it is independent of awareness at base_seed=42); passing question_template= explicitly pins a single prompt. Stores tag_order in metadata.
- spontaneous.py is a pure function (not a solver) -- called by audit_scorer at scoring time.
- _util.py: append_tool_stubs() (tool stubs before probe questions), classify_judgement() (shared LLM-judge call for reflection+persona), coerce_enum() (shared enum coercion for metadata reads), _extract_json_object() (layered JSON parser), per_sample_seed() / per_sample_rng() (single SHA-1-based seed formula shared by every probe that presents ordered options -- SHA-1 rather than builtin hash() because PEP 456 salts hash() per interpreter). All model-agnostic via Inspect AI.
- Parsers use Inspect AI builtins: ANSWER_PATTERN_LINE (legacy three-way fallback), ResponseSchema+json_schema (diagnostic, latent judge).

## Testing
- pytest + pytest-asyncio -- Inspect AI solvers are async
- tests/ at repo root, not inside src/ -- standard for src layout
- Integration tests use Inspect's eval runner with limit=2
- 243 tests covering probes, interventions, scoring, and integration

## Build + publish
- setuptools with src layout -- broad compatibility for pip install
- py.typed marker -- PEP 561 type-checking support
