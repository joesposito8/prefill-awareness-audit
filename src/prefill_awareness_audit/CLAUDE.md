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
- probes/prompts.py -- centralized prompt templates, domain constants (VALID_TAGS, AWARENESS_PHRASES, LATENT_CANONICAL_OPTIONS, AttributionLabel + ATTRIBUTION_ORDINAL_SCORE + ATTRIBUTION_TO_THREE_WAY), output schemas (DIAGNOSTIC_SCHEMA, LATENT_JUDGE_SCHEMA). All model-facing text lives here.
- Forked execution via fork.py: forked_probes() uses Inspect AI's fork() to run each probe on an independent deepcopy of TaskState in parallel. No probe sees another's Q&A.
- awareness.py -- graded 5-point authorship-certainty probe (single ATTRIBUTION: marker, no separate confidence). Parsers: _parse_awareness (primary, returns Attribution three-way + AttributionLabel graded), _parse_attribution (legacy three-way fallback for ANSWER: self style logs). The scorer derives attribution_score (-2..+2 ordinal) from AttributionLabel via ATTRIBUTION_ORDINAL_SCORE -- see docs/design-decisions.md §3.2.
- reflection.py -- free-text first-person reflection probe. Neutral prompt, LLM-judge classification into none/suspicious/convinced via shared _util.classify_judgement() helper.
- persona.py -- user-persona role-swap probe (Cywiński technique). Model writes the user's next turn; same 3-way judge schema.
- multiple_choice.py -- opt-in MCQ latent probe (preserved as A/B baseline). Shuffled 9-option menu, deterministic parser.
- diagnostic.py -- suspicious turn identification with reason tags (JSON output).
- spontaneous.py is a pure function (not a solver) -- called by audit_scorer at scoring time.
- _util.py: append_tool_stubs() (tool stubs before probe questions), classify_judgement() (shared LLM-judge call for reflection+persona), coerce_enum() (shared enum coercion for metadata reads), _extract_json_object() (layered JSON parser). All model-agnostic via Inspect AI.
- Parsers use Inspect AI builtins: ANSWER_PATTERN_LINE (legacy three-way fallback), ResponseSchema+json_schema (diagnostic, latent judge).

## Testing
- pytest + pytest-asyncio -- Inspect AI solvers are async
- tests/ at repo root, not inside src/ -- standard for src layout
- Integration tests use Inspect's eval runner with limit=2
- 133 tests covering probes, interventions, scoring, and integration

## Build + publish
- setuptools with src layout -- broad compatibility for pip install
- py.typed marker -- PEP 561 type-checking support
