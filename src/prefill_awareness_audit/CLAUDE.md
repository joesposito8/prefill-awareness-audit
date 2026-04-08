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
- probes/prompts.py -- centralized prompt templates, domain constants (VALID_TAGS, AWARENESS_PHRASES), and structured output schemas. All model-facing text lives here.
- Forked execution via fork.py: forked_probes() uses Inspect AI's fork() to run each probe on an independent deepcopy of TaskState in parallel. No probe sees another's Q&A.
- awareness.py -- combined attribution+confidence in one prompt (ATTRIBUTION:/CONFIDENCE: markers). Contains _parse_attribution() and _parse_confidence() parsers.
- counterfactual.py -- latent awareness via indirect observation matching (OBSERVATION:/MATCH: markers)
- diagnostic.py -- suspicious turn identification with reason tags (JSON output)
- spontaneous.py is a pure function (not a solver) -- called by audit_scorer at scoring time
- _util.py:append_tool_stubs() inserts ChatMessageTool stubs before probe questions -- model-agnostic via Inspect AI
- Parsers use Inspect AI builtins: ANSWER_PATTERN_LINE (attribution fallback), str_to_float/strip_numeric_punctuation (confidence), ResponseSchema+json_schema (diagnostic)

## Testing
- pytest + pytest-asyncio -- Inspect AI solvers are async
- tests/ at repo root, not inside src/ -- standard for src layout
- Integration tests use Inspect's eval runner with limit=2
- 133 tests covering probes, interventions, scoring, and integration

## Build + publish
- setuptools with src layout -- broad compatibility for pip install
- py.typed marker -- PEP 561 type-checking support
