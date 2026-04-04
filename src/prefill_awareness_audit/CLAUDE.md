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

## Testing
- pytest + pytest-asyncio -- Inspect AI solvers are async
- tests/ at repo root, not inside src/ -- standard for src layout
- Integration tests use Inspect's eval runner with limit=2

## Build + publish
- setuptools with src layout -- broad compatibility for pip install
- py.typed marker -- PEP 561 type-checking support
