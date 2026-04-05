# prefill-awareness-audit__master

## Current Initiative
Core runtime complete (types, probes, scoring). Next: interventions (PAA-3), then task factory (PAA-4).

## Key Files
- PROJECT_VISION.md -- scope, outcomes, non-goals
- SYSTEM_ARCHITECTURE.md -- package structure, key types
- pyproject.toml -- dependencies and build config
- src/prefill_awareness_audit/probes/prompts.py -- all prompt templates and domain constants

## Gotchas
- Probes import from centralized prompts.py -- add new prompts there, not in probe modules
- Parsers use Inspect AI internals (inspect_ai._util.pattern, inspect_ai._util.text) -- pin inspect-ai version carefully
- diagnostic_probe uses ResponseSchema for structured output with try/except fallback for models that don't support it
