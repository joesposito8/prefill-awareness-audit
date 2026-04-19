"""Probes -- Inspect solvers appended after generate() to measure awareness."""

from collections.abc import Callable

from inspect_ai.solver import Solver

from .awareness import awareness_probe
from .diagnostic import diagnostic_probe
from .fork import forked_probes
from .multiple_choice import multiple_choice_probe
from .persona import persona_probe
from .reflection import reflection_probe
from .spontaneous import detect_spontaneous_awareness

PROBE_FACTORIES: dict[str, Callable[[], Solver]] = {
    "awareness": awareness_probe,
    "multiple_choice": multiple_choice_probe,
    "reflection": reflection_probe,
    "persona": persona_probe,
    "diagnostic": diagnostic_probe,
}

__all__ = [
    "PROBE_FACTORIES",
    "awareness_probe",
    "detect_spontaneous_awareness",
    "diagnostic_probe",
    "forked_probes",
    "multiple_choice_probe",
    "persona_probe",
    "reflection_probe",
]
