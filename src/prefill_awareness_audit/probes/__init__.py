"""Probes -- Inspect solvers appended after generate() to measure awareness."""

from collections.abc import Callable

from inspect_ai.solver import Solver

from .awareness import awareness_probe
from .diagnostic import diagnostic_probe
from .fork import forked_probes
from .latent import latent_probe
from .spontaneous import detect_spontaneous_awareness

PROBE_FACTORIES: dict[str, Callable[[], Solver]] = {
    "awareness": awareness_probe,
    "latent": latent_probe,
    "diagnostic": diagnostic_probe,
}

__all__ = [
    "PROBE_FACTORIES",
    "awareness_probe",
    "diagnostic_probe",
    "detect_spontaneous_awareness",
    "forked_probes",
    "latent_probe",
]
