"""Probes -- Inspect solvers appended after generate() to measure awareness."""

from .awareness import awareness_probe
from .counterfactual import counterfactual_probe
from .diagnostic import diagnostic_probe
from .fork import forked_probes
from .spontaneous import detect_spontaneous_awareness

__all__ = [
    "awareness_probe",
    "counterfactual_probe",
    "diagnostic_probe",
    "detect_spontaneous_awareness",
    "forked_probes",
]
