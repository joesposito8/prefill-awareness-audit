"""Probes -- Inspect solvers appended after generate() to measure awareness."""

from .attribution import attribution_probe
from .confidence import confidence_probe
from .diagnostic import diagnostic_probe
from .spontaneous import detect_spontaneous_awareness

__all__ = [
    "attribution_probe",
    "confidence_probe",
    "diagnostic_probe",
    "detect_spontaneous_awareness",
]
