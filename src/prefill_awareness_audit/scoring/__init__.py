"""Scoring -- audit scorer wrapper and custom Inspect metrics."""

from .audit_scorer import audit_scorer
from .metrics import (
    awareness_rates,
    confidence_stats,
    coupling_metrics,
    diagnostic_tag_rates,
    latent_awareness_stats,
    spontaneous_rate_metric,
)

__all__ = [
    "audit_scorer",
    "awareness_rates",
    "confidence_stats",
    "coupling_metrics",
    "diagnostic_tag_rates",
    "latent_awareness_stats",
    "spontaneous_rate_metric",
]
