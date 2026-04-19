"""Scoring -- audit scorer wrapper and custom Inspect metrics."""

from .audit_scorer import audit_scorer
from .metrics import (
    confidence_stats,
    coupling_metrics,
    diagnostic_tag_rates,
    ground_truth_metrics,
    latent_rates,
    multiple_choice_rates,
    spontaneous_rate_metric,
)

__all__ = [
    "audit_scorer",
    "confidence_stats",
    "coupling_metrics",
    "diagnostic_tag_rates",
    "ground_truth_metrics",
    "latent_rates",
    "multiple_choice_rates",
    "spontaneous_rate_metric",
]
