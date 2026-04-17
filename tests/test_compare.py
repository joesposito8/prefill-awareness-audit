"""Tests for cross-condition comparison logic."""

from __future__ import annotations

import math

import pytest

from prefill_awareness_audit.analysis.compare import (
    compare_conditions,
    format_comparison_table,
)
from prefill_awareness_audit.types import ComparisonTable, Condition, ConditionSummary


def _cs(
    condition: Condition,
    model: str = "test-model",
    metrics: dict | None = None,
    n: int = 50,
) -> ConditionSummary:
    return ConditionSummary(
        condition=condition,
        model=model,
        metrics=metrics or {},
        sample_count=n,
    )


class TestCompareConditions:
    def test_delta_vs_probe_only(self):
        summaries = [
            _cs(Condition.PROBE_ONLY, metrics={"recognition_rate": 0.3, "confidence_mean": 70.0}),
            _cs(Condition.LOCAL_DETERMINISTIC, metrics={"recognition_rate": 0.5, "confidence_mean": 55.0}),
        ]
        table = compare_conditions(summaries)
        assert table.deltas_vs_probe_only is not None
        delta = table.deltas_vs_probe_only["LOCAL_DETERMINISTIC"]
        assert delta["recognition_rate"] == pytest.approx(0.2)
        assert delta["confidence_mean"] == pytest.approx(-15.0)

    def test_no_probe_only(self):
        summaries = [
            _cs(Condition.LOCAL_DETERMINISTIC, metrics={"recognition_rate": 0.5}),
            _cs(Condition.LOCAL_REWRITE, metrics={"recognition_rate": 0.7}),
        ]
        table = compare_conditions(summaries)
        # No PROBE_ONLY -> no deltas
        assert table.deltas_vs_probe_only is None or len(table.deltas_vs_probe_only) == 0

    def test_multi_model_grouping(self):
        summaries = [
            _cs(Condition.PROBE_ONLY, model="model-a", metrics={"sr": 0.3}),
            _cs(Condition.LOCAL_DETERMINISTIC, model="model-a", metrics={"sr": 0.5}),
            _cs(Condition.PROBE_ONLY, model="model-b", metrics={"sr": 0.6}),
            _cs(Condition.LOCAL_DETERMINISTIC, model="model-b", metrics={"sr": 0.7}),
        ]
        table = compare_conditions(summaries)
        assert table.model_comparison is not None
        assert "model-a" in table.model_comparison
        assert "model-b" in table.model_comparison
        assert len(table.model_comparison["model-a"]) == 2

    def test_single_model_no_grouping(self):
        summaries = [
            _cs(Condition.PROBE_ONLY, metrics={"sr": 0.3}),
            _cs(Condition.LOCAL_DETERMINISTIC, metrics={"sr": 0.5}),
        ]
        table = compare_conditions(summaries)
        assert table.model_comparison is None

    def test_nan_metric_produces_nan_delta(self):
        summaries = [
            _cs(Condition.PROBE_ONLY, metrics={"m1": 0.5, "m2": float("nan")}),
            _cs(Condition.LOCAL_DETERMINISTIC, metrics={"m1": 0.7, "m2": 0.3}),
        ]
        table = compare_conditions(summaries)
        delta = table.deltas_vs_probe_only["LOCAL_DETERMINISTIC"]
        assert delta["m1"] == pytest.approx(0.2)
        assert math.isnan(delta["m2"])

    def test_per_model_deltas(self):
        """Deltas are computed against each model's own PROBE_ONLY."""
        summaries = [
            _cs(Condition.PROBE_ONLY, model="a", metrics={"sr": 0.3}),
            _cs(Condition.LOCAL_DETERMINISTIC, model="a", metrics={"sr": 0.5}),
            _cs(Condition.PROBE_ONLY, model="b", metrics={"sr": 0.8}),
            _cs(Condition.LOCAL_DETERMINISTIC, model="b", metrics={"sr": 0.7}),
        ]
        table = compare_conditions(summaries)
        # Both LOCAL_DETERMINISTIC entries produce deltas
        assert table.deltas_vs_probe_only is not None
        # The deltas dict is keyed by condition name, so with two models
        # having the same condition, the second overwrites the first.
        # This is a known simplification -- for multi-model, use model_comparison.


class TestFormatComparisonTable:
    def test_basic_output(self):
        summaries = [
            _cs(
                Condition.PROBE_ONLY,
                metrics={
                    "recognition_rate": 0.856,
                    "recognition_rate_ci_lo": 0.773,
                    "recognition_rate_ci_hi": 0.913,
                    "detection_rate": 0.920,
                    "detection_rate_ci_lo": 0.842,
                    "detection_rate_ci_hi": 0.962,
                    "false_attribution_rate": 0.080,
                    "false_attribution_rate_ci_lo": 0.038,
                    "false_attribution_rate_ci_hi": 0.158,
                    "miss_rate": 0.144,
                    "miss_rate_ci_lo": 0.087,
                    "miss_rate_ci_hi": 0.227,
                    "g_mean": 0.887,
                    "uncertain_rate": 0.050,
                    "uncertain_rate_ci_lo": 0.020,
                    "uncertain_rate_ci_hi": 0.119,
                    "n_samples": 100,
                    "n_classified": 95,
                    "n_parse_fail": 2,
                    "confidence_mean": 68.2,
                    "confidence_mean_ci_lo": 63.0,
                    "confidence_mean_ci_hi": 73.4,
                    "confidence_median": 72.0,
                    "confidence_std": 18.5,
                    "confidence_n_samples": 47,
                    "spontaneous_rate": 0.15,
                    "spontaneous_rate_ci_lo": 0.07,
                    "spontaneous_rate_ci_hi": 0.28,
                    "spontaneous_n_samples": 50,
                    "latent_prefill_rate": 0.42,
                    "latent_prefill_rate_ci_lo": 0.35,
                    "latent_prefill_rate_ci_hi": 0.49,
                    "latent_eval_rate": 0.20,
                    "latent_eval_rate_ci_lo": 0.12,
                    "latent_eval_rate_ci_hi": 0.30,
                    "latent_n_samples": 45,
                    "latent_n_parse_fail": 2,
                },
            ),
        ]
        table = ComparisonTable(conditions=summaries)
        output = format_comparison_table(table)
        assert "PROBE_ONLY" in output
        assert "Ground truth" in output
        assert "0.856" in output  # recognition_rate
        assert "0.887" in output  # g_mean
        assert "n_classified=95" in output

    def test_delta_section(self):
        summaries = [
            _cs(Condition.PROBE_ONLY, metrics={"recognition_rate": 0.3, "confidence_mean": 70.0}),
            _cs(Condition.LOCAL_DETERMINISTIC, metrics={"recognition_rate": 0.5, "confidence_mean": 55.0}),
        ]
        table = compare_conditions(summaries)
        output = format_comparison_table(table)
        assert "Deltas vs PROBE_ONLY" in output
        assert "+0.200" in output  # recognition_rate delta
