"""Tests for custom Inspect AI metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
from inspect_ai.scorer import SampleScore, Score

from prefill_awareness_audit.scoring.metrics import (
    _cohens_d,
    _fisher_z_ci,
    _mean_ci,
    _wilson_ci,
    awareness_rates,
    confidence_stats,
    coupling_metrics,
    diagnostic_tag_rates,
    latent_awareness_stats,
    spontaneous_rate_metric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ss(value: dict, metadata: dict | None = None) -> SampleScore:
    """Create a SampleScore with given value dict and optional metadata."""
    return SampleScore(
        score=Score(value=value, metadata=metadata),
        sample_id="test",
    )


# ---------------------------------------------------------------------------
# Statistical helper tests
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_zero_successes(self):
        lo, hi = _wilson_ci(0, 10)
        assert lo == 0.0
        assert 0.0 < hi < 0.5  # upper bound is non-trivial

    def test_all_successes(self):
        lo, hi = _wilson_ci(10, 10)
        assert 0.5 < lo < 1.0
        assert hi == 1.0

    def test_half(self):
        lo, hi = _wilson_ci(5, 10)
        assert 0.0 < lo < 0.5
        assert 0.5 < hi < 1.0
        # Symmetric around 0.5 (approximately)
        assert abs((lo + hi) / 2 - 0.5) < 0.05

    def test_empty(self):
        lo, hi = _wilson_ci(0, 0)
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_large_n_narrows(self):
        lo_small, hi_small = _wilson_ci(50, 100)
        lo_large, hi_large = _wilson_ci(500, 1000)
        # Same proportion, larger n -> narrower CI
        assert (hi_large - lo_large) < (hi_small - lo_small)


class TestMeanCI:
    def test_basic(self):
        m, med, s, ci_lo, ci_hi = _mean_ci([10.0, 20.0, 30.0])
        assert m == pytest.approx(20.0)
        assert med == pytest.approx(20.0)
        assert ci_lo < 20.0
        assert ci_hi > 20.0

    def test_empty(self):
        m, med, s, ci_lo, ci_hi = _mean_ci([])
        assert math.isnan(m)
        assert math.isnan(med)
        assert math.isnan(s)

    def test_single_value(self):
        m, med, s, ci_lo, ci_hi = _mean_ci([5.0])
        assert m == 5.0
        assert med == 5.0
        assert s == 0.0
        # CI collapses to point
        assert ci_lo == 5.0
        assert ci_hi == 5.0


class TestFisherZCI:
    def test_moderate_correlation(self):
        lo, hi = _fisher_z_ci(0.5, 50)
        assert lo < 0.5
        assert hi > 0.5
        assert lo > 0.0
        assert hi < 1.0

    def test_small_n_returns_nan(self):
        lo, hi = _fisher_z_ci(0.5, 3)
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_zero_correlation(self):
        lo, hi = _fisher_z_ci(0.0, 100)
        # CI should be symmetric around 0
        assert lo < 0.0
        assert hi > 0.0
        assert abs(lo + hi) < 0.01

    def test_near_one(self):
        lo, hi = _fisher_z_ci(0.99, 50)
        assert lo > 0.9
        assert hi <= 1.0


class TestCohensD:
    def test_identical_means(self):
        d = _cohens_d([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        assert d == 0.0

    def test_known_effect(self):
        # Two groups with clear separation
        g1 = [10.0, 11.0, 12.0, 13.0, 14.0]
        g2 = [0.0, 1.0, 2.0, 3.0, 4.0]
        d = _cohens_d(g1, g2)
        # Large positive effect
        assert d > 3.0

    def test_insufficient_data(self):
        d = _cohens_d([1.0], [2.0])
        assert math.isnan(d)

    def test_sign(self):
        d = _cohens_d([10.0, 11.0, 12.0], [20.0, 21.0, 22.0])
        assert d < 0  # group1 mean < group2 mean


# ---------------------------------------------------------------------------
# Awareness rates
# ---------------------------------------------------------------------------


class TestAwarenessRates:
    def test_all_self(self):
        scores = [_ss({"attribution": 1}) for _ in range(10)]
        result = awareness_rates()(scores)
        assert result["attribution_self_rate"] == 1.0
        assert result["attribution_not_self_rate"] == 0.0
        assert result["attribution_uncertain_rate"] == 0.0
        assert result["attribution_n_samples"] == 10.0

    def test_mixed(self):
        scores = (
            [_ss({"attribution": 1}) for _ in range(3)]
            + [_ss({"attribution": 0}) for _ in range(5)]
            + [_ss({"attribution": -1}) for _ in range(2)]
        )
        result = awareness_rates()(scores)
        assert result["attribution_self_rate"] == pytest.approx(0.3)
        assert result["attribution_not_self_rate"] == pytest.approx(0.5)
        assert result["attribution_uncertain_rate"] == pytest.approx(0.2)
        assert result["attribution_n_samples"] == 10.0

    def test_parse_failures_excluded(self):
        scores = (
            [_ss({"attribution": 1}) for _ in range(4)]
            + [_ss({"attribution": -2}) for _ in range(6)]  # parse failures
        )
        result = awareness_rates()(scores)
        assert result["attribution_self_rate"] == 1.0
        assert result["attribution_n_samples"] == 4.0

    def test_wilson_ci_present(self):
        scores = [_ss({"attribution": 0}) for _ in range(20)]
        result = awareness_rates()(scores)
        assert "attribution_not_self_rate_ci_lo" in result
        assert "attribution_not_self_rate_ci_hi" in result
        assert result["attribution_not_self_rate_ci_lo"] > 0.5
        assert result["attribution_not_self_rate_ci_hi"] == pytest.approx(1.0)

    def test_empty_after_filter(self):
        scores = [_ss({"attribution": -2}) for _ in range(5)]
        result = awareness_rates()(scores)
        assert math.isnan(result["attribution_self_rate"])
        assert result["attribution_n_samples"] == 0.0


# ---------------------------------------------------------------------------
# Confidence stats
# ---------------------------------------------------------------------------


class TestConfidenceStats:
    def test_basic(self):
        scores = [_ss({"prefill_confidence": v}) for v in [40, 60, 80]]
        result = confidence_stats()(scores)
        assert result["confidence_mean"] == pytest.approx(60.0)
        assert result["confidence_median"] == pytest.approx(60.0)
        assert result["confidence_n_samples"] == 3.0

    def test_sentinels_excluded(self):
        scores = [
            _ss({"prefill_confidence": 50}),
            _ss({"prefill_confidence": -1}),  # sentinel
            _ss({"prefill_confidence": 70}),
        ]
        result = confidence_stats()(scores)
        assert result["confidence_mean"] == pytest.approx(60.0)
        assert result["confidence_n_samples"] == 2.0

    def test_all_sentinels(self):
        scores = [_ss({"prefill_confidence": -1}) for _ in range(5)]
        result = confidence_stats()(scores)
        assert math.isnan(result["confidence_mean"])
        assert result["confidence_n_samples"] == 0.0

    def test_ci_present(self):
        scores = [_ss({"prefill_confidence": v}) for v in [40, 60, 80]]
        result = confidence_stats()(scores)
        assert result["confidence_mean_ci_lo"] < 60.0
        assert result["confidence_mean_ci_hi"] > 60.0


# ---------------------------------------------------------------------------
# Spontaneous rate
# ---------------------------------------------------------------------------


class TestSpontaneousRate:
    def test_basic(self):
        scores = (
            [_ss({"spontaneous_detected": 1}) for _ in range(3)]
            + [_ss({"spontaneous_detected": 0}) for _ in range(7)]
        )
        result = spontaneous_rate_metric()(scores)
        assert result["spontaneous_rate"] == pytest.approx(0.3)
        assert result["spontaneous_n_samples"] == 10.0

    def test_wilson_ci(self):
        scores = [_ss({"spontaneous_detected": 1}) for _ in range(5)] + [
            _ss({"spontaneous_detected": 0}) for _ in range(5)
        ]
        result = spontaneous_rate_metric()(scores)
        assert result["spontaneous_rate_ci_lo"] < 0.5
        assert result["spontaneous_rate_ci_hi"] > 0.5


# ---------------------------------------------------------------------------
# Latent awareness stats
# ---------------------------------------------------------------------------


class TestLatentAwarenessStats:
    def test_basic(self):
        scores = [_ss({"latent_awareness": v}) for v in [0.2, 0.4, 0.6]]
        result = latent_awareness_stats()(scores)
        assert result["latent_awareness_mean"] == pytest.approx(0.4)
        assert result["latent_awareness_median"] == pytest.approx(0.4)
        assert result["latent_awareness_n_samples"] == 3.0

    def test_sentinels_excluded(self):
        scores = [
            _ss({"latent_awareness": 0.5}),
            _ss({"latent_awareness": -1}),  # sentinel
        ]
        result = latent_awareness_stats()(scores)
        assert result["latent_awareness_mean"] == pytest.approx(0.5)
        assert result["latent_awareness_n_samples"] == 1.0

    def test_ci_and_std(self):
        scores = [_ss({"latent_awareness": v}) for v in [0.2, 0.4, 0.6, 0.8]]
        result = latent_awareness_stats()(scores)
        assert result["latent_awareness_std"] > 0
        assert result["latent_awareness_mean_ci_lo"] < 0.5
        assert result["latent_awareness_mean_ci_hi"] > 0.5


# ---------------------------------------------------------------------------
# Diagnostic tag rates
# ---------------------------------------------------------------------------


class TestDiagnosticTagRates:
    def test_basic(self):
        scores = [
            _ss({}, metadata={"diagnostic_tags": ["style_mismatch"], "diagnostic_turn": 2}),
            _ss({}, metadata={"diagnostic_tags": ["style_mismatch", "other"], "diagnostic_turn": 4}),
            _ss({}, metadata={"diagnostic_tags": [], "diagnostic_turn": None}),
        ]
        result = diagnostic_tag_rates()(scores)
        assert result["diagnostic_style_mismatch_rate"] == pytest.approx(2 / 3)
        assert result["diagnostic_other_rate"] == pytest.approx(1 / 3)
        assert result["diagnostic_formatting_artifact_rate"] == pytest.approx(0.0)
        assert result["diagnostic_n_samples"] == 3.0

    def test_parse_failures_excluded(self):
        scores = [
            _ss({}, metadata={"diagnostic_tags": ["style_mismatch"], "diagnostic_turn": 2}),
            _ss({}, metadata={"diagnostic_tags": [], "diagnostic_parse_failed": True}),
        ]
        result = diagnostic_tag_rates()(scores)
        assert result["diagnostic_style_mismatch_rate"] == 1.0
        assert result["diagnostic_n_samples"] == 1.0

    def test_flagged_turn_stats(self):
        scores = [
            _ss({}, metadata={"diagnostic_tags": ["other"], "diagnostic_turn": 2}),
            _ss({}, metadata={"diagnostic_tags": ["other"], "diagnostic_turn": 4}),
        ]
        result = diagnostic_tag_rates()(scores)
        assert result["diagnostic_flagged_turn_mean"] == pytest.approx(3.0)
        assert result["diagnostic_flagged_turn_std"] > 0

    def test_wilson_ci_on_tags(self):
        scores = [
            _ss({}, metadata={"diagnostic_tags": ["style_mismatch"]}),
            _ss({}, metadata={"diagnostic_tags": []}),
        ]
        result = diagnostic_tag_rates()(scores)
        assert "diagnostic_style_mismatch_rate_ci_lo" in result
        assert "diagnostic_style_mismatch_rate_ci_hi" in result


# ---------------------------------------------------------------------------
# Coupling metrics
# ---------------------------------------------------------------------------


class TestCouplingMetrics:
    def test_conditional_means(self):
        scores = [
            _ss({"attribution": 1, "aligned": 0.9, "prefill_confidence": 20}),
            _ss({"attribution": 1, "aligned": 0.8, "prefill_confidence": 30}),
            _ss({"attribution": 0, "aligned": 0.5, "prefill_confidence": 80}),
            _ss({"attribution": 0, "aligned": 0.6, "prefill_confidence": 70}),
        ]
        result = coupling_metrics(["aligned"])(scores)
        assert result["coupling_aligned_given_self"] == pytest.approx(0.85)
        assert result["coupling_aligned_given_not_self"] == pytest.approx(0.55)
        assert result["coupling_aligned_n_samples"] == 4.0

    def test_cohens_d(self):
        scores = [
            _ss({"attribution": 1, "aligned": 0.9, "prefill_confidence": 20}),
            _ss({"attribution": 1, "aligned": 0.8, "prefill_confidence": 30}),
            _ss({"attribution": 1, "aligned": 0.85, "prefill_confidence": 25}),
            _ss({"attribution": 0, "aligned": 0.5, "prefill_confidence": 80}),
            _ss({"attribution": 0, "aligned": 0.6, "prefill_confidence": 70}),
            _ss({"attribution": 0, "aligned": 0.55, "prefill_confidence": 75}),
        ]
        result = coupling_metrics(["aligned"])(scores)
        # Large positive d (self group has higher scores)
        assert result["coupling_aligned_cohens_d"] > 1.0

    def test_correlation_with_ci(self):
        # Negative correlation: higher confidence -> lower score
        scores = [
            _ss({"attribution": 1, "score": 0.9, "prefill_confidence": 10}),
            _ss({"attribution": 0, "score": 0.5, "prefill_confidence": 50}),
            _ss({"attribution": 0, "score": 0.2, "prefill_confidence": 90}),
            _ss({"attribution": 0, "score": 0.3, "prefill_confidence": 80}),
            _ss({"attribution": 1, "score": 0.8, "prefill_confidence": 20}),
        ]
        result = coupling_metrics(["score"])(scores)
        assert result["coupling_score_confidence_corr"] < 0
        assert result["coupling_score_confidence_corr_ci_lo"] < result["coupling_score_confidence_corr"]
        assert result["coupling_score_confidence_corr_ci_hi"] > result["coupling_score_confidence_corr"]

    def test_insufficient_for_correlation(self):
        scores = [
            _ss({"attribution": 1, "aligned": 0.9, "prefill_confidence": 20}),
            _ss({"attribution": 0, "aligned": 0.5, "prefill_confidence": 80}),
        ]
        result = coupling_metrics(["aligned"])(scores)
        assert math.isnan(result["coupling_aligned_confidence_corr"])
        assert math.isnan(result["coupling_aligned_confidence_corr_ci_lo"])

    def test_parse_failures_excluded(self):
        scores = [
            _ss({"attribution": 1, "aligned": 0.9, "prefill_confidence": 20}),
            _ss({"attribution": -2, "aligned": 0.5, "prefill_confidence": 80}),  # parse failure
        ]
        result = coupling_metrics(["aligned"])(scores)
        assert result["coupling_aligned_given_self"] == pytest.approx(0.9)
        assert math.isnan(result["coupling_aligned_given_not_self"])
        assert result["coupling_aligned_n_samples"] == 1.0

    def test_multiple_fields(self):
        scores = [
            _ss({"attribution": 1, "aligned": 0.9, "confession": 0.1, "prefill_confidence": 20}),
            _ss({"attribution": 0, "aligned": 0.5, "confession": 0.4, "prefill_confidence": 80}),
            _ss({"attribution": 0, "aligned": 0.6, "confession": 0.3, "prefill_confidence": 70}),
        ]
        result = coupling_metrics(["aligned", "confession"])(scores)
        assert "coupling_aligned_given_self" in result
        assert "coupling_confession_given_self" in result
        assert "coupling_aligned_cohens_d" in result
        assert "coupling_confession_cohens_d" in result
