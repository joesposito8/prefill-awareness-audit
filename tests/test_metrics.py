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
    confidence_stats,
    coupling_metrics,
    diagnostic_tag_rates,
    ground_truth_metrics,
    latent_rates,
    multiple_choice_rates,
    spontaneous_rate_metric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ss(value: dict, metadata: dict | None = None) -> SampleScore:
    """Create a SampleScore with given value dict and optional metadata.

    ``attribution`` is auto-routed from ``value`` into ``metadata`` so tests
    can stay readable while matching the production convention (strings must
    live in ``Score.metadata`` or Inspect's mean_score reducer demotes them).
    """
    value = dict(value)
    meta = dict(metadata or {})
    if "attribution" in value:
        meta.setdefault("attribution", value.pop("attribution"))
    return SampleScore(
        score=Score(value=value, metadata=meta),
        sample_id="test",
    )


# ---------------------------------------------------------------------------
# Statistical helper tests
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_half(self):
        lo, hi = _wilson_ci(5, 10)
        assert 0.0 < lo < 0.5 < hi < 1.0

    def test_empty(self):
        lo, hi = _wilson_ci(0, 0)
        assert math.isnan(lo)
        assert math.isnan(hi)


class TestMeanCI:
    def test_basic(self):
        m, med, s, ci_lo, ci_hi = _mean_ci([10.0, 20.0, 30.0])
        assert m == pytest.approx(20.0)
        assert med == pytest.approx(20.0)
        assert ci_lo < 20.0 < ci_hi

    def test_empty(self):
        m, med, s, ci_lo, ci_hi = _mean_ci([])
        assert math.isnan(m)
        assert math.isnan(med)
        assert math.isnan(s)


class TestFisherZCI:
    def test_moderate_correlation(self):
        lo, hi = _fisher_z_ci(0.5, 50)
        assert 0.0 < lo < 0.5 < hi < 1.0

    def test_small_n_returns_nan(self):
        lo, hi = _fisher_z_ci(0.5, 3)
        assert math.isnan(lo)
        assert math.isnan(hi)


class TestCohensD:
    def test_known_effect(self):
        g1 = [10.0, 11.0, 12.0, 13.0, 14.0]
        g2 = [0.0, 1.0, 2.0, 3.0, 4.0]
        assert _cohens_d(g1, g2) > 3.0
        assert _cohens_d(g2, g1) < -3.0

    def test_insufficient_data(self):
        assert math.isnan(_cohens_d([1.0], [2.0]))


# ---------------------------------------------------------------------------
# Ground truth metrics
# ---------------------------------------------------------------------------


class TestGroundTruthMetrics:
    def test_mixed_confusion_matrix(self):
        scores = (
            [_ss({"attribution": "self", "prefill_authentic": True}) for _ in range(4)]
            + [_ss({"attribution": "not_self", "prefill_authentic": True}) for _ in range(1)]
            + [_ss({"attribution": "self", "prefill_authentic": False}) for _ in range(2)]
            + [_ss({"attribution": "not_self", "prefill_authentic": False}) for _ in range(3)]
            + [_ss({"attribution": "uncertain", "prefill_authentic": False}) for _ in range(2)]
            + [_ss({"attribution": None, "prefill_authentic": False}) for _ in range(1)]
        )
        result = ground_truth_metrics()(scores)

        assert result["recognition_rate"] == pytest.approx(0.8)
        assert result["miss_rate"] == pytest.approx(0.2)
        assert result["detection_rate"] == pytest.approx(0.6)
        assert result["false_attribution_rate"] == pytest.approx(0.4)
        assert result["g_mean"] == pytest.approx(math.sqrt(0.48))
        assert result["uncertain_rate"] == pytest.approx(2 / 12)

        for key in (
            "recognition_rate",
            "miss_rate",
            "detection_rate",
            "false_attribution_rate",
            "uncertain_rate",
        ):
            assert f"{key}_ci_lo" in result
            assert f"{key}_ci_hi" in result

        assert result["n_classified"] == 10.0
        assert result["n_samples"] == 13.0
        assert result["n_parse_fail"] == 1.0

    def test_all_synthetic_edge_case(self):
        scores = (
            [_ss({"attribution": "not_self", "prefill_authentic": False}) for _ in range(8)]
            + [_ss({"attribution": "self", "prefill_authentic": False}) for _ in range(2)]
        )
        result = ground_truth_metrics()(scores)
        assert result["detection_rate"] == pytest.approx(0.8)
        assert result["false_attribution_rate"] == pytest.approx(0.2)
        assert math.isnan(result["recognition_rate"])
        assert math.isnan(result["miss_rate"])
        assert math.isnan(result["g_mean"])

        empty_result = ground_truth_metrics()([])
        assert math.isnan(empty_result["detection_rate"])
        assert math.isnan(empty_result["recognition_rate"])
        assert empty_result["n_classified"] == 0.0
        assert empty_result["n_samples"] == 0.0


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

    def test_none_excluded(self):
        scores = [
            _ss({"prefill_confidence": 50}),
            _ss({"prefill_confidence": None}),
            _ss({"prefill_confidence": 70}),
        ]
        result = confidence_stats()(scores)
        assert result["confidence_mean"] == pytest.approx(60.0)
        assert result["confidence_n_samples"] == 2.0

    def test_all_none(self):
        scores = [_ss({"prefill_confidence": None}) for _ in range(5)]
        result = confidence_stats()(scores)
        assert math.isnan(result["confidence_mean"])
        assert result["confidence_n_samples"] == 0.0


# ---------------------------------------------------------------------------
# Spontaneous rate
# ---------------------------------------------------------------------------


class TestSpontaneousRate:
    def test_basic(self):
        scores = (
            [_ss({"spontaneous_detected": True}) for _ in range(3)]
            + [_ss({"spontaneous_detected": False}) for _ in range(7)]
        )
        result = spontaneous_rate_metric()(scores)
        assert result["spontaneous_rate"] == pytest.approx(0.3)
        assert result["spontaneous_n_samples"] == 10.0
        assert result["spontaneous_rate_ci_lo"] < 0.3 < result["spontaneous_rate_ci_hi"]


# ---------------------------------------------------------------------------
# Multiple-choice rates
# ---------------------------------------------------------------------------


def _mc_ss(prefill: int, eval_: int, choices: list[str]) -> SampleScore:
    """Build a SampleScore with multiple-choice probe output shape."""
    return SampleScore(
        score=Score(
            value={
                "multiple_choice_prefill": prefill,
                "multiple_choice_eval": eval_,
            },
            metadata={"multiple_choice_choices": choices},
        ),
        sample_id="test",
    )


class TestMultipleChoiceRates:
    def test_independent_rates(self):
        scores = [
            _mc_ss(1, 0, ["prefill"]),
            _mc_ss(1, 1, ["prefill", "eval"]),
            _mc_ss(0, 0, ["none"]),
        ]
        result = multiple_choice_rates()(scores)
        assert result["mc_prefill_rate"] == pytest.approx(2 / 3)
        assert result["mc_eval_rate"] == pytest.approx(1 / 3)
        assert result["mc_n_samples"] == 3.0
        for canon_id in (
            "role_play", "contradict", "prefill", "sycophancy",
            "tool_mismatch", "overreach", "outdated", "eval", "none",
        ):
            assert f"mc_option_{canon_id}_rate" in result

    def test_parse_failures_excluded(self):
        scores = [
            _mc_ss(1, 0, ["prefill"]),
            _mc_ss(-1, -1, []),  # parse failure
            _mc_ss(0, 1, ["eval"]),
        ]
        result = multiple_choice_rates()(scores)
        assert result["mc_n_samples"] == 2.0
        assert result["mc_n_parse_fail"] == 1.0
        assert result["mc_prefill_rate"] == pytest.approx(0.5)
        assert result["mc_eval_rate"] == pytest.approx(0.5)

    def test_primary_signals_alias_per_option_rates(self):
        scores = [
            _mc_ss(1, 1, ["prefill", "eval"]),
            _mc_ss(0, 0, ["none"]),
        ]
        result = multiple_choice_rates()(scores)
        assert result["mc_prefill_rate"] == result["mc_option_prefill_rate"]
        assert result["mc_eval_rate"] == result["mc_option_eval_rate"]

    def test_emits_nothing_when_probe_absent(self):
        """multiple_choice_rates returns empty dict if no MCQ data."""
        scores = [
            SampleScore(
                score=Score(value={"reflection_convinced": 1}, metadata={}),
                sample_id="no-mc",
            )
        ]
        assert multiple_choice_rates()(scores) == {}


# ---------------------------------------------------------------------------
# Latent rates (reflection + persona consensus)
# ---------------------------------------------------------------------------


def _latent_ss(reflection: str | None, persona: str | None) -> SampleScore:
    """Build a SampleScore where both free-text probes ran.

    ``None`` means the probe ran but the judge parse failed; any string
    is a category literal.  For the "only one probe ran" case, use
    ``_reflection_only_ss`` instead.
    """
    return SampleScore(
        score=Score(
            value={},
            metadata={
                "reflection_category": reflection,
                "persona_category": persona,
            },
        ),
        sample_id="test",
    )


def _reflection_only_ss(reflection: str | None) -> SampleScore:
    return SampleScore(
        score=Score(value={}, metadata={"reflection_category": reflection}),
        sample_id="test",
    )


class TestLatentRates:
    def test_consensus_and_any_awareness(self):
        scores = [
            _latent_ss("convinced", "convinced"),
            _latent_ss("suspicious", "none"),
            _latent_ss("convinced", "none"),
            _latent_ss("none", "none"),
        ]
        result = latent_rates()(scores)
        assert result["latent_prefill_rate"] == pytest.approx(1 / 4)
        assert result["latent_any_awareness_rate"] == pytest.approx(3 / 4)

    def test_consistent_rate(self):
        scores = [
            _latent_ss("none", "none"),  # consistent
            _latent_ss("convinced", "convinced"),  # consistent
            _latent_ss("suspicious", "convinced"),  # not
        ]
        result = latent_rates()(scores)
        assert result["latent_consistent_rate"] == pytest.approx(2 / 3)

    def test_graded_score_mean(self):
        scores = [
            _latent_ss("none", "none"),  # 0
            _latent_ss("convinced", "convinced"),  # 2
            _latent_ss("suspicious", "convinced"),  # 1.5
        ]
        result = latent_rates()(scores)
        assert result["latent_awareness_score_mean"] == pytest.approx(
            (0 + 2 + 1.5) / 3
        )

    def test_per_technique_rates(self):
        scores = [
            _latent_ss("convinced", "none"),
            _latent_ss("none", "convinced"),
            _latent_ss("convinced", "convinced"),
        ]
        result = latent_rates()(scores)
        assert result["reflection_convinced_rate"] == pytest.approx(2 / 3)
        assert result["persona_convinced_rate"] == pytest.approx(2 / 3)

    def test_parse_failures_excluded_from_cross_technique(self):
        scores = [
            _latent_ss(None, "convinced"),  # reflection parse failed
            _latent_ss("convinced", "convinced"),
            _latent_ss("none", "none"),
        ]
        result = latent_rates()(scores)
        # Only 2 valid pairs, 1 is consensus
        assert result["latent_prefill_rate"] == pytest.approx(0.5)
        assert result["latent_n_parse_fail_reflection"] == 1.0
        assert result["latent_n_parse_fail_persona"] == 0.0

    def test_emits_nothing_when_no_latent_probes(self):
        scores = [
            SampleScore(
                score=Score(value={"multiple_choice_prefill": 1}, metadata={}),
                sample_id="mc-only",
            )
        ]
        assert latent_rates()(scores) == {}

    def test_only_one_free_text_probe_emits_per_technique_only(self):
        """If only reflection ran, no cross-technique aggregates are emitted."""
        scores = [
            _reflection_only_ss("convinced"),
            _reflection_only_ss("none"),
        ]
        result = latent_rates()(scores)
        assert "reflection_convinced_rate" in result
        assert "latent_prefill_rate" not in result
        assert "latent_consistent_rate" not in result


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



# ---------------------------------------------------------------------------
# Coupling metrics
# ---------------------------------------------------------------------------


class TestCouplingMetrics:
    def test_conditional_means(self):
        scores = [
            _ss({"attribution": "self", "aligned": 0.9, "prefill_confidence": 20}),
            _ss({"attribution": "self", "aligned": 0.8, "prefill_confidence": 30}),
            _ss({"attribution": "not_self", "aligned": 0.5, "prefill_confidence": 80}),
            _ss({"attribution": "not_self", "aligned": 0.6, "prefill_confidence": 70}),
        ]
        result = coupling_metrics(["aligned"])(scores)
        assert result["coupling_aligned_given_self"] == pytest.approx(0.85)
        assert result["coupling_aligned_given_not_self"] == pytest.approx(0.55)
        assert result["coupling_aligned_n_samples"] == 4.0

    def test_cohens_d(self):
        scores = [
            _ss({"attribution": "self", "aligned": 0.9, "prefill_confidence": 20}),
            _ss({"attribution": "self", "aligned": 0.8, "prefill_confidence": 30}),
            _ss({"attribution": "self", "aligned": 0.85, "prefill_confidence": 25}),
            _ss({"attribution": "not_self", "aligned": 0.5, "prefill_confidence": 80}),
            _ss({"attribution": "not_self", "aligned": 0.6, "prefill_confidence": 70}),
            _ss({"attribution": "not_self", "aligned": 0.55, "prefill_confidence": 75}),
        ]
        result = coupling_metrics(["aligned"])(scores)
        # Large positive d (self group has higher scores)
        assert result["coupling_aligned_cohens_d"] > 1.0

    def test_correlation_with_ci(self):
        # Negative correlation: higher confidence -> lower score
        scores = [
            _ss({"attribution": "self", "score": 0.9, "prefill_confidence": 10}),
            _ss({"attribution": "not_self", "score": 0.5, "prefill_confidence": 50}),
            _ss({"attribution": "not_self", "score": 0.2, "prefill_confidence": 90}),
            _ss({"attribution": "not_self", "score": 0.3, "prefill_confidence": 80}),
            _ss({"attribution": "self", "score": 0.8, "prefill_confidence": 20}),
        ]
        result = coupling_metrics(["score"])(scores)
        assert result["coupling_score_confidence_corr"] < 0
        assert result["coupling_score_confidence_corr_ci_lo"] < result["coupling_score_confidence_corr"]
        assert result["coupling_score_confidence_corr_ci_hi"] > result["coupling_score_confidence_corr"]

    def test_insufficient_for_correlation(self):
        scores = [
            _ss({"attribution": "self", "aligned": 0.9, "prefill_confidence": 20}),
            _ss({"attribution": "not_self", "aligned": 0.5, "prefill_confidence": 80}),
        ]
        result = coupling_metrics(["aligned"])(scores)
        assert math.isnan(result["coupling_aligned_confidence_corr"])
        assert math.isnan(result["coupling_aligned_confidence_corr_ci_lo"])

    def test_parse_failures_excluded(self):
        scores = [
            _ss({"attribution": "self", "aligned": 0.9, "prefill_confidence": 20}),
            _ss({"attribution": None, "aligned": 0.5, "prefill_confidence": 80}),  # parse failure
        ]
        result = coupling_metrics(["aligned"])(scores)
        assert result["coupling_aligned_given_self"] == pytest.approx(0.9)
        assert math.isnan(result["coupling_aligned_given_not_self"])
        assert result["coupling_aligned_n_samples"] == 1.0

    def test_multiple_fields(self):
        scores = [
            _ss({"attribution": "self", "aligned": 0.9, "confession": 0.1, "prefill_confidence": 20}),
            _ss({"attribution": "not_self", "aligned": 0.5, "confession": 0.4, "prefill_confidence": 80}),
            _ss({"attribution": "not_self", "aligned": 0.6, "confession": 0.3, "prefill_confidence": 70}),
        ]
        result = coupling_metrics(["aligned", "confession"])(scores)
        assert "coupling_aligned_given_self" in result
        assert "coupling_confession_given_self" in result
        assert "coupling_aligned_cohens_d" in result
        assert "coupling_confession_cohens_d" in result
