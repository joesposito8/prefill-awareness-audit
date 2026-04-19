"""Cross-condition and cross-model comparison from Inspect eval logs."""

from __future__ import annotations

import math
from pathlib import Path

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

from ..types import ComparisonTable, Condition, ConditionSummary


def load_experiment_logs(log_dir: str | Path) -> list[EvalLog]:
    """Find and load all eval logs in a directory (headers only).

    Args:
        log_dir: Directory containing Inspect eval log files.

    Returns:
        List of EvalLog objects with results but no sample data.
    """
    log_infos = list_eval_logs(str(log_dir))
    return [read_eval_log(info, header_only=True) for info in log_infos]


def extract_condition_summary(log: EvalLog) -> ConditionSummary:
    """Extract a flat metrics dict from an EvalLog.

    Reads the condition from task metadata and all metrics from the
    first scorer's results.

    Args:
        log: An EvalLog (header_only=True is sufficient).

    Returns:
        ConditionSummary with condition, model, flat metrics dict.

    Raises:
        ValueError: If condition metadata is missing or results are empty.
    """
    # Extract condition from task metadata
    metadata = log.eval.metadata or {}
    condition_str = metadata.get("condition")
    if condition_str is None:
        raise ValueError(
            f"Log {log.eval.eval_id} has no 'condition' in metadata. "
            "Was this produced by make_audit_task()?"
        )
    condition = Condition(condition_str)

    # Extract model
    model = log.eval.model

    # Extract metrics from results
    metrics: dict[str, float] = {}
    if log.results and log.results.scores:
        for eval_score in log.results.scores:
            for name, eval_metric in eval_score.metrics.items():
                metrics[name] = eval_metric.value

    sample_count = log.results.completed_samples if log.results else 0

    return ConditionSummary(
        condition=condition,
        model=model,
        metrics=metrics,
        sample_count=sample_count,
    )


def compare_conditions(summaries: list[ConditionSummary]) -> ComparisonTable:
    """Build cross-condition delta table using PROBE_ONLY as reference.

    Args:
        summaries: List of ConditionSummary from different eval runs.

    Returns:
        ComparisonTable with deltas and optional model grouping.
    """
    # Find PROBE_ONLY baseline (per model)
    by_model: dict[str, list[ConditionSummary]] = {}
    for s in summaries:
        by_model.setdefault(s.model, []).append(s)

    # Compute deltas vs PROBE_ONLY
    deltas: dict[str, dict[str, float]] = {}
    for s in summaries:
        if s.condition == Condition.PROBE_ONLY:
            continue
        # Find PROBE_ONLY for this model
        probe_only = next(
            (p for p in by_model.get(s.model, []) if p.condition == Condition.PROBE_ONLY),
            None,
        )
        if probe_only is None:
            continue
        delta: dict[str, float] = {}
        for k, v in s.metrics.items():
            baseline_v = probe_only.metrics.get(k)
            if baseline_v is not None and not math.isnan(v) and not math.isnan(baseline_v):
                delta[k] = v - baseline_v
            else:
                delta[k] = float("nan")
        deltas[s.condition.value] = delta

    return ComparisonTable(
        conditions=summaries,
        deltas_vs_probe_only=deltas if deltas else None,
        model_comparison=by_model if len(by_model) > 1 else None,
    )


# ---------------------------------------------------------------------------
# Key metrics for the summary card display
# ---------------------------------------------------------------------------

_KEY_METRICS = [
    "recognition_rate",
    "detection_rate",
    "false_attribution_rate",
    "g_mean",
    "uncertain_rate",
    "confidence_mean",
    "spontaneous_rate",
    "latent_prefill_rate",
    "latent_any_awareness_rate",
    "latent_consistent_rate",
    "latent_awareness_score_mean",
    "reflection_convinced_rate",
    "persona_convinced_rate",
    "mc_prefill_rate",
    "mc_eval_rate",
]


def format_comparison_table(table: ComparisonTable) -> str:
    """Render comparison as Option 3 style condition cards.

    Each condition gets a self-contained card showing the full awareness
    profile, followed by a delta summary vs PROBE_ONLY.
    """
    lines: list[str] = []

    for summary in table.conditions:
        m = summary.metrics
        lines.append("")
        lines.append(
            f"=== {summary.condition.value} "
            f"(n={summary.sample_count}, model={summary.model}) ==="
        )
        lines.append("")

        # Ground truth confusion matrix (TPR / TNR / FPR / FNR)
        lines.append("  Ground truth")
        _append_rate_line(lines, "    recognition", m, "recognition_rate", "")
        _append_rate_line(lines, "    detection  ", m, "detection_rate", "")
        _append_rate_line(lines, "    false attr ", m, "false_attribution_rate", "")
        _append_rate_line(lines, "    miss       ", m, "miss_rate", "")
        g_mean = m.get("g_mean", float("nan"))
        lines.append(f"    g_mean     : {_fmt(g_mean)}")
        _append_rate_line(lines, "    uncertain  ", m, "uncertain_rate", "")
        n_classified = int(m.get("n_classified", 0))
        n_samples = int(m.get("n_samples", 0))
        n_parse_fail = int(m.get("n_parse_fail", 0))
        lines.append(
            f"    (n_classified={n_classified}, n_samples={n_samples}, "
            f"parse_fail={n_parse_fail})"
        )

        # Confidence
        _append_mean_line(lines, "  Confidence ", m, "confidence_mean", "confidence_median", "confidence_std")
        _append_n(lines, m, "confidence_n_samples")

        # Spontaneous
        _append_rate_line(lines, "  Spontaneous", m, "spontaneous_rate", "rate")
        _append_n(lines, m, "spontaneous_n_samples")

        # Latent awareness — free-text consensus (reflection + persona)
        if "latent_prefill_rate" in m:
            _append_rate_line(
                lines, "  Latent prefill (both convinced)", m, "latent_prefill_rate", ""
            )
            _append_rate_line(
                lines, "  Latent any awareness           ", m, "latent_any_awareness_rate", ""
            )
            _append_rate_line(
                lines, "  Latent consistent              ", m, "latent_consistent_rate", ""
            )
            score_mean = m.get("latent_awareness_score_mean", float("nan"))
            lines.append(f"  Latent score mean              : {_fmt(score_mean)}")
        if "reflection_convinced_rate" in m:
            _append_rate_line(
                lines, "    reflection convinced", m, "reflection_convinced_rate", ""
            )
        if "persona_convinced_rate" in m:
            _append_rate_line(
                lines, "    persona convinced   ", m, "persona_convinced_rate", ""
            )
        if (
            "latent_prefill_rate" in m
            or "reflection_convinced_rate" in m
            or "persona_convinced_rate" in m
        ):
            _append_n(lines, m, "latent_n_samples")

        # Latent awareness — multiple-choice probe (opt-in)
        if "mc_prefill_rate" in m:
            _append_rate_line(lines, "  MC prefill", m, "mc_prefill_rate", "")
            _append_rate_line(lines, "  MC eval   ", m, "mc_eval_rate", "")
            _append_n(lines, m, "mc_n_samples")

        # Diagnostic tags
        lines.append("  Diagnostic  ", )
        for tag in sorted(_get_diagnostic_tags(m)):
            key = f"diagnostic_{tag}_rate"
            if key in m:
                _append_rate_line(lines, f"    {tag}", m, key, "")
        _append_n(lines, m, "diagnostic_n_samples")

        # Coupling (dynamic based on what fields exist)
        coupling_fields = _get_coupling_fields(m)
        for field in coupling_fields:
            lines.append(f"  Coupling: {field}")
            gs = m.get(f"coupling_{field}_given_self", float("nan"))
            gns = m.get(f"coupling_{field}_given_not_self", float("nan"))
            d = m.get(f"coupling_{field}_cohens_d", float("nan"))
            r = m.get(f"coupling_{field}_confidence_corr", float("nan"))
            r_lo = m.get(f"coupling_{field}_confidence_corr_ci_lo", float("nan"))
            r_hi = m.get(f"coupling_{field}_confidence_corr_ci_hi", float("nan"))
            lines.append(f"    given self: {_fmt(gs)} | given not_self: {_fmt(gns)} | d: {_fmt(d)}")
            lines.append(f"    corr: {_fmt(r)} [{_fmt(r_lo)}, {_fmt(r_hi)}]")
            n_key = f"coupling_{field}_n_samples"
            _append_n(lines, m, n_key)

    # Delta summary
    if table.deltas_vs_probe_only:
        lines.append("")
        lines.append("=== Deltas vs PROBE_ONLY ===")
        lines.append("")
        for cond_name, deltas in table.deltas_vs_probe_only.items():
            delta_parts = []
            for k in _KEY_METRICS:
                if k in deltas and not math.isnan(deltas[k]):
                    delta_parts.append(f"{k}: {_fmt_delta(deltas[k])}")
            if delta_parts:
                lines.append(f"  {cond_name}")
                for part in delta_parts:
                    lines.append(f"    {part}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(v: float | None) -> str:
    if v is None or math.isnan(v):
        return "n/a"
    return f"{v:.3f}"


def _fmt_delta(v: float) -> str:
    if math.isnan(v):
        return "n/a"
    return f"{v:+.3f}"


def _append_rate_line(
    lines: list[str], prefix: str, m: dict, key: str, label: str
) -> None:
    rate = m.get(key, float("nan"))
    lo = m.get(f"{key}_ci_lo", float("nan"))
    hi = m.get(f"{key}_ci_hi", float("nan"))
    suffix = f" {label}:" if label else ":"
    lines.append(f"{prefix}{suffix} {_fmt(rate)} [{_fmt(lo)}, {_fmt(hi)}]")


def _append_mean_line(
    lines: list[str], prefix: str, m: dict,
    mean_key: str, median_key: str, std_key: str,
) -> None:
    mean_v = m.get(mean_key, float("nan"))
    lo = m.get(f"{mean_key}_ci_lo", float("nan"))
    hi = m.get(f"{mean_key}_ci_hi", float("nan"))
    med = m.get(median_key, float("nan"))
    std = m.get(std_key, float("nan"))
    lines.append(
        f"{prefix} mean: {_fmt(mean_v)} [{_fmt(lo)}, {_fmt(hi)}] "
        f"| median: {_fmt(med)} | std: {_fmt(std)}"
    )


def _append_n(lines: list[str], m: dict, key: str) -> None:
    n = m.get(key, 0)
    lines.append(f"    (n={int(n)})")


def _get_diagnostic_tags(m: dict) -> list[str]:
    tags = []
    for k in m:
        if k.startswith("diagnostic_") and k.endswith("_rate") and not k.endswith("_ci_lo") and not k.endswith("_ci_hi"):
            tag = k[len("diagnostic_"):-len("_rate")]
            tags.append(tag)
    return tags


def _get_coupling_fields(m: dict) -> list[str]:
    fields = []
    for k in m:
        if k.startswith("coupling_") and k.endswith("_given_self"):
            field = k[len("coupling_"):-len("_given_self")]
            if field not in fields:
                fields.append(field)
    return fields
