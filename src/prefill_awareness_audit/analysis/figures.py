"""Figure generation for cross-condition comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..types import ComparisonTable


def plot_awareness_by_condition(
    table: ComparisonTable, output_path: str | Path
) -> None:
    """Grouped bar chart: recognition/detection/false-attribution rates per condition.

    Recognition (TPR) is computed over authentic samples only; detection (TNR)
    and false attribution (FPR) over non-authentic samples only. Includes
    Wilson 95% CI error bars when available.
    """
    conditions = table.conditions
    labels = [s.condition.value for s in conditions]
    n = len(conditions)

    recognition = [s.metrics.get("recognition_rate", 0) for s in conditions]
    detection = [s.metrics.get("detection_rate", 0) for s in conditions]
    false_attr = [s.metrics.get("false_attribution_rate", 0) for s in conditions]

    # Error bars from Wilson CI
    def _errs(key: str, rates: list[float]) -> list[list[float]]:
        lo_errs = []
        hi_errs = []
        for i, s in enumerate(conditions):
            lo = s.metrics.get(f"{key}_ci_lo", rates[i])
            hi = s.metrics.get(f"{key}_ci_hi", rates[i])
            lo_errs.append(rates[i] - lo)
            hi_errs.append(hi - rates[i])
        return [lo_errs, hi_errs]

    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), 5))
    ax.bar(x - width, recognition, width, label="recognition (TPR)",
           yerr=_errs("recognition_rate", recognition), capsize=3)
    ax.bar(x, detection, width, label="detection (TNR)",
           yerr=_errs("detection_rate", detection), capsize=3)
    ax.bar(x + width, false_attr, width, label="false attribution (FPR)",
           yerr=_errs("false_attribution_rate", false_attr), capsize=3)

    ax.set_ylabel("Rate")
    ax.set_title("Awareness Ground-Truth Rates by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_confidence_distribution(
    table: ComparisonTable, output_path: str | Path
) -> None:
    """Bar chart of mean confidence per condition with CI error bars.

    For a full box plot, sample-level data would be needed via
    read_eval_log(header_only=False).
    """
    conditions = table.conditions
    labels = [s.condition.value for s in conditions]
    means = [s.metrics.get("confidence_mean", 0) for s in conditions]

    lo_errs = []
    hi_errs = []
    for i, s in enumerate(conditions):
        lo = s.metrics.get("confidence_mean_ci_lo", means[i])
        hi = s.metrics.get("confidence_mean_ci_hi", means[i])
        lo_errs.append(means[i] - lo)
        hi_errs.append(hi - means[i])

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.5), 5))
    x = np.arange(len(conditions))
    ax.bar(x, means, yerr=[lo_errs, hi_errs], capsize=5)
    ax.set_ylabel("Tamper Confidence (0-100)")
    ax.set_title("Mean Tamper Confidence by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_delta_heatmap(
    table: ComparisonTable, output_path: str | Path
) -> None:
    """Heatmap of key metric deltas vs PROBE_ONLY across conditions."""
    if not table.deltas_vs_probe_only:
        return

    key_metrics = [
        "recognition_rate",
        "detection_rate",
        "false_attribution_rate",
        "g_mean",
        "confidence_mean",
        "spontaneous_rate",
        "latent_prefill_rate",
        "latent_eval_rate",
    ]

    cond_names = list(table.deltas_vs_probe_only.keys())
    if not cond_names:
        return

    data = []
    for cond in cond_names:
        row = []
        for metric in key_metrics:
            v = table.deltas_vs_probe_only[cond].get(metric, float("nan"))
            row.append(v if not np.isnan(v) else 0.0)
        data.append(row)

    arr = np.array(data)
    fig, ax = plt.subplots(figsize=(max(6, len(key_metrics) * 1.5), max(4, len(cond_names) * 0.8)))
    im = ax.imshow(arr, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(key_metrics)))
    ax.set_xticklabels([m.replace("_", "\n") for m in key_metrics],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(cond_names)))
    ax.set_yticklabels(cond_names)

    # Annotate cells
    for i in range(len(cond_names)):
        for j in range(len(key_metrics)):
            ax.text(j, i, f"{arr[i, j]:+.2f}", ha="center", va="center", fontsize=9)

    ax.set_title("Metric Deltas vs PROBE_ONLY")
    fig.colorbar(im, ax=ax, label="Delta")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
