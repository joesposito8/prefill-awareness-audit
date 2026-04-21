"""Deep audit of the 20-sample validation logs.

Reads each .eval log, pulls per-sample probe metadata, and prints a
structured audit summary per run: probe fire rates, judge category
distributions, metric families emitted, ground-truth classification,
parse failures, and coupling emission for misalignment.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_samples

RUNS = {
    "baseline-default": "logs/validation-audit-2026-04-20/baseline-default",
    "baseline-mc": "logs/validation-audit-2026-04-20/baseline-mc",
    "bfcl-default": "logs/validation-audit-2026-04-20/bfcl-default",
    "bfcl-mc": "logs/validation-audit-2026-04-20/bfcl-mc",
    "misalignment-default": "logs/validation-audit-2026-04-20/misalignment-default",
}


def find_log(dir_path: str) -> Path:
    p = Path(dir_path)
    evals = sorted(p.glob("*.eval"))
    if not evals:
        raise FileNotFoundError(f"no .eval in {p}")
    return evals[-1]


def metric_families(metrics: dict) -> dict[str, list[str]]:
    fams: dict[str, list[str]] = {}
    for k in metrics:
        head = k.split("_", 1)[0] if "_" in k else k
        # finer bucketing for well-known prefixes
        if k.startswith("recognition_") or k.startswith("detection_") \
                or k.startswith("false_attribution_") or k.startswith("miss_") \
                or k.startswith("g_mean") or k.startswith("uncertain_") \
                or k in ("n_classified", "n_parse_fail"):
            head = "ground_truth"
        elif k.startswith("confidence_"):
            head = "confidence"
        elif k.startswith("spontaneous_"):
            head = "spontaneous"
        elif k.startswith("latent_") or k.startswith("reflection_") \
                or k.startswith("persona_"):
            head = "latent"
        elif k.startswith("mc_"):
            head = "mc"
        elif k.startswith("diagnostic_"):
            head = "diagnostic"
        elif k.startswith("coupling_"):
            head = "coupling"
        else:
            head = "other"
        fams.setdefault(head, []).append(k)
    return fams


def audit_run(name: str, log_dir: str) -> dict:
    path = find_log(log_dir)
    log = read_eval_log(str(path), header_only=True)
    samples = list(read_eval_log_samples(str(path)))

    # Eval-level metrics from the scorer
    metrics: dict[str, float] = {}
    for scorer_spec in log.results.scores if log.results else []:
        for mname, mv in (scorer_spec.metrics or {}).items():
            metrics[mname] = mv.value if hasattr(mv, "value") else mv

    # Per-sample analysis
    probe_present = Counter()  # probe-name → samples with signal
    judge_cats = {
        "reflection": Counter(),
        "persona": Counter(),
    }
    ground_truth_rows = []  # (source_model, probe_model, authentic, attribution)
    errors = 0
    # latent_n_parse_fail_* pulled from metrics block; reflection/persona surface
    # judge-parse failures that would otherwise be invisible in probe_present.
    parse_fails = {
        "diagnostic": 0,
        "reflection": int(metrics.get("latent_n_parse_fail_reflection", 0)),
        "persona": int(metrics.get("latent_n_parse_fail_persona", 0)),
    }
    spot = []

    probe_model = str(log.eval.model) if log.eval else "?"

    for s in samples:
        if s.error is not None:
            errors += 1
            continue
        score = (s.scores or {}).get("_audit_scorer") or next(
            iter((s.scores or {}).values()), None
        )
        if score is None:
            continue
        v = score.value if isinstance(score.value, dict) else {}
        md = score.metadata or {}
        src = (s.metadata or {}).get("source_model", "unknown")
        ground_truth_rows.append(
            {
                "sample_id": s.id,
                "source_model": src,
                "probe_model": probe_model,
                "prefill_authentic": v.get("prefill_authentic"),
                "attribution": (
                    md.get("attribution").value
                    if hasattr(md.get("attribution"), "value")
                    else md.get("attribution")
                ),
                "spontaneous": v.get("spontaneous_detected"),
                "prefill_confidence": v.get("prefill_confidence"),
            }
        )
        if md.get("attribution") is not None:
            probe_present["awareness"] += 1
        if md.get("reflection_category") is not None:
            probe_present["reflection"] += 1
            judge_cats["reflection"][md.get("reflection_category")] += 1
        if md.get("persona_category") is not None:
            probe_present["persona"] += 1
            judge_cats["persona"][md.get("persona_category")] += 1
        if v.get("multiple_choice_prefill") is not None:
            probe_present["multiple_choice"] += 1
        if md.get("diagnostic_tags") is not None or md.get("diagnostic_parse_failed") \
                is not None:
            probe_present["diagnostic"] += 1
            if md.get("diagnostic_parse_failed"):
                parse_fails["diagnostic"] += 1

        if len(spot) < 3:
            raw_attr = md.get("attribution_raw") or ""
            spot.append(
                {
                    "id": s.id,
                    "source": src,
                    "attribution": (
                        md.get("attribution").value
                        if hasattr(md.get("attribution"), "value")
                        else md.get("attribution")
                    ),
                    "confidence": v.get("prefill_confidence"),
                    "attribution_raw_snippet": (raw_attr or "")[:400],
                    "reflection_category": md.get("reflection_category"),
                    "reflection_snippet": (md.get("reflection_raw_response", "") or "")[
                        :400
                    ],
                    "persona_category": md.get("persona_category"),
                    "persona_snippet": (md.get("persona_raw_response", "") or "")[:400],
                    "diagnostic_tags": md.get("diagnostic_tags"),
                    "mc_choices": md.get("multiple_choice_choices"),
                }
            )

    return {
        "run": name,
        "log_path": str(path),
        "probe_model": probe_model,
        "n_samples": len(samples),
        "n_errors": errors,
        "metric_families": {k: len(v) for k, v in metric_families(metrics).items()},
        "metrics_flat": metrics,
        "probe_present": dict(probe_present),
        "judge_category_dist": {k: dict(v) for k, v in judge_cats.items()},
        "parse_fails": parse_fails,
        "ground_truth_rows": ground_truth_rows,
        "spot_checks": spot,
    }


def main() -> None:
    out = {}
    for name, d in RUNS.items():
        try:
            out[name] = audit_run(name, d)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
