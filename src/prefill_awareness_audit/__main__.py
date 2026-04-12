"""CLI entry point: python -m prefill_awareness_audit compare ..."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prefill_awareness_audit",
        description="Prefill Awareness Audit tools",
    )
    sub = parser.add_subparsers(dest="command")

    compare_parser = sub.add_parser(
        "compare", help="Cross-condition comparison from eval logs"
    )
    compare_parser.add_argument(
        "--log-dir", type=Path, required=True, help="Directory containing eval logs"
    )
    compare_parser.add_argument(
        "--figures", action="store_true", help="Generate figures"
    )
    compare_parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory for figures"
    )

    args = parser.parse_args()

    if args.command == "compare":
        from .analysis.compare import (
            compare_conditions,
            extract_condition_summary,
            format_comparison_table,
            load_experiment_logs,
        )

        logs = load_experiment_logs(args.log_dir)
        summaries = [extract_condition_summary(log) for log in logs]
        table = compare_conditions(summaries)
        print(format_comparison_table(table))

        if args.figures:
            from .analysis.figures import (
                plot_awareness_by_condition,
                plot_confidence_distribution,
                plot_delta_heatmap,
            )

            out = args.output_dir or args.log_dir / "figures"
            out.mkdir(parents=True, exist_ok=True)
            plot_awareness_by_condition(table, out / "awareness_by_condition.png")
            plot_confidence_distribution(table, out / "confidence_distribution.png")
            plot_delta_heatmap(table, out / "delta_heatmap.png")
            print(f"\nFigures saved to {out}/")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
