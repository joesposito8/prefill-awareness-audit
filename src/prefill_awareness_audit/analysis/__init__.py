"""Analysis -- cross-condition comparison and figure generation."""

from .compare import (
    compare_conditions,
    extract_condition_summary,
    format_comparison_table,
    load_experiment_logs,
)

__all__ = [
    "compare_conditions",
    "extract_condition_summary",
    "format_comparison_table",
    "load_experiment_logs",
]
