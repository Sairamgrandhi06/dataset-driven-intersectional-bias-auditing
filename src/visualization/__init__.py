"""Visualization modules for fairness plots and charts."""

from src.visualization.plots import (
    plot_intersectional_metric_bars,
    plot_all_intersectional_figures,
    plot_tradeoff_pareto_front,
    plot_before_after_comparison,
    plot_baseline_vs_mitigated_performance,
    plot_baseline_vs_mitigated_fairness,
    plot_baseline_vs_mitigated_group_metric,
    plot_calibration_curves,
    plot_probability_distributions,
    plot_tradeoff_summary_chart
)

__all__ = [
    "plot_intersectional_metric_bars",
    "plot_all_intersectional_figures",
    "plot_tradeoff_pareto_front",
    "plot_before_after_comparison",
    "plot_baseline_vs_mitigated_performance",
    "plot_baseline_vs_mitigated_fairness",
    "plot_baseline_vs_mitigated_group_metric",
    "plot_calibration_curves",
    "plot_probability_distributions",
    "plot_tradeoff_summary_chart"
]
