"""Fairness auditing, intersectional group analysis, calibration, and trade-off reporting modules."""

from src.fairness.single_attribute import (
    calculate_group_metrics,
    audit_single_attribute,
    run_single_attribute_audits
)
from src.fairness.intersectional import (
    create_intersectional_attribute,
    audit_intersectional_attributes
)
from src.fairness.calibration import (
    calculate_expected_calibration_error,
    evaluate_calibration,
    compare_calibration
)
from src.fairness.trade_off import (
    compute_metric_change,
    calculate_comprehensive_tradeoff,
    generate_tradeoff_summary_df,
    generate_tradeoff_interpretation
)

__all__ = [
    "calculate_group_metrics",
    "audit_single_attribute",
    "run_single_attribute_audits",
    "create_intersectional_attribute",
    "audit_intersectional_attributes",
    "calculate_expected_calibration_error",
    "evaluate_calibration",
    "compare_calibration",
    "compute_metric_change",
    "calculate_comprehensive_tradeoff",
    "generate_tradeoff_summary_df",
    "generate_tradeoff_interpretation"
]
