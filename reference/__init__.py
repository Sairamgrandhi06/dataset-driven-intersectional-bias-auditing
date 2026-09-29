"""
Independent Reference Audit Package.

This package provides a completely standalone, independent reference implementation
for intersectional fairness auditing (Sex x Race) and expected-result oracle verification.
"""

from reference.intersectional_reference import (
    create_reference_intersectional_labels,
    calculate_reference_group_metrics,
    calculate_reference_disparities,
    audit_intersectional_reference
)

__all__ = [
    "create_reference_intersectional_labels",
    "calculate_reference_group_metrics",
    "calculate_reference_disparities",
    "audit_intersectional_reference"
]
