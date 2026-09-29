"""
Unit tests for Intersectional Fairness Auditing Module.
"""

import pytest
import numpy as np
import pandas as pd

from src.fairness.intersectional import (
    create_intersectional_attribute,
    audit_intersectional_attributes
)


@pytest.fixture
def synthetic_intersectional_data():
    """Fixture providing synthetic labels, predictions, and protected attributes."""
    np.random.seed(42)
    n = 200

    y_true = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    y_pred = np.random.choice([0, 1], size=n, p=[0.75, 0.25])

    # 4 Groups: Female+White (80), Male+White (80), Female+Black (30), Male+Black (10)
    sexes = ["Female"] * 80 + ["Male"] * 80 + ["Female"] * 30 + ["Male"] * 10
    races = ["White"] * 80 + ["White"] * 80 + ["Black"] * 30 + ["Black"] * 10

    A_df = pd.DataFrame({"sex": sexes, "race": races})

    return y_true, y_pred, A_df


def test_create_intersectional_attribute(synthetic_intersectional_data):
    """Verify combination of sex and race columns into single string Series."""
    _, _, A_df = synthetic_intersectional_data

    intersectional = create_intersectional_attribute(A_df, attributes=["sex", "race"], join_str=" + ")

    assert len(intersectional) == 200
    assert intersectional.iloc[0] == "Female + White"
    assert intersectional.iloc[80] == "Male + White"
    assert intersectional.iloc[160] == "Female + Black"
    assert intersectional.iloc[190] == "Male + Black"

    # Test error handling when column is missing
    with pytest.raises(ValueError, match="Missing attributes"):
        create_intersectional_attribute(A_df, attributes=["sex", "non_existent"])


def test_intersectional_group_counts(synthetic_intersectional_data):
    """Verify that group sample counts sum up to the total dataset length."""
    y_true, y_pred, A_df = synthetic_intersectional_data

    audit = audit_intersectional_attributes(y_true, y_pred, A_df, attributes=["sex", "race"], min_group_size=30)

    total_samples = sum(m["sample_count"] for m in audit["all_group_metrics"].values())
    assert total_samples == 200
    assert audit["total_groups_discovered"] == 4


def test_min_group_size_thresholding(synthetic_intersectional_data):
    """Verify proper partitioning into primary groups vs low-sample groups."""
    y_true, y_pred, A_df = synthetic_intersectional_data

    # Group counts: Female+White (80), Male+White (80), Female+Black (30), Male+Black (10)
    # With min_group_size=30: 3 primary groups, 1 low-sample group ('Male + Black')
    audit = audit_intersectional_attributes(y_true, y_pred, A_df, attributes=["sex", "race"], min_group_size=30)

    assert audit["min_group_size_threshold"] == 30
    assert audit["primary_group_count"] == 3
    assert audit["low_sample_group_count"] == 1
    assert "Male + Black" in audit["low_sample_groups"]
    assert "Male + Black" not in audit["primary_groups"]


def test_audit_intersectional_metrics_and_disparities(synthetic_intersectional_data):
    """Verify metric and disparity calculations on synthetic intersectional dataset."""
    y_true, y_pred, A_df = synthetic_intersectional_data

    audit = audit_intersectional_attributes(y_true, y_pred, A_df, attributes=["sex", "race"], min_group_size=30)

    assert "disparities" in audit
    assert "rankings" in audit

    disparities = audit["disparities"]
    assert "demographic_parity_difference" in disparities
    assert "disparate_impact_ratio" in disparities
    assert "equalized_odds_difference" in disparities
    assert "equal_opportunity_difference" in disparities
    assert "false_positive_rate_difference" in disparities

    rankings = audit["rankings"]
    assert rankings["most_disadvantaged_group"]["group"] is not None
    assert rankings["highest_performing_group"]["group"] is not None


def test_intersectional_audit_reproducibility(synthetic_intersectional_data):
    """Verify deterministic audit execution."""
    y_true, y_pred, A_df = synthetic_intersectional_data

    audit1 = audit_intersectional_attributes(y_true, y_pred, A_df, min_group_size=30)
    audit2 = audit_intersectional_attributes(y_true, y_pred, A_df, min_group_size=30)

    assert audit1["disparities"] == audit2["disparities"]
    assert audit1["rankings"] == audit2["rankings"]
