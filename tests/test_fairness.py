"""
Unit tests for Single-Attribute Fairness Auditing Module.
"""

import pytest
import numpy as np
import pandas as pd

from src.fairness.single_attribute import (
    calculate_group_metrics,
    audit_single_attribute,
    run_single_attribute_audits
)


def test_calculate_group_metrics():
    """Verify group-level metric calculations on a known synthetic slice."""
    # y_true: [1, 1, 0, 0, 1]
    # y_pred: [1, 0, 1, 0, 1]
    # TP=2, FN=1, FP=1, TN=1
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])
    mask = np.array([True, True, True, True, True])

    metrics = calculate_group_metrics(y_true, y_pred, mask)

    assert metrics["sample_count"] == 5
    assert metrics["selection_rate"] == pytest.approx(3 / 5)  # 3 positive predictions
    assert metrics["true_positive_rate"] == pytest.approx(2 / 3)  # TP=2, FN=1
    assert metrics["false_positive_rate"] == pytest.approx(1 / 2)  # FP=1, TN=1
    assert metrics["false_negative_rate"] == pytest.approx(1 / 3)  # FN=1, TP=2
    assert metrics["true_negative_rate"] == pytest.approx(1 / 2)   # TN=1, FP=1
    assert metrics["confusion_matrix"]["TP"] == 2
    assert metrics["confusion_matrix"]["FP"] == 1
    assert metrics["confusion_matrix"]["TN"] == 1
    assert metrics["confusion_matrix"]["FN"] == 1


def test_calculate_group_metrics_empty_mask():
    """Verify zero/empty mask yields clean fallback metrics without raising zero division error."""
    y_true = np.array([1, 0])
    y_pred = np.array([1, 0])
    mask = np.array([False, False])

    metrics = calculate_group_metrics(y_true, y_pred, mask)

    assert metrics["sample_count"] == 0
    assert metrics["selection_rate"] == 0.0
    assert metrics["true_positive_rate"] == 0.0
    assert metrics["false_positive_rate"] == 0.0


def test_audit_single_attribute():
    """Verify single attribute audit calculates correct group metrics and disparities."""
    # Create synthetic dataset with 2 groups: 'GroupA' and 'GroupB'
    # GroupA (4 samples): y_true=[1, 1, 0, 0], y_pred=[1, 1, 0, 0] -> SR=0.5, TPR=1.0, FPR=0.0
    # GroupB (4 samples): y_true=[1, 1, 0, 0], y_pred=[1, 0, 1, 0] -> SR=0.5, TPR=0.5, FPR=0.5
    y_true = pd.Series([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = pd.Series([1, 1, 0, 0, 1, 0, 1, 0])
    protected = pd.Series(["GroupA", "GroupA", "GroupA", "GroupA",
                           "GroupB", "GroupB", "GroupB", "GroupB"])

    audit = audit_single_attribute(y_true, y_pred, protected, attribute_name="test_group")

    assert audit["attribute_name"] == "test_group"
    assert "GroupA" in audit["groups"]
    assert "GroupB" in audit["groups"]

    grp_a = audit["groups"]["GroupA"]
    grp_b = audit["groups"]["GroupB"]

    assert grp_a["selection_rate"] == pytest.approx(0.5)
    assert grp_b["selection_rate"] == pytest.approx(0.5)

    assert grp_a["true_positive_rate"] == pytest.approx(1.0)
    assert grp_b["true_positive_rate"] == pytest.approx(0.5)

    disparities = audit["disparities"]
    # Demographic Parity Difference = |0.5 - 0.5| = 0.0
    assert disparities["demographic_parity_difference"] == pytest.approx(0.0)
    # Disparate Impact Ratio = 0.5 / 0.5 = 1.0
    assert disparities["disparate_impact_ratio"] == pytest.approx(1.0)
    # Equal Opportunity Difference = |1.0 - 0.5| = 0.5
    assert disparities["equal_opportunity_difference"] == pytest.approx(0.5)
    # FPR Difference = |0.0 - 0.5| = 0.5
    assert disparities["false_positive_rate_difference"] == pytest.approx(0.5)
    # Equalized Odds Difference = max(0.5, 0.5) = 0.5
    assert disparities["equalized_odds_difference"] == pytest.approx(0.5)

    # Check that metric definitions are present
    assert "demographic_parity_difference" in audit["metric_definitions"]


def test_run_single_attribute_audits():
    """Verify run_single_attribute_audits iterates across all columns in DataFrame A."""
    y_true = pd.Series([1, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0, 0])
    A_df = pd.DataFrame({
        "sex": ["Male", "Male", "Female", "Female"],
        "race": ["White", "Black", "White", "Black"]
    })

    audits = run_single_attribute_audits(y_true, y_pred, A_df)

    assert "sex" in audits
    assert "race" in audits
    assert audits["sex"]["attribute_name"] == "sex"
    assert audits["race"]["attribute_name"] == "race"
