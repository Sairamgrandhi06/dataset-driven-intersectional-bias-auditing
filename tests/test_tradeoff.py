"""
Unit tests for Calibration Evaluation and Three-Way Trade-off Reporting Module.
"""

import pytest
import numpy as np
import pandas as pd

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


@pytest.fixture
def synthetic_calibration_data():
    """Fixture providing synthetic labels and probabilities for calibration testing."""
    np.random.seed(42)
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.85, 0.15, 0.65, 0.05])
    return y_true, y_prob


def test_brier_score_and_ece_calculation(synthetic_calibration_data):
    """Verify Brier Score and ECE computation on synthetic probabilities."""
    y_true, y_prob = synthetic_calibration_data

    calib = evaluate_calibration(y_true, y_prob, n_bins=5)

    assert "brier_score" in calib
    assert "expected_calibration_error" in calib
    assert "reliability_curve" in calib

    # Brier score range [0.0, 1.0]
    assert 0.0 <= calib["brier_score"] <= 1.0
    assert 0.0 <= calib["expected_calibration_error"] <= 1.0


def test_calibration_comparison():
    """Verify compare_calibration yields accurate deltas and status."""
    base_calib = {"brier_score": 0.12, "expected_calibration_error": 0.08}
    mit_calib = {"brier_score": 0.15, "expected_calibration_error": 0.10}

    comp = compare_calibration(base_calib, mit_calib)

    # Brier score increased by +0.03 -> Worsened
    assert comp["brier_score"]["absolute_change"] == pytest.approx(0.03)
    assert comp["brier_score"]["impact_status"] == "Worsened"

    # ECE increased by +0.02 -> Worsened
    assert comp["expected_calibration_error"]["absolute_change"] == pytest.approx(0.02)
    assert comp["expected_calibration_error"]["impact_status"] == "Worsened"


def test_compute_metric_change():
    """Verify metric delta and percentage change calculation."""
    res = compute_metric_change(0.80, 0.84, higher_is_better=True)

    assert res["baseline"] == pytest.approx(0.80)
    assert res["mitigated"] == pytest.approx(0.84)
    assert res["absolute_change"] == pytest.approx(0.04)
    assert res["percentage_change"] == pytest.approx(5.0)  # +5%
    assert res["impact_status"] == "Improved"


def test_comprehensive_tradeoff_and_summary_df():
    """Verify full calculate_comprehensive_tradeoff and CSV DataFrame generation."""
    base_perf = {"accuracy": 0.84, "precision": 0.75, "recall": 0.60, "f1_score": 0.667, "roc_auc": 0.88}
    mit_perf = {"accuracy": 0.76, "precision": 0.65, "recall": 0.55, "f1_score": 0.595, "roc_auc": 0.78}

    base_calib = evaluate_calibration(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]))
    mit_calib = evaluate_calibration(np.array([1, 0, 1, 0]), np.array([0.6, 0.4, 0.7, 0.3]))

    base_audit = {
        "disparities": {
            "demographic_parity_difference": 0.30,
            "disparate_impact_ratio": 0.20,
            "equalized_odds_difference": 0.45,
            "equal_opportunity_difference": 0.45,
            "false_positive_rate_difference": 0.12
        }
    }
    mit_audit = {
        "disparities": {
            "demographic_parity_difference": 0.05,
            "disparate_impact_ratio": 0.80,
            "equalized_odds_difference": 0.10,
            "equal_opportunity_difference": 0.10,
            "false_positive_rate_difference": 0.03
        }
    }

    tradeoff = calculate_comprehensive_tradeoff(
        base_perf, mit_perf, base_audit, mit_audit, base_calib, mit_calib
    )

    assert "performance" in tradeoff
    assert "calibration" in tradeoff
    assert "fairness" in tradeoff
    assert "interpretation" in tradeoff

    df = generate_tradeoff_summary_df(tradeoff)

    assert len(df) >= 10
    assert "category" in df.columns
    assert "metric_name" in df.columns
    assert "baseline_value" in df.columns
    assert "mitigated_value" in df.columns
    assert "absolute_change" in df.columns
    assert "percentage_change" in df.columns
    assert "impact_status" in df.columns


def test_missing_metric_handling():
    """Verify graceful handling when ROC-AUC or calibration metrics are missing/None."""
    base_perf = {"accuracy": 0.80, "precision": 0.70, "recall": 0.50, "f1_score": 0.58, "roc_auc": None}
    mit_perf = {"accuracy": 0.78, "precision": 0.68, "recall": 0.48, "f1_score": 0.56, "roc_auc": None}

    base_calib = {"brier_score": 0.15, "expected_calibration_error": 0.05}
    mit_calib = {"brier_score": 0.16, "expected_calibration_error": 0.06}

    base_audit = {
        "disparities": {
            "demographic_parity_difference": 0.20,
            "disparate_impact_ratio": 0.50,
            "equalized_odds_difference": 0.30,
            "equal_opportunity_difference": 0.30,
            "false_positive_rate_difference": 0.10
        }
    }
    mit_audit = {
        "disparities": {
            "demographic_parity_difference": 0.10,
            "disparate_impact_ratio": 0.70,
            "equalized_odds_difference": 0.15,
            "equal_opportunity_difference": 0.15,
            "false_positive_rate_difference": 0.05
        }
    }

    tradeoff = calculate_comprehensive_tradeoff(base_perf, mit_perf, base_audit, mit_audit, base_calib, mit_calib)

    assert tradeoff["performance"]["ROC-AUC"]["baseline"] == 0.0
    assert tradeoff["performance"]["ROC-AUC"]["mitigated"] == 0.0
    assert tradeoff["performance"]["ROC-AUC"]["absolute_change"] == 0.0
