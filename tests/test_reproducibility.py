"""
Unit tests for End-to-End Reproducibility and Authoritative Result Consistency.
"""

import pytest
import numpy as np
from dashboard.utils import load_json_file, load_csv_file


def test_final_results_json_exists_and_valid():
    """Verify final_results.json exists and contains complete authoritative metrics."""
    data = load_json_file("final_results.json")
    assert data is not None
    assert "performance_metrics" in data
    assert "fairness_metrics" in data
    assert "calibration_metrics" in data


def test_authoritative_performance_metrics():
    """Verify baseline and mitigated performance metrics match authoritative targets."""
    data = load_json_file("final_results.json")
    assert data is not None
    perf_b = data["performance_metrics"]["baseline"]
    perf_m = data["performance_metrics"]["mitigated"]

    # Baseline performance targets
    assert perf_b["accuracy"] == pytest.approx(0.8399, abs=1e-3)
    assert perf_b["precision"] == pytest.approx(0.7267, abs=1e-3)
    assert perf_b["recall"] == pytest.approx(0.5719, abs=1e-3)
    assert perf_b["f1_score"] == pytest.approx(0.6401, abs=1e-3)
    assert perf_b["roc_auc"] == pytest.approx(0.8841, abs=1e-3)

    # Mitigated performance targets
    assert perf_m["accuracy"] == pytest.approx(0.7578, abs=1e-3)
    assert perf_m["precision"] == pytest.approx(0.5936, abs=1e-3)
    assert perf_m["recall"] == pytest.approx(0.0866, abs=1e-3)
    assert perf_m["f1_score"] == pytest.approx(0.1511, abs=1e-3)


def test_authoritative_fairness_and_calibration_metrics():
    """Verify baseline and mitigated fairness and calibration metrics match targets."""
    data = load_json_file("final_results.json")
    assert data is not None
    fair_b = data["fairness_metrics"]["baseline"]
    fair_m = data["fairness_metrics"]["mitigated"]
    cal_b = data["calibration_metrics"]["baseline"]
    cal_m = data["calibration_metrics"]["mitigated"]

    # Equalized Odds Difference: 0.4825 -> 0.1667
    assert fair_b["equalized_odds_difference"] == pytest.approx(0.4825, abs=1e-3)
    assert fair_m["equalized_odds_difference"] == pytest.approx(0.1667, abs=1e-3)

    # Calibration: Brier 0.1128 -> 0.2156 | ECE 0.0151 -> 0.2166
    assert cal_b["brier_score"] == pytest.approx(0.1128, abs=1e-3)
    assert cal_m["brier_score"] == pytest.approx(0.2156, abs=1e-3)
    assert cal_b["expected_calibration_error"] == pytest.approx(0.0151, abs=1e-3)
    assert cal_m["expected_calibration_error"] == pytest.approx(0.2166, abs=1e-3)


def test_final_results_csv_consistency():
    """Verify final_results.csv matches final_results.json exactly."""
    df = load_csv_file("final_results.csv")
    assert df is not None
    assert len(df) >= 10
    acc_row = df[df["metric_name"] == "Accuracy"].iloc[0]
    assert acc_row["baseline_value"] == pytest.approx(0.8399, abs=1e-3)
    assert acc_row["mitigated_value"] == pytest.approx(0.7578, abs=1e-3)
