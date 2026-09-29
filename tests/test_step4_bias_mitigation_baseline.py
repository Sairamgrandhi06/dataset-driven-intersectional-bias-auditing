"""
Regression tests for Step 4 — Bias Mitigation Control Center baseline state,
single-attribute audit sourcing, intersectional not-estimable handling,
and pre/post mitigation comparative evaluations.
"""

import pytest
import pandas as pd
import numpy as np

from dashboard.utils import (
    load_dataset_run_result,
    normalize_run_results,
    get_step4_baseline_fairness_scope,
    get_step4_comparative_evaluations,
    execute_interactive_mitigation_workflow
)


def test_step4_baseline_fairness_sourced_from_single_attribute_audit_gender():
    """
    Verify Step 4 reads completed Step 3 single-attribute audit for 'gender' scope
    on final_project_loan_demo__1:
    - EOD = 0.1097
    - DPD = 0.0853
    - EOppD = 0.0971
    - DIR = 0.9034
    """
    raw_run = load_dataset_run_result("final_project_loan_demo__1")
    assert raw_run is not None, "final_project_loan_demo__1 run results not found"
    
    norm = normalize_run_results(raw_run)
    scope = get_step4_baseline_fairness_scope(norm, selected_attribute="gender")

    assert scope["scope_name"] == "Gender"
    assert scope["scope_label"] == "Baseline Gender Fairness"
    assert scope["equalized_odds_difference"] == pytest.approx(0.1097, abs=1e-4)
    assert scope["demographic_parity_difference"] == pytest.approx(0.0853, abs=1e-4)
    assert scope["equal_opportunity_difference"] == pytest.approx(0.0971, abs=1e-4)
    assert scope["disparate_impact_ratio"] == pytest.approx(0.9034, abs=1e-4)


def test_step4_baseline_fairness_sourced_from_single_attribute_audit_race():
    """
    Verify Step 4 reads completed Step 3 single-attribute audit for 'race' scope
    on final_project_loan_demo__1:
    - EOD = 0.2639
    - DPD = 0.0860
    - EOppD = 0.2000
    - DIR = 0.9044
    """
    raw_run = load_dataset_run_result("final_project_loan_demo__1")
    assert raw_run is not None
    
    norm = normalize_run_results(raw_run)
    scope = get_step4_baseline_fairness_scope(norm, selected_attribute="race")

    assert scope["scope_name"] == "Race"
    assert scope["scope_label"] == "Baseline Race Fairness"
    assert scope["equalized_odds_difference"] == pytest.approx(0.2639, abs=1e-4)
    assert scope["demographic_parity_difference"] == pytest.approx(0.0860, abs=1e-4)
    assert scope["equal_opportunity_difference"] == pytest.approx(0.2000, abs=1e-4)
    assert scope["disparate_impact_ratio"] == pytest.approx(0.9044, abs=1e-4)


def test_step4_intersectional_na_does_not_overwrite_valid_single_attribute_metrics():
    """
    Verify that intersectional N/A (Not Estimable due to N < 30) does not overwrite
    valid single-attribute metrics, and does not fabricate fake 0.0 or 1.0 values.
    """
    raw_run = load_dataset_run_result("final_project_loan_demo__1")
    norm = normalize_run_results(raw_run)
    scope = get_step4_baseline_fairness_scope(norm, selected_attribute="gender")

    # Intersectional status must be NOT_ESTIMABLE with explanation
    assert scope["intersectional_estimable"] is False
    assert scope["intersectional_status"] == "NOT_ESTIMABLE"
    assert "not estimable" in scope["intersectional_message"].lower()

    # Single-attribute disparities must be populated with real numbers, not None, not fabricated 0 or 1
    assert scope["equalized_odds_difference"] is not None
    assert 0.0 < scope["equalized_odds_difference"] < 1.0
    assert scope["demographic_parity_difference"] is not None
    assert 0.0 < scope["demographic_parity_difference"] < 1.0


def test_step4_initial_state_before_mitigation_execution():
    """
    Verify that before mitigation is executed, Step 4 does not present identical Before/After
    values or +0.0000 fairness gains as a completed run, but displays an explicit initial state.
    """
    comp = get_step4_comparative_evaluations(active_mit=None, selected_attribute="gender")
    assert comp["has_executed"] is False
    assert "Mitigation not executed yet" in comp["status_message"]
    assert len(comp["table_data"]) == 0
    assert comp["fairness_gains"]["eod_gain"] is None
    assert comp["fairness_gains"]["dpd_gain"] is None


def test_step4_after_execution_populates_actual_metrics_and_deltas():
    """
    Verify that after interactive mitigation execution, actual Before vs After comparative
    evaluations, percentage changes, and real fairness gains are computed.
    """
    mock_active_mit = {
        "status": "SUCCESS",
        "dataset_id": "final_project_loan_demo__1",
        "baseline": {
            "performance": {"accuracy": 0.6833, "f1_score": 0.7841},
            "calibration": {"brier_score": 0.1974, "ece": 0.0512},
            "single_attribute": {
                "gender": {
                    "disparities": {
                        "equalized_odds_difference": 0.1097,
                        "demographic_parity_difference": 0.0853,
                        "disparate_impact_ratio": 0.9034
                    }
                }
            }
        },
        "mitigated": {
            "performance": {"accuracy": 0.6500, "f1_score": 0.7500},
            "calibration": {"brier_score": 0.2100, "ece": 0.0450},
            "single_attribute": {
                "gender": {
                    "disparities": {
                        "equalized_odds_difference": 0.0350,
                        "demographic_parity_difference": 0.0210,
                        "disparate_impact_ratio": 0.9500
                    }
                }
            }
        }
    }

    comp = get_step4_comparative_evaluations(mock_active_mit, selected_attribute="gender")
    assert comp["has_executed"] is True
    assert len(comp["table_data"]) > 0

    # Verify rows in table
    metrics_by_name = {row["Metric"]: row for row in comp["table_data"]}
    assert "Accuracy" in metrics_by_name
    assert metrics_by_name["Accuracy"]["Before"] == "0.6833"
    assert metrics_by_name["Accuracy"]["After"] == "0.6500"
    assert metrics_by_name["Accuracy"]["Delta (Δ)"] == "-0.0333"

    assert "Equalized Odds Difference (Gender)" in metrics_by_name
    assert metrics_by_name["Equalized Odds Difference (Gender)"]["Before"] == "0.1097"
    assert metrics_by_name["Equalized Odds Difference (Gender)"]["After"] == "0.0350"
    assert metrics_by_name["Equalized Odds Difference (Gender)"]["Delta (Δ)"] == "-0.0747"

    # Verify gains
    fg = comp["fairness_gains"]
    assert fg["eod_gain"] == pytest.approx(0.1097 - 0.0350, abs=1e-4)
    assert fg["dpd_gain"] == pytest.approx(0.0853 - 0.0210, abs=1e-4)
    assert fg["scope_name"] == "Gender"
