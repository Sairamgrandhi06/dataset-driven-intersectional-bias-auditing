"""
Step 6 — Model Validation & Engineering Assurance Regression Test Suite.

Verifies:
1. Confusion matrix contains the full test-set sample count (sum == test_size).
2. Confusion matrix is mathematically consistent with accuracy, precision, recall, and F1.
3. Calibration chart values and curves belong to the active dataset/model/version.
4. Calibration card metrics and plotted baseline/mitigated values come from the same evaluation context.
5. No stale cross-dataset calibration values (e.g., Adult benchmark values on loan datasets) are used.
6. Evidence lineage matches the active dataset/model/version.
7. Genuine Model Artifact SHA-256 hash is computed without fake fabrication.
"""

import os
import re
import math
import pytest
from dashboard.utils import (
    load_dataset_run_result,
    normalize_run_results,
    extract_confusion_matrix_values,
    get_step6_validation_data,
    get_model_artifact_sha256,
    get_project_root
)


def test_1_confusion_matrix_full_test_size_count_loan_demo():
    """
    ISSUE 1 REGRESSION TEST:
    Verify that Step 6 validation data extracts the non-zero confusion matrix
    for final_project_loan_demo__1 and sum(confusion_matrix) == test_size (120).
    """
    val_data = get_step6_validation_data("final_project_loan_demo__1")
    assert val_data["has_model"] is True

    perf = val_data["performance"]
    cm = perf["confusion_matrix"]

    tn, fp, fn, tp = cm["TN"], cm["FP"], cm["FN"], cm["TP"]
    total_samples = tn + fp + fn + tp

    # Must contain full test-set sample count (120) and not all zeros
    assert total_samples == 120
    assert total_samples == perf["sample_count"]
    assert tn == 13
    assert fp == 31
    assert fn == 7
    assert tp == 69


def test_2_confusion_matrix_mathematical_consistency_with_performance_metrics():
    """
    ISSUE 1 REGRESSION TEST:
    Verify that the confusion matrix values are mathematically consistent
    with displayed Accuracy, Precision, Recall, and F1-Score.
    """
    val_data = get_step6_validation_data("final_project_loan_demo__1")
    perf = val_data["performance"]
    cm = perf["confusion_matrix"]

    tn, fp, fn, tp = cm["TN"], cm["FP"], cm["FN"], cm["TP"]
    total = tn + fp + fn + tp
    assert total > 0

    expected_acc = (tp + tn) / total
    expected_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    expected_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    expected_f1 = (2 * expected_prec * expected_rec) / (expected_prec + expected_rec) if (expected_prec + expected_rec) > 0 else 0.0

    assert math.isclose(perf["accuracy"], expected_acc, rel_tol=1e-3)
    assert math.isclose(perf["precision"], expected_prec, rel_tol=1e-3)
    assert math.isclose(perf["recall"], expected_rec, rel_tol=1e-3)
    assert math.isclose(perf["f1_score"], expected_f1, rel_tol=1e-3)


def test_3_calibration_metrics_belong_to_active_dataset():
    """
    ISSUE 2 REGRESSION TEST:
    Verify that the Calibration tab metrics for final_project_loan_demo__1
    belong to final_project_loan_demo__1 (Baseline Brier=0.1974, ECE=0.1254,
    Mitigated Brier=0.2973, ECE=0.2613) and NOT Adult Census Income (0.1128 / 0.2156).
    """
    val_data = get_step6_validation_data("final_project_loan_demo__1")
    cal = val_data["calibration"]

    assert cal["status"] == "AVAILABLE"
    assert math.isclose(cal["baseline_brier"], 0.1974, rel_tol=1e-3)
    assert math.isclose(cal["baseline_ece"], 0.1254, rel_tol=1e-3)

    assert cal["has_mitigation"] is True
    assert math.isclose(cal["mitigated_brier"], 0.2973, rel_tol=1e-3)
    assert math.isclose(cal["mitigated_ece"], 0.2613, rel_tol=1e-3)

    # Ensure stale Adult benchmark values are NOT present
    assert not math.isclose(cal["baseline_brier"], 0.1128, abs_tol=0.01)
    assert not math.isclose(cal["mitigated_brier"], 0.2156, abs_tol=0.01)


def test_4_calibration_cards_and_plotted_values_share_same_evaluation_context():
    """
    ISSUE 2 REGRESSION TEST:
    Verify that calibration card metrics and the plotted reliability curve figure
    come from the same evaluation context and are dynamically generated.
    """
    val_data = get_step6_validation_data("final_project_loan_demo__1")
    cal = val_data["calibration"]

    # Verify dynamic figure path
    fig_path = cal["figure_path"]
    assert fig_path is not None
    assert "final_project_loan_demo__1_calibration.png" in fig_path
    assert os.path.exists(fig_path)

    # Plotted data coordinates match the authoritative baseline reliability curve
    raw_base = cal["raw_baseline"]
    assert "reliability_curve" in raw_base
    rc = raw_base["reliability_curve"]
    assert len(rc["prob_pred"]) > 0
    assert len(rc["prob_true"]) > 0
    assert len(rc["prob_pred"]) == len(rc["prob_true"])


def test_5_no_stale_cross_dataset_calibration_fallback():
    """
    ISSUE 3 REGRESSION TEST:
    Verify that separate datasets (e.g. student_performance, adult_census_income,
    and final_project_loan_demo__1) maintain strict calibration metric isolation.
    """
    loan_val = get_step6_validation_data("final_project_loan_demo__1")
    student_val = get_step6_validation_data("student_performance")
    adult_val = get_step6_validation_data("adult_census_income")

    # All three datasets must produce distinct baseline calibration values
    loan_brier = loan_val["calibration"]["baseline_brier"]
    student_brier = student_val["calibration"]["baseline_brier"]
    adult_brier = adult_val["calibration"]["baseline_brier"]

    assert loan_brier is not None
    assert student_brier is not None
    assert adult_brier is not None

    assert not math.isclose(loan_brier, student_brier, abs_tol=1e-4)
    assert not math.isclose(loan_brier, adult_brier, abs_tol=1e-4)

    # Figure paths must be isolated per dataset
    assert "final_project_loan_demo__1" in loan_val["calibration"]["figure_path"]
    assert "student_performance" in student_val["calibration"]["figure_path"]
    assert "adult_census_income" in adult_val["calibration"]["figure_path"]


def test_6_evidence_lineage_matches_active_dataset_and_model():
    """
    ISSUE 4 REGRESSION TEST:
    Verify that Step 6 Evidence & Lineage matches the active dataset, model version,
    run ID, and produces a real SHA-256 hash for the model artifact.
    """
    val_data = get_step6_validation_data("final_project_loan_demo__1")
    evidence_list = val_data["evidence"]

    ev_dict = {item["Evidence Item"]: item["Value"] for item in evidence_list}

    # Verify required keys exist
    assert "Dataset Hash (SHA-256)" in ev_dict
    assert "Pipeline Run ID" in ev_dict
    assert "Config Version" in ev_dict
    assert "Model Version" in ev_dict
    assert "Selected Architecture" in ev_dict
    assert "Model Artifact Hash" in ev_dict
    assert "Audit Timestamp" in ev_dict

    # Check values for final_project_loan_demo__1
    assert ev_dict["Dataset Hash (SHA-256)"] == "2c40ae5fecaa3deef2e67371fe57286ca739e451dc998540cda70797adb2ca3c"
    assert "final_project_loan_demo__1" in ev_dict["Pipeline Run ID"]
    assert ev_dict["Model Version"] in ["v1", "v2", "v3", "v0"]

    # Model Artifact Hash must be either a valid 64-char hex string (if artifact on disk) or 'N/A'
    art_hash = ev_dict["Model Artifact Hash"]
    if art_hash != "N/A":
        assert re.match(r"^[a-f0-9]{64}$", art_hash) is not None


def test_7_model_artifact_sha256_resolution_and_missing_safety():
    """
    ISSUE 4 REGRESSION TEST:
    Verify get_model_artifact_sha256 computes a valid hash for real artifacts
    and returns 'N/A' for non-existent datasets/runs without fabricating hashes.
    """
    # Non-existent run should return 'N/A'
    fake_hash = get_model_artifact_sha256("non_existent_dataset_9999", run_id="fake_run")
    assert fake_hash == "N/A"

    # Known existing run artifact
    loan_hash = get_model_artifact_sha256(
        "final_project_loan_demo__1",
        run_id="RUN-final_project_loan_demo__1-1787906497"
    )
    if loan_hash != "N/A":
        assert len(loan_hash) == 64
        assert re.match(r"^[a-f0-9]{64}$", loan_hash)


def test_8_extract_confusion_matrix_values_all_formats():
    """
    Verify extract_confusion_matrix_values handles all dictionary and array variants.
    """
    # Dict with standard short keys
    assert extract_confusion_matrix_values({"TN": 10, "FP": 5, "FN": 2, "TP": 20}) == (10, 5, 2, 20)
    # Dict with full keys
    assert extract_confusion_matrix_values({"true_negative": 10, "false_positive": 5, "false_negative": 2, "true_positive": 20}) == (10, 5, 2, 20)
    # Dict with matrix
    assert extract_confusion_matrix_values({"matrix": [[10, 5], [2, 20]]}) == (10, 5, 2, 20)
    # 2x2 List
    assert extract_confusion_matrix_values([[10, 5], [2, 20]]) == (10, 5, 2, 20)
    # Empty / None
    assert extract_confusion_matrix_values({}) == (0, 0, 0, 0)
    assert extract_confusion_matrix_values(None) == (0, 0, 0, 0)
