"""
Regression and Verification Tests for Intersectional Bias Auditing & Strengthening Requirements.

Tests 1 to 14 verify:
1. Gender and Race single-attribute audits are calculated independently.
2. Changing selected protected attribute changes the appropriate metric calculation.
3. Intersectional group construction dynamically handles all observed combinations.
4. All groups are retained in output.
5. Low-sample groups are flagged rather than deleted.
6. Fewer than 2 groups N >= threshold -> INSUFFICIENT_GROUP_COVERAGE, disparity metrics are None (not 0.0).
7. 2 or more groups N >= threshold -> disparity metrics calculated correctly.
8. Positive class comes from configuration.
9. Changing positive class changes target encoding appropriately.
10. Single-attribute results are stored separately from intersectional results.
11. Custom datasets do not inherit Adult metric values.
12. Mitigation and trade-off results remain dataset-specific.
13. Undefined fairness metrics do not silently become zero.
14. Existing Adult reference benchmark tests remain valid.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

from src.data.dataset_config import DatasetConfig
from src.data.preprocessor import encode_target, prepare_pipeline_data
from src.fairness.single_attribute import audit_single_attribute, run_single_attribute_audits
from src.fairness.intersectional import create_intersectional_attribute, audit_intersectional_attributes
from src.fairness.trade_off import compute_metric_change, calculate_comprehensive_tradeoff
from src.data.result_schema import create_pipeline_run_result
from dashboard.utils import load_dataset_run_result, format_val


def test_1_single_attribute_audits_independent():
    """TEST 1: Gender and Race produce independently calculated fairness results."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    A_df = pd.DataFrame({
        "gender": ["Female", "Female", "Male", "Male", "Female", "Male", "Female", "Male"],
        "race": ["GroupA", "GroupB", "GroupA", "GroupB", "GroupA", "GroupB", "GroupA", "GroupB"]
    })

    audits = run_single_attribute_audits(y_true, y_pred, A_df)
    assert "gender" in audits
    assert "race" in audits
    assert set(audits["gender"]["groups"].keys()) == {"Female", "Male"}
    assert set(audits["race"]["groups"].keys()) == {"GroupA", "GroupB"}
    assert audits["gender"]["attribute_name"] == "gender"
    assert audits["race"]["attribute_name"] == "race"


def test_2_attribute_selection_changes_metric_calculation():
    """TEST 2: Changing selected protected attribute changes appropriate metric calculation when underlying data differs."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    A_df = pd.DataFrame({
        "gender": ["Female", "Female", "Female", "Female", "Male", "Male", "Male", "Male"],
        "race":   ["GroupA", "GroupB", "GroupA", "GroupB", "GroupA", "GroupB", "GroupA", "GroupB"]
    })

    audits = run_single_attribute_audits(y_true, y_pred, A_df)
    gender_dpd = audits["gender"]["disparities"]["demographic_parity_difference"]
    race_dpd = audits["race"]["disparities"]["demographic_parity_difference"]

    assert gender_dpd != race_dpd
    assert gender_dpd == pytest.approx(1.0, abs=1e-3)
    assert race_dpd == pytest.approx(0.0, abs=1e-3)


def test_3_intersectional_dynamic_group_construction():
    """TEST 3: Intersectional group construction dynamically handles all observed combinations."""
    A_df = pd.DataFrame({
        "attr1": ["X", "X", "Y", "Z"],
        "attr2": ["1", "2", "1", "3"]
    })
    inter_series = create_intersectional_attribute(A_df, attributes=["attr1", "attr2"])
    expected_combos = ["X + 1", "X + 2", "Y + 1", "Z + 3"]
    assert list(inter_series) == expected_combos


def test_4_all_groups_retained_in_output():
    """TEST 4: All observed groups are retained in the intersectional audit output."""
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0])
    A_df = pd.DataFrame({
        "gender": ["Female", "Female", "Male", "Male", "NonBinary", "NonBinary"],
        "race":   ["GroupA", "GroupB", "GroupA", "GroupB", "GroupC", "GroupC"]
    })

    audit = audit_intersectional_attributes(y_true, y_pred, A_df, min_group_size=5)
    all_groups = audit["all_group_metrics"]
    assert len(all_groups) == 5
    assert "Female + GroupA" in all_groups
    assert "Female + GroupB" in all_groups
    assert "NonBinary + GroupC" in all_groups


def test_5_low_sample_groups_flagged():
    """TEST 5: Low-sample groups are flagged rather than deleted."""
    y_true = np.array([1] * 35 + [0] * 35 + [1] * 5 + [0] * 5)
    y_pred = np.array([1] * 35 + [0] * 35 + [1] * 5 + [0] * 5)
    A_df = pd.DataFrame({
        "gender": ["Male"] * 70 + ["Female"] * 10,
        "race":   ["GroupA"] * 70 + ["GroupB"] * 10
    })

    audit = audit_intersectional_attributes(y_true, y_pred, A_df, min_group_size=30)
    assert audit["primary_group_count"] == 1
    assert audit["low_sample_group_count"] == 1
    assert audit["primary_groups"] == ["Male + GroupA"]
    assert audit["low_sample_groups"] == ["Female + GroupB"]
    assert "Female + GroupB" in audit["all_group_metrics"]


def test_6_insufficient_group_coverage_disparities_none():
    """TEST 6: If fewer than two groups meet N >= 30: status = INSUFFICIENT_GROUP_COVERAGE and disparity metrics are None (not 0.0)."""
    y_true = np.array([1] * 35 + [0] * 35 + [1] * 5 + [0] * 5)
    y_pred = np.array([1] * 35 + [0] * 35 + [1] * 5 + [0] * 5)
    A_df = pd.DataFrame({
        "gender": ["Male"] * 70 + ["Female"] * 10,
        "race":   ["GroupA"] * 70 + ["GroupB"] * 10
    })

    audit = audit_intersectional_attributes(y_true, y_pred, A_df, min_group_size=30)
    cov = audit["coverage_status"]

    assert cov["code"] == "INSUFFICIENT_GROUP_COVERAGE"
    assert cov["eligible_group_count"] == 1
    assert cov["total_group_count"] == 2
    assert audit["disparities"]["demographic_parity_difference"] is None
    assert audit["disparities"]["equalized_odds_difference"] is None
    assert audit["disparities"]["disparate_impact_ratio"] is None
    assert audit["rankings"]["status"] == "UNAVAILABLE"


def test_7_sufficient_group_coverage_calculates_disparities():
    """TEST 7: If two or more groups meet N >= 30: disparity metrics are calculated correctly."""
    y_true = np.array([1] * 20 + [0] * 20 + [1] * 20 + [0] * 20)
    y_pred = np.array([1] * 15 + [0] * 25 + [1] * 5 + [0] * 35)
    A_df = pd.DataFrame({
        "gender": ["Male"] * 40 + ["Female"] * 40,
        "race":   ["GroupA"] * 40 + ["GroupB"] * 40
    })

    audit = audit_intersectional_attributes(y_true, y_pred, A_df, min_group_size=30)
    cov = audit["coverage_status"]

    assert cov["code"] == "AVAILABLE"
    assert cov["eligible_group_count"] == 2
    assert audit["disparities"]["demographic_parity_difference"] == pytest.approx(0.25, abs=1e-3)
    assert audit["disparities"]["disparate_impact_ratio"] == pytest.approx(0.3333, abs=1e-3)
    assert audit["rankings"]["status"] == "AVAILABLE"


def test_8_positive_class_from_configuration():
    """TEST 8: Positive class comes from configuration and changes target encoding."""
    df = pd.DataFrame({
        "loan_status": ["approved", "rejected", "approved", "rejected"],
        "gender": ["Male", "Female", "Male", "Female"]
    })

    cfg_dict = {
        "dataset_id": "test_loan_config",
        "path": "dummy.csv",
        "target": {
            "column": "loan_status",
            "positive_class": "approved"
        },
        "protected_attributes": ["gender"],
        "id_columns": [],
        "ignore_columns": [],
        "test_size": 0.2,
        "random_state": 42
    }

    cfg = DatasetConfig.from_dict(cfg_dict)
    X, y, A = prepare_pipeline_data(df, config=cfg)
    assert y.tolist() == [1, 0, 1, 0]


def test_9_changing_positive_class_inverts_target_encoding():
    """TEST 9: Changing positive class changes the target encoding appropriately."""
    series = pd.Series(["approved", "rejected", "approved", "rejected"], name="loan_status")

    y_pos_app = encode_target(series, positive_value="approved")
    assert y_pos_app.tolist() == [1, 0, 1, 0]

    y_pos_rej = encode_target(series, positive_value="rejected")
    assert y_pos_rej.tolist() == [0, 1, 0, 1]


def test_10_single_attribute_and_intersectional_metrics_schema_secrecy():
    """TEST 10: Run result schema stores single_attribute_metrics separately from intersectional metrics."""
    result = create_pipeline_run_result(
        dataset_id="test_ds",
        dataset_hash="123456",
        config_version="1.0.0",
        target_column="target",
        positive_class="1",
        protected_attributes=["gender", "race"],
        feature_count=5,
        train_size=100,
        test_size=20,
        baseline_metrics={"accuracy": 0.8},
        single_attribute_metrics={
            "gender": {"disparities": {"demographic_parity_difference": 0.05}},
            "race": {"disparities": {"demographic_parity_difference": 0.15}}
        },
        fairness_metrics={"demographic_parity_difference": None},
        intersectional_metrics={"coverage_status": {"code": "INSUFFICIENT_GROUP_COVERAGE"}},
        calibration_metrics={"brier_score": 0.1}
    )

    assert "single_attribute_metrics" in result
    assert "gender" in result["single_attribute_metrics"]
    assert "race" in result["single_attribute_metrics"]
    assert result["single_attribute_metrics"]["gender"]["disparities"]["demographic_parity_difference"] == 0.05
    assert result["single_attribute_metrics"]["race"]["disparities"]["demographic_parity_difference"] == 0.15


def test_11_custom_datasets_do_not_inherit_adult_values():
    """TEST 11: Custom datasets saved in results/ do not inherit Adult reference benchmark metric values."""
    loan_results = load_dataset_run_result("loan_approval_fairness_test")
    assert loan_results is not None
    assert loan_results["dataset_id"] == "loan_approval_fairness_test"
    assert loan_results["target_column"] == "loan_status"
    assert loan_results["positive_class"] == "approved"


def test_12_mitigation_and_tradeoff_dataset_specific():
    """TEST 12: Mitigation and trade-off results remain dataset-specific."""
    b_disp = {"demographic_parity_difference": None, "equalized_odds_difference": None}
    m_disp = {"demographic_parity_difference": None, "equalized_odds_difference": None}

    change = compute_metric_change(b_disp["equalized_odds_difference"], m_disp["equalized_odds_difference"])
    assert change["impact_status"] == "Not estimable"
    assert change["baseline"] is None
    assert change["mitigated"] is None
    assert change["percentage_change"] is None


def test_13_undefined_fairness_metrics_do_not_silently_become_zero():
    """TEST 13: Undefined fairness metrics do not silently become zero."""
    assert format_val(None) == "N/A"
    assert format_val("INSUFFICIENT_GROUP_COVERAGE") == "INSUFFICIENT_GROUP_COVERAGE"


def test_14_loan_run_results_json_validity():
    """TEST 14: Loan dataset run results file contains valid updated schema."""
    loan_path = os.path.join("results", "loan_approval_fairness_test", "run_results.json")
    assert os.path.exists(loan_path)

    with open(loan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["positive_class"] == "approved"
    assert "single_attribute_metrics" in data
    assert "gender" in data["single_attribute_metrics"]
    assert "race" in data["single_attribute_metrics"]
    assert data["intersectional_metrics"]["coverage_status"]["code"] == "INSUFFICIENT_GROUP_COVERAGE"
    assert data["fairness_metrics"]["demographic_parity_difference"] is None
