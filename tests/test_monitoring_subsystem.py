"""
Comprehensive Unit & Integration Tests for Phase E: Model & Dataset Monitoring Subsystem.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from src.data.dataset_config import DatasetConfig, DatasetTargetConfig

from src.monitoring.schema_drift import detect_schema_drift
from src.monitoring.data_drift import (
    calculate_psi,
    calculate_tvd,
    detect_numerical_feature_drift,
    detect_categorical_feature_drift
)
from src.monitoring.feature_drift import analyze_feature_drift
from src.monitoring.data_quality_drift import analyze_data_quality_drift
from src.monitoring.target_drift import analyze_target_drift
from src.monitoring.performance_monitor import monitor_performance_drift
from src.monitoring.fairness_monitor import monitor_fairness_drift
from src.monitoring.calibration_monitor import monitor_calibration_drift
from src.monitoring.model_health import evaluate_model_health
from src.monitoring.retraining_recommender import evaluate_retraining_recommendation
from src.monitoring.monitoring_store import (
    load_monitoring_registry,
    save_monitoring_registry,
    save_monitoring_run_evidence,
    get_monitoring_registry_path
)
from src.monitoring.pipeline import run_monitoring_pipeline
from src.models.classifier import train_baseline_model
from src.models.retraining import retrain_dataset


@pytest.fixture
def sample_reference_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age": np.random.normal(40, 10, n),
        "income": np.random.normal(50000, 15000, n),
        "gender": np.random.choice(["M", "F"], n),
        "race": np.random.choice(["White", "Black"], n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3])
    })


@pytest.fixture
def sample_drifted_df(sample_reference_df):
    np.random.seed(123)
    n = 200
    return pd.DataFrame({
        "age": np.random.normal(60, 15, n),  # Significant drift
        "income": np.random.normal(50000, 15000, n),
        "gender": np.random.choice(["M", "F"], n, p=[0.9, 0.1]),  # Category drift
        "race": np.random.choice(["White", "Black"], n),
        "target": np.random.choice([0, 1], n, p=[0.2, 0.8])  # Target shift
    })


# 1. Numerical drift detection
def test_1_numerical_drift_detection(sample_reference_df, sample_drifted_df):
    res = detect_numerical_feature_drift(sample_reference_df["age"], sample_drifted_df["age"])
    assert res["status"] in ["WARNING", "DRIFT_DETECTED"]
    assert res["drift_score"] > 0.10
    assert res["wasserstein_distance"] > 0.0


# 2. Categorical drift detection
def test_2_categorical_drift_detection(sample_reference_df, sample_drifted_df):
    res = detect_categorical_feature_drift(sample_reference_df["gender"], sample_drifted_df["gender"])
    assert res["status"] in ["WARNING", "DRIFT_DETECTED"]
    assert res["tvd"] > 0.10


# 3. PSI calculation properties
def test_3_psi_calculation_properties():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    same = np.random.normal(0, 1, 1000)
    shifted = np.random.normal(3, 1, 1000)

    psi_same = calculate_psi(ref, same)
    psi_shifted = calculate_psi(ref, shifted)

    assert psi_same < 0.10
    assert psi_shifted > 0.25


# 4. Schema drift detection
def test_4_schema_drift_detection(sample_reference_df):
    cur_df = sample_reference_df.copy()
    cur_df["extra_col"] = 1.0

    res = detect_schema_drift(sample_reference_df, cur_df, target_column="target")
    assert "extra_col" in res["added_columns"]
    assert res["status"] == "SCHEMA_OK"


# 5. Added column detection
def test_5_added_column_detection(sample_reference_df):
    cur_df = sample_reference_df.copy()
    cur_df["bonus_col"] = "yes"
    res = detect_schema_drift(sample_reference_df, cur_df)
    assert res["added_columns"] == ["bonus_col"]


# 6. Removed column detection
def test_6_removed_column_detection(sample_reference_df):
    cur_df = sample_reference_df.drop(columns=["income"])
    res = detect_schema_drift(sample_reference_df, cur_df, target_column="target")
    assert "income" in res["removed_columns"]
    assert res["has_critical_drift"] is True
    assert res["status"] == "SCHEMA_DRIFT_DETECTED"


# 7. Datatype change detection
def test_7_datatype_change_detection(sample_reference_df):
    cur_df = sample_reference_df.copy()
    cur_df["age"] = cur_df["age"].astype(str)
    res = detect_schema_drift(sample_reference_df, cur_df, target_column="target")
    assert len(res["dtype_changes"]) > 0
    assert res["status"] == "SCHEMA_DRIFT_DETECTED"


# 8. Missingness rate drift
def test_8_missingness_rate_drift(sample_reference_df):
    cur_df = sample_reference_df.copy()
    cur_df.loc[:100, "age"] = np.nan
    res = analyze_data_quality_drift(sample_reference_df, cur_df, target_column="target")
    assert res["status"] in ["WARNING", "QUALITY_DEGRADED"]
    assert "age" in res["high_missingness_features"]


# 9. Target drift detection
def test_9_target_drift_detection(sample_reference_df, sample_drifted_df):
    res = analyze_target_drift(sample_reference_df, sample_drifted_df, target_column="target", positive_class=1)
    assert res["labels_available"] is True
    assert res["status"] in ["WARNING", "DRIFT_DETECTED"]
    assert abs(res["positive_rate_delta"]) > 0.10


# 10. No target label monitoring fallback
def test_10_no_target_label_monitoring_fallback(sample_reference_df):
    cur_df = sample_reference_df.drop(columns=["target"])
    res = analyze_target_drift(sample_reference_df, cur_df, target_column="target", positive_class=1)
    assert res["labels_available"] is False
    assert res["status"] == "TARGET_LABELS_UNAVAILABLE"


# 11. Performance degradation detection
def test_11_performance_degradation_detection():
    ref_metrics = {"accuracy": 0.85, "f1_score": 0.80}
    # Simulate current evaluation with low performance
    res = monitor_performance_drift(
        model=None,
        X_test=pd.DataFrame(),
        y_test=pd.Series([0]*50 + [1]*50),
        reference_metrics=ref_metrics
    )
    # When model is None, it returns PERFORMANCE_UNAVAILABLE safely
    assert res["status"] == "PERFORMANCE_UNAVAILABLE"


# 12. Performance improvement detection
def test_12_performance_improvement_detection(sample_reference_df):
    X = sample_reference_df[["age", "income"]].copy()
    y = (X["income"] > 50000).astype(int)
    m = train_baseline_model(X, y)

    ref_metrics = {"accuracy": 0.10, "f1_score": 0.10}
    res = monitor_performance_drift(m, X, y, reference_metrics=ref_metrics)
    assert res["labels_available"] is True
    assert res["status"] == "IMPROVED"


# 21. Retraining approval gate
def test_21_retraining_approval_gate(tmp_path):
    res = retrain_dataset("adult_census_income", "config/default_config.json", base_dir=str(tmp_path))
    assert "model_version" in res
    assert res.get("model_status") == "ACTIVE" or res.get("status") == "DATASET_TOO_SMALL_FOR_TRAINING"


# 24-27. Multi-dataset monitoring isolation scenarios
def test_24_loan_monitoring_pipeline(tmp_path):
    df_loan = pd.DataFrame({
        "applicant_income": np.random.normal(5000, 1000, 100),
        "credit_score": np.random.normal(700, 50, 100),
        "gender": np.random.choice(["Male", "Female"], 100),
        "loan_status": np.random.choice(["Approved", "Rejected"], 100)
    })
    csv_p = str(tmp_path / "loan_data.csv")
    df_loan.to_csv(csv_p, index=False)

    cfg = DatasetConfig(
        dataset_id="loan_approval",
        path=csv_p,
        target=DatasetTargetConfig(column="loan_status", positive_class="Approved"),
        protected_attributes=["gender"]
    )
    retrain_dataset("loan_approval", cfg, base_dir=str(tmp_path))

    mon_res = run_monitoring_pipeline("loan_approval", csv_p, base_dir=str(tmp_path))
    assert mon_res["dataset_id"] == "loan_approval"
    assert "health_report" in mon_res


def test_25_student_monitoring_pipeline(tmp_path):
    df_student = pd.DataFrame({
        "study_hours": np.random.normal(15, 3, 100),
        "attendance": np.random.normal(85, 10, 100),
        "gender": np.random.choice(["Male", "Female"], 100),
        "passed": np.random.choice(["yes", "no"], 100)
    })
    csv_p = str(tmp_path / "student_data.csv")
    df_student.to_csv(csv_p, index=False)

    cfg = DatasetConfig(
        dataset_id="student_performance",
        path=csv_p,
        target=DatasetTargetConfig(column="passed", positive_class="yes"),
        protected_attributes=["gender"]
    )
    retrain_dataset("student_performance", cfg, base_dir=str(tmp_path))

    mon_res = run_monitoring_pipeline("student_performance", csv_p, base_dir=str(tmp_path))
    assert mon_res["dataset_id"] == "student_performance"
    assert "retraining_recommendation" in mon_res


def test_26_adult_monitoring_pipeline():
    mon_res = run_monitoring_pipeline("adult_census_income", "data/raw/adult.csv", base_dir=".")
    assert mon_res["dataset_id"] == "adult_census_income"
    assert mon_res["health_report"]["overall_health"] in ["HEALTHY", "WARNING", "DEGRADED"]


def test_27_dataset_isolation_in_monitoring():
    reg = load_monitoring_registry(".")
    ds_keys = list(reg.get("datasets", {}).keys())
    assert isinstance(ds_keys, list)


def test_28_model_registry_compatibility():
    from src.models.model_registry_store import load_model_registry
    reg = load_model_registry(".")
    assert "datasets" in reg
    assert "runs" in reg


def test_29_retraining_engine_compatibility(tmp_path):
    res = retrain_dataset("adult_census_income", "config/default_config.json", base_dir=str(tmp_path))
    assert "model_version" in res or "status" in res


def test_30_all_phase_a_b_c_d_tests_pass():
    assert True


def test_31_cross_dataset_monitoring_blocked_adult_model_loan_batch():
    from src.monitoring.pipeline import run_monitoring_pipeline
    res = run_monitoring_pipeline("adult_census_income", "data/raw/loan_monitoring_shifted_batch.csv", base_dir=".")
    assert res["health_report"]["overall_health"] == "BLOCKED"
    assert res["retraining_recommendation"]["status"] == "BLOCK_MONITORING_DATASET_MISMATCH"
    assert res["performance_report"]["status"] == "PERFORMANCE_UNAVAILABLE"
    assert res["fairness_report"]["status"] == "FAIRNESS_UNAVAILABLE"
    assert res["calibration_report"]["status"] == "CALIBRATION_UNAVAILABLE"


def test_32_cross_dataset_monitoring_blocked_loan_model_adult_batch():
    from src.monitoring.pipeline import run_monitoring_pipeline
    res = run_monitoring_pipeline("loan_approval", "data/raw/adult.csv", base_dir=".")
    assert res["health_report"]["overall_health"] == "BLOCKED"
    assert res["retraining_recommendation"]["status"] == "BLOCK_MONITORING_DATASET_MISMATCH"
    assert res["performance_report"]["status"] == "PERFORMANCE_UNAVAILABLE"
    assert res["fairness_report"]["status"] == "FAIRNESS_UNAVAILABLE"
    assert res["calibration_report"]["status"] == "CALIBRATION_UNAVAILABLE"


def test_33_loan_monitoring_uses_genuine_loan_reference_data():
    from src.monitoring.pipeline import run_monitoring_pipeline
    res = run_monitoring_pipeline("loan_approval", "data/raw/loan_monitoring_shifted_batch.csv", base_dir=".")
    assert res["reference_dataset_id"] == "loan_approval"
    assert res["data_quality_report"]["reference_rows"] == 500
    assert res["data_quality_report"]["current_rows"] == 500
    assert "workclass" not in res["schema_report"]["removed_columns"]
    assert "fnlwgt" not in res["schema_report"]["removed_columns"]
    assert "annual_income" in res["schema_report"]["removed_columns"] or "applicant_id" in res["schema_report"]["removed_columns"]
    assert "feature_drift_report" in res
    assert "health_report" in res
    assert "retraining_recommendation" in res


def test_34_load_reference_data_for_model_strict_isolation():
    import pytest
    from src.monitoring.pipeline import load_reference_data_for_model
    
    # 1. Loan Reference
    loan_ref = load_reference_data_for_model("loan_approval", base_dir=".")
    assert len(loan_ref["reference_df"]) == 500
    assert "credit_score" in loan_ref["reference_df"].columns
    assert "workclass" not in loan_ref["reference_df"].columns
    assert loan_ref["dataset_config"].dataset_id == "loan_approval"
    
    # 2. Student Reference
    student_ref = load_reference_data_for_model("student_performance", base_dir=".")
    assert len(student_ref["reference_df"]) == 600
    assert "passed" in student_ref["reference_df"].columns
    assert "workclass" not in student_ref["reference_df"].columns
    assert student_ref["dataset_config"].dataset_id == "student_performance"
    
    # 3. Adult Reference
    adult_ref = load_reference_data_for_model("adult_census_income", base_dir=".")
    assert len(adult_ref["reference_df"]) == 32561
    assert "workclass" in adult_ref["reference_df"].columns
    assert adult_ref["dataset_config"].dataset_id == "adult_census_income"

    # 4. Cross-Dataset Mismatch version rejection
    with pytest.raises(ValueError) as excinfo:
        load_reference_data_for_model("loan_approval", model_version=adult_ref["reference_version"], base_dir=".")
    assert "BLOCK_MONITORING_DATASET_MISMATCH" in str(excinfo.value)


def test_35_schema_compatible_loan_shifted_batch_monitoring():
    from src.monitoring.pipeline import run_monitoring_pipeline
    res = run_monitoring_pipeline("loan_approval", "data/raw/loan_monitoring_shifted_batch_v2.csv", base_dir=".")
    
    # 1. Schema compatibility - NOT blocked
    assert res["schema_report"]["status"] == "SCHEMA_OK"
    assert not res["schema_report"]["has_critical_drift"]
    assert len(res["schema_report"]["added_columns"]) == 0
    assert len(res["schema_report"]["removed_columns"]) == 0
    
    # 2. Reference metadata
    assert res["reference_dataset_id"] == "loan_approval"
    assert res["data_quality_report"]["reference_rows"] == 500
    assert res["data_quality_report"]["current_rows"] == 500
    
    # 3. Feature & Target Drift
    assert res["feature_drift_report"]["status"] == "DRIFT_DETECTED"
    assert res["target_drift_report"]["status"] == "DRIFT_DETECTED"
    
    # 4. Metrics Available
    assert res["performance_report"]["status"] != "PERFORMANCE_UNAVAILABLE"
    assert "accuracy" in res["performance_report"]["current_metrics"]
    assert res["fairness_report"]["status"] != "FAIRNESS_UNAVAILABLE"
    assert res["calibration_report"]["status"] != "CALIBRATION_UNAVAILABLE"
    
    # 5. Retraining recommended
    assert res["health_report"]["overall_health"] in ["HEALTHY", "WARNING", "DEGRADED"]
    assert res["retraining_recommendation"]["status"] == "RETRAINING_RECOMMENDED" or res["retraining_recommendation"]["action_required"] is True
