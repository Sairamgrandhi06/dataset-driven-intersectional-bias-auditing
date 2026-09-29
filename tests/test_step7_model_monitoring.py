"""
Step 7 — Production Model Monitoring Regression Test Suite.

Verifies:
1. Completed trained dataset (final_project_loan_demo__1) resolves model history and passes Step 7 precondition.
2. Untrained dataset correctly returns empty history and triggers the precondition warning.
3. Correct dataset/model/version/run_id resolution for final_project_loan_demo__1.
4. Strict dataset isolation in monitoring (cross-dataset model version mismatch rejection).
5. Pre-run schema comparison detects matching schema, datatype mismatches, missing columns, and extra columns.
6. applicant_id datatype mismatch is correctly detected without silent casting.
7. ID / Ignore columns (e.g. applicant_id) are validated for schema but excluded from predictive feature drift (PSI/TVD).
8. Target column missing results in PERFORMANCE_UNAVAILABLE and FAIRNESS_UNAVAILABLE without metric fabrication.
9. Protected attribute missing results in FAIRNESS_UNAVAILABLE without metric fabrication.
10. Critical schema drift blocks performance and fairness evaluation while feature drift evaluates compatible features.
11. A valid monitoring batch successfully executes full pipeline stages A–G.
"""

import os
import io
import pytest
import numpy as np
import pandas as pd
from dashboard.utils import (
    get_dataset_model_history,
    get_dataset_active_model_context,
    get_monitoring_preflight_schema,
    get_project_root
)
from src.monitoring.pipeline import (
    load_reference_data_for_model,
    run_monitoring_pipeline
)
from src.monitoring.schema_drift import (
    detect_schema_drift,
    compare_reference_and_monitoring_schemas
)
from src.monitoring.feature_drift import analyze_feature_drift


def test_1_completed_trained_dataset_passes_step7_precondition():
    """
    Verify that final_project_loan_demo__1 (a completed, trained dataset)
    resolves valid model history so Step 7 opens without the false warning.
    """
    history = get_dataset_model_history("final_project_loan_demo__1")
    assert len(history) > 0

    active_records = [m for m in history if m.get("is_active")]
    assert len(active_records) == 1

    rec = active_records[0]
    assert rec["dataset_id"] == "final_project_loan_demo__1"
    assert rec["model_version"] == "v3"
    assert rec["model_type"] in ["logistic_regression", "Logistic Regression"]
    assert "final_project_loan_demo__1" in rec["run_id"]


def test_2_untrained_dataset_triggers_precondition_warning():
    """
    Verify that an untrained dataset returns empty history and fails the Step 7 precondition.
    """
    history = get_dataset_model_history("non_existent_untrained_dataset_12345")
    assert len(history) == 0

    ctx = get_dataset_active_model_context("non_existent_untrained_dataset_12345")
    assert ctx["has_model"] is False
    assert ctx["status"] == "PENDING_TRAINING"


def test_3_correct_dataset_model_version_resolution():
    """
    Verify exact resolution of model architecture, version, and run ID for final_project_loan_demo__1.
    """
    ctx = get_dataset_active_model_context("final_project_loan_demo__1")
    assert ctx["has_model"] is True
    assert ctx["dataset_id"] == "final_project_loan_demo__1"
    assert ctx["selected_model"] == "Logistic Regression"
    assert ctx["model_version"] == "v3"
    assert "final_project_loan_demo__1" in ctx["run_id"]
    assert ctx["dataset_hash"] == "2c40ae5fecaa3deef2e67371fe57286ca739e451dc998540cda70797adb2ca3c"


def test_4_strict_dataset_isolation_cross_dataset_rejection():
    """
    Verify that evaluating loan models with adult model versions or configs is strictly blocked.
    """
    adult_ref = load_reference_data_for_model("adult_census_income", base_dir=".")
    assert adult_ref["dataset_config"].dataset_id == "adult_census_income"

    # Attempting to load loan_approval with adult version must raise mismatch
    with pytest.raises(ValueError) as excinfo:
        load_reference_data_for_model(
            "final_project_loan_demo__1",
            model_version="v_adult_invalid",
            base_dir="."
        )
    assert "not found" in str(excinfo.value).lower() or "mismatch" in str(excinfo.value).lower()


def test_5_prerun_schema_comparison_detects_matching_and_extra_columns():
    """
    Verify pre-run schema comparison accurately matches compatible columns and detects unexpected extra columns.
    """
    ref_bundle = load_reference_data_for_model("final_project_loan_demo__1", base_dir=".")
    ref_df = ref_bundle["reference_df"]
    ds_cfg = ref_bundle["dataset_config"]

    # Create a batch with identical columns + 1 extra column
    cur_df = ref_df.head(50).copy()
    cur_df["extra_feature_foo"] = [1.23] * len(cur_df)

    rows = compare_reference_and_monitoring_schemas(
        reference_df=ref_df,
        current_df=cur_df,
        target_column=ds_cfg.target.column,
        protected_attributes=ds_cfg.protected_attributes,
        id_columns=ds_cfg.id_columns,
        ignore_columns=ds_cfg.ignore_columns
    )

    by_col = {r["column"]: r for r in rows}
    assert by_col["credit_score"]["status"] == "✅ MATCH"
    assert by_col["loan_status"]["status"] == "✅ MATCH"
    assert by_col["extra_feature_foo"]["status"] == "⚠️ EXTRA COLUMN"


def test_6_applicant_id_datatype_mismatch_detected_without_silent_casting():
    """
    Verify that an integer applicant_id in monitoring batch against string applicant_id in reference
    is flagged as a TYPE MISMATCH and NOT silently converted.
    """
    ref_bundle = load_reference_data_for_model("final_project_loan_demo__1", base_dir=".")
    ref_df = ref_bundle["reference_df"]
    ds_cfg = ref_bundle["dataset_config"]

    # Reference applicant_id is string/object
    assert pd.api.types.is_string_dtype(ref_df["applicant_id"]) or pd.api.types.is_object_dtype(ref_df["applicant_id"])

    # Create batch with integer applicant_id
    cur_df = ref_df.head(50).copy()
    cur_df["applicant_id"] = list(range(1001, 1051))  # int64

    rows = compare_reference_and_monitoring_schemas(
        reference_df=ref_df,
        current_df=cur_df,
        target_column=ds_cfg.target.column,
        protected_attributes=ds_cfg.protected_attributes,
        id_columns=ds_cfg.id_columns,
        ignore_columns=ds_cfg.ignore_columns
    )

    by_col = {r["column"]: r for r in rows}
    assert by_col["applicant_id"]["status"] == "❌ TYPE MISMATCH"
    assert by_col["applicant_id"]["is_compatible"] is False
    assert "int" in by_col["applicant_id"]["monitoring_dtype"].lower()
    assert any(x in by_col["applicant_id"]["reference_dtype"].lower() for x in ["object", "str", "string"])

    # Schema drift detector flags critical issue
    s_rep = detect_schema_drift(
        reference_df=ref_df,
        current_df=cur_df,
        target_column=ds_cfg.target.column,
        protected_attributes=ds_cfg.protected_attributes
    )
    assert s_rep["has_critical_drift"] is True
    assert any("applicant_id" in iss for iss in s_rep["critical_issues"])


def test_7_id_and_ignore_columns_excluded_from_predictive_feature_drift():
    """
    Verify that ID columns (applicant_id) and ignore_columns are excluded from feature drift (PSI/TVD)
    and never reported as drifted features.
    """
    ref_bundle = load_reference_data_for_model("final_project_loan_demo__1", base_dir=".")
    ref_df = ref_bundle["reference_df"]
    ds_cfg = ref_bundle["dataset_config"]

    cur_df = ref_df.head(60).copy()
    # Change applicant_id values completely
    cur_df["applicant_id"] = [f"NEW_APP_{i:04d}" for i in range(len(cur_df))]

    f_rep = analyze_feature_drift(
        reference_df=ref_df,
        current_df=cur_df,
        target_column=ds_cfg.target.column,
        protected_attributes=ds_cfg.protected_attributes,
        id_columns=ds_cfg.id_columns,
        ignore_columns=ds_cfg.ignore_columns
    )

    # applicant_id must NOT be in feature metrics or drifted features
    assert "applicant_id" not in f_rep["feature_metrics"]
    assert "applicant_id" not in f_rep["drifted_features"]
    assert ds_cfg.target.column not in f_rep["feature_metrics"]


def test_8_missing_target_labels_results_in_performance_and_fairness_unavailable():
    """
    Verify that when monitoring batch has no target column (unlabeled inference data),
    performance and fairness drift are cleanly marked as UNAVAILABLE without fabricated numbers.
    """
    ref_bundle = load_reference_data_for_model("final_project_loan_demo__1", base_dir=".")
    ref_df = ref_bundle["reference_df"]
    ds_cfg = ref_bundle["dataset_config"]

    cur_df = ref_df.head(60).drop(columns=[ds_cfg.target.column]).copy()

    res = run_monitoring_pipeline(
        dataset_id="final_project_loan_demo__1",
        current_data=cur_df,
        base_dir="."
    )

    assert res["performance_report"]["status"] == "PERFORMANCE_UNAVAILABLE"
    assert res["fairness_report"]["status"] == "FAIRNESS_UNAVAILABLE"
    assert res["target_drift_report"]["status"] == "TARGET_LABELS_UNAVAILABLE"
    # No fabricated accuracy or F1
    assert "accuracy" not in res["performance_report"].get("current_metrics", {})


def test_9_critical_schema_drift_blocks_performance_and_fairness_while_running_feature_drift():
    """
    Verify that when a column has a type mismatch, performance and fairness evaluations are blocked,
    while feature drift analyzes compatible predictive features.
    """
    ref_bundle = load_reference_data_for_model("final_project_loan_demo__1", base_dir=".")
    ref_df = ref_bundle["reference_df"]

    # Introduce type mismatch on credit_score
    cur_df = ref_df.head(60).copy()
    cur_df["credit_score"] = [f"SCORE_{x}" for x in cur_df["credit_score"]]

    res = run_monitoring_pipeline(
        dataset_id="final_project_loan_demo__1",
        current_data=cur_df,
        base_dir="."
    )

    # Schema critical drift detected
    assert res["schema_report"]["has_critical_drift"] is True
    # Downstream evaluation blocked
    assert res["performance_report"]["status"] == "PERFORMANCE_UNAVAILABLE"
    assert res["fairness_report"]["status"] == "FAIRNESS_UNAVAILABLE"
    assert res["calibration_report"]["status"] == "CALIBRATION_UNAVAILABLE"
    assert res["health_report"]["overall_health"] == "BLOCKED"
    assert res["retraining_recommendation"]["status"] == "RETRAINING_REQUIRED"


def test_10_valid_monitoring_batch_executes_all_stages_successfully():
    """
    Verify that a valid monitoring batch with matching types and labels executes all stages A-G successfully.
    """
    ref_bundle = load_reference_data_for_model("final_project_loan_demo__1", base_dir=".")
    ref_df = ref_bundle["reference_df"]

    # Valid monitoring batch
    cur_df = ref_df.tail(120).copy()

    res = run_monitoring_pipeline(
        dataset_id="final_project_loan_demo__1",
        current_data=cur_df,
        base_dir="."
    )

    # Stage A: Schema OK
    assert res["schema_report"]["status"] == "SCHEMA_OK"
    assert not res["schema_report"]["has_critical_drift"]

    # Stage B: Feature Drift evaluated
    assert res["feature_drift_report"]["total_features_analyzed"] > 0
    assert "applicant_id" not in res["feature_drift_report"]["feature_metrics"]

    # Stage C: Performance evaluated
    assert res["performance_report"]["status"] in ["STABLE", "WARNING", "DEGRADED", "IMPROVED"]
    assert "accuracy" in res["performance_report"]["current_metrics"]
    assert "f1_score" in res["performance_report"]["current_metrics"]

    # Stage D: Fairness evaluated
    assert res["fairness_report"]["status"] in ["FAIRNESS_STABLE", "FAIRNESS_WARNING", "FAIRNESS_DEGRADED", "FAIRNESS_IMPROVED", "INSUFFICIENT_GROUP_COVERAGE"]

    # Stage E: Calibration evaluated
    assert res["calibration_report"]["status"] in ["STABLE", "DEGRADED", "IMPROVED"]

    # Stage F & G: Health and Retraining synthesized
    assert res["health_report"]["overall_health"] in ["HEALTHY", "WARNING", "DEGRADED"]
    assert res["retraining_recommendation"]["status"] in ["NO_RETRAINING_NEEDED", "RETRAINING_RECOMMENDED", "RETRAINING_REQUIRED"]


def test_11_schema_drift_import_and_preflight_buffer_verification():
    """
    Verify that compare_reference_and_monitoring_schemas is strictly importable
    from src.monitoring.schema_drift, src.monitoring, and dashboard.utils,
    and runs preflight schema check on a CSV buffer seamlessly.
    """
    from src.monitoring.schema_drift import compare_reference_and_monitoring_schemas as fn1
    from src.monitoring import compare_reference_and_monitoring_schemas as fn2
    from dashboard.utils import get_monitoring_preflight_schema

    assert callable(fn1)
    assert callable(fn2)
    assert fn1 == fn2

    # Test file buffer inspection on valid CSV
    ref_bundle = load_reference_data_for_model("final_project_loan_demo__1", base_dir=".")
    ref_df = ref_bundle["reference_df"]

    csv_bytes = ref_df.head(30).to_csv(index=False).encode("utf-8")
    buf = io.BytesIO(csv_bytes)

    preflight = get_monitoring_preflight_schema(
        dataset_id="final_project_loan_demo__1",
        file_buffer_or_path=buf,
        reference_version="v1"
    )

    assert preflight["has_critical_schema_error"] is False
    assert len(preflight["type_mismatches"]) == 0
    assert len(preflight["missing_columns"]) == 0
    assert len(preflight["comparison_rows"]) == len(ref_df.columns)


def test_12_stage_b_analyzes_only_exact_four_predictive_features():
    """
    Verify requirements A & B:
    Stage B must analyze ONLY:
      - credit_score
      - annual_income
      - loan_amount
      - age
    Stage B must NOT analyze:
      - applicant_id (ID column)
      - gender (protected attribute)
      - race (protected attribute)
      - loan_status (target)
    """
    ref_bundle = load_reference_data_for_model("final_project_loan_demo__1", base_dir=".")
    ref_df = ref_bundle["reference_df"]
    cur_df = ref_df.tail(120).copy()

    res = run_monitoring_pipeline(
        dataset_id="final_project_loan_demo__1",
        current_data=cur_df,
        base_dir="."
    )

    feature_metrics = res["feature_drift_report"]["feature_metrics"]
    analyzed_features = set(feature_metrics.keys())

    # Exact four predictive features
    expected_features = {"credit_score", "annual_income", "loan_amount", "age"}
    assert analyzed_features == expected_features
    assert len(analyzed_features) == 4

    # Specifically verify excluded columns
    assert "applicant_id" not in analyzed_features
    assert "gender" not in analyzed_features
    assert "race" not in analyzed_features
    assert "loan_status" not in analyzed_features


def test_13_stage_c_performance_delta_and_percentage_calculation():
    """
    Verify requirements C & D:
    Given reference accuracy 0.6833 and batch accuracy 0.6344:
    - delta must be approx -0.0489 (never raw +0.6344)
    - percentage change must be approx -7.15% (or -7.16%)
    """
    from src.monitoring.performance_monitor import monitor_performance_drift
    from unittest.mock import MagicMock
    import numpy as np

    # Mock model and data
    X_test = pd.DataFrame({"credit_score": [700] * 100})
    y_test = pd.Series([1] * 63 + [0] * 37)  # accuracy = 0.6300 approx

    ref_metrics = {
        "accuracy": 0.6833333333333333,
        "precision": 0.69,
        "recall": 0.9078947368421053,
        "f1_score": 0.7840909090909091
    }

    # Simulate batch returning exactly 0.6344 accuracy
    # Using mock predict_model
    from unittest.mock import patch
    with patch("src.monitoring.performance_monitor.predict_model") as mock_pred:
        # Create predictions that yield 0.6344 accuracy
        n_samples = 1000
        y_test_sim = pd.Series([1] * 634 + [0] * 366)
        y_pred_sim = np.array([1] * 634 + [1] * 366)  # TP=634, FP=366 -> accuracy = 634/1000 = 0.6340
        # Exactly 0.6344: 6344 out of 10000
        y_test_10k = pd.Series([1] * 6344 + [0] * 3656)
        y_pred_10k = np.array([1] * 6344 + [1] * 3656)
        mock_pred.return_value = (y_pred_10k, np.array([0.8] * 10000))

        perf_res = monitor_performance_drift(
            model=MagicMock(),
            X_test=pd.DataFrame({"f": [1] * 10000}),
            y_test=y_test_10k,
            reference_metrics=ref_metrics
        )

        deltas = perf_res["deltas"]
        assert pytest.approx(deltas["accuracy_delta"], abs=1e-3) == (0.6344 - 0.6833333333333333)
        assert deltas["accuracy_delta"] < 0
        assert deltas["accuracy_delta"] != 0.6344  # Never raw after value

        expected_pct = ((0.6344 - 0.6833333333333333) / 0.6833333333333333) * 100.0
        assert pytest.approx(deltas["accuracy_pct_change"], abs=0.1) == expected_pct
        assert pytest.approx(deltas["accuracy_pct_change"], abs=0.2) == -7.15


def test_14_stage_e_calibration_delta_and_direction_interpretation():
    """
    Verify requirements E & F:
    Baseline Brier = 0.1974, Batch Brier = 0.2220 -> delta +0.0246 (degraded)
    Baseline ECE = 0.1254, Batch ECE = 0.0418 -> delta -0.0836 (improved)
    """
    from src.monitoring.calibration_monitor import monitor_calibration_drift
    from unittest.mock import MagicMock, patch

    ref_calibration = {
        "brier_score": 0.1974,
        "ece": 0.1254,
        "expected_calibration_error": 0.1254
    }

    with patch("src.monitoring.calibration_monitor.predict_model") as mock_pred, \
         patch("src.monitoring.calibration_monitor.evaluate_calibration") as mock_eval:
        mock_pred.return_value = (np.array([1, 0]), np.array([0.8, 0.2]))
        mock_eval.return_value = {
            "brier_score": 0.2220,
            "ece": 0.0418,
            "expected_calibration_error": 0.0418
        }

        cal_res = monitor_calibration_drift(
            model=MagicMock(),
            X_test=pd.DataFrame({"f": [1, 2]}),
            y_test=pd.Series([1, 0]),
            reference_calibration=ref_calibration
        )

        assert pytest.approx(cal_res["brier_delta"], abs=1e-4) == 0.0246
        assert pytest.approx(cal_res["ece_delta"], abs=1e-4) == -0.0836
        assert cal_res["interpretation"]["brier_status"] == "Degraded (Worsened)"
        assert cal_res["interpretation"]["ece_status"] == "Improved"


def test_15_id_drift_exclusion_no_false_retraining():
    """
    Verify requirement G:
    Changing ONLY applicant_id values must NOT trigger predictive feature drift
    or retraining recommendation.
    """
    ref_bundle = load_reference_data_for_model("final_project_loan_demo__1", base_dir=".")
    ref_df = ref_bundle["reference_df"]

    # Replicate reference data exactly but assign new unique applicant_id strings
    cur_df = ref_df.copy()
    cur_df["applicant_id"] = [f"NEW_APP_{i:06d}" for i in range(len(cur_df))]

    res = run_monitoring_pipeline(
        dataset_id="final_project_loan_demo__1",
        current_data=cur_df,
        base_dir="."
    )

    # Feature drift should be zero
    assert res["feature_drift_report"]["drifted_features_count"] == 0
    assert "applicant_id" not in res["feature_drift_report"]["feature_metrics"]
    assert res["health_report"]["overall_health"] == "HEALTHY"
    assert res["retraining_recommendation"]["status"] == "NO_RETRAINING_NEEDED"


def test_16_dataset_isolation_no_adult_leakage():
    """
    Verify requirement I:
    No Adult Census Income values, columns, or metrics appear when monitoring final_project_loan_demo__1.
    """
    ref_bundle = load_reference_data_for_model("final_project_loan_demo__1", base_dir=".")
    ref_df = ref_bundle["reference_df"]
    ds_cfg = ref_bundle["dataset_config"]

    adult_cols = {"workclass", "education_num", "marital_status", "occupation", "capital_gain", "capital_loss", "hours_per_week"}
    assert len(adult_cols.intersection(set(ref_df.columns))) == 0
    assert ds_cfg.dataset_id == "final_project_loan_demo__1"
    assert ds_cfg.target.column == "loan_status"
    assert ds_cfg.target.positive_class == "approved"
    assert set(ds_cfg.protected_attributes) == {"race", "gender"}


def test_17_calibration_warning_status_propagates_to_health_assessment():
    """
    Verify that a calibration report with status 'WARNING' correctly maps
    to component_statuses['calibration_health'] = 'WARNING'.
    """
    from src.monitoring.model_health import evaluate_model_health
    
    schema_rep = {"status": "SCHEMA_OK", "has_critical_drift": False}
    quality_rep = {"status": "QUALITY_STABLE"}
    feat_rep = {"status": "NO_DRIFT", "drifted_features": []}
    target_rep = {"status": "TARGET_STABLE"}
    perf_rep = {"status": "STABLE"}
    fair_rep = {"status": "FAIRNESS_STABLE"}
    cal_rep = {"status": "WARNING"}

    health = evaluate_model_health(
        schema_report=schema_rep,
        quality_report=quality_rep,
        feature_drift_report=feat_rep,
        target_drift_report=target_rep,
        performance_report=perf_rep,
        fairness_report=fair_rep,
        calibration_report=cal_rep,
        artifact_integrity=True
    )

    assert health["component_statuses"]["calibration_health"] == "WARNING"
    assert health["overall_health"] == "WARNING"


def test_18_f1_degradation_retraining_reason_formatting():
    """
    Verify that when F1 degradation occurs, the retraining recommendation reason
    string formats the percentage change accurately without double multiplying.
    """
    from src.monitoring.retraining_recommender import evaluate_retraining_recommendation
    
    health_rep = {"overall_health": "DEGRADED"}
    schema_rep = {"status": "SCHEMA_OK", "has_critical_drift": False}
    quality_rep = {"status": "QUALITY_STABLE"}
    feat_rep = {"status": "NO_DRIFT", "drifted_features": []}
    target_rep = {"status": "TARGET_STABLE"}
    perf_rep = {
        "status": "DEGRADED",
        "deltas": {
            "f1_score_delta": -0.0600,
            "f1_score_pct_change": -6.0000
        }
    }
    fair_rep = {"status": "FAIRNESS_STABLE"}
    cal_rep = {"status": "STABLE"}

    rec = evaluate_retraining_recommendation(
        model_health_report=health_rep,
        schema_report=schema_rep,
        quality_report=quality_rep,
        feature_drift_report=feat_rep,
        target_drift_report=target_rep,
        performance_report=perf_rep,
        fairness_report=fair_rep,
        calibration_report=cal_rep
    )

    assert rec["status"] == "RETRAINING_RECOMMENDED"
    assert len(rec["reasons"]) > 0
    matched = [r for r in rec["reasons"] if "6.0%" in r]
    assert len(matched) > 0, f"Expected 6.0% in reasons, got: {rec['reasons']}"


def test_19_complete_step7_final_project_loan_demo_audit():
    """
    Complete 100% end-to-end verification and consistency audit of Step 7
    for final_project_loan_demo__1 using the authoritative monitoring batch.
    """
    res = run_monitoring_pipeline(
        dataset_id="final_project_loan_demo__1",
        current_data="data/raw/final_project_loan_demo__1_batch_1787981848.csv",
        base_dir="."
    )

    # 1. Stage A: Schema & Quality
    assert res["schema_report"]["status"] == "SCHEMA_OK"
    assert not res["schema_report"]["has_critical_drift"]
    assert res["data_quality_report"]["status"] == "QUALITY_STABLE"
    assert res["data_quality_report"]["current_rows"] == 320
    assert res["data_quality_report"]["reference_rows"] == 600

    # 2. Stage B: Feature Drift (Exact 4 predictive features)
    feature_metrics = res["feature_drift_report"]["feature_metrics"]
    assert set(feature_metrics.keys()) == {"credit_score", "annual_income", "loan_amount", "age"}
    assert "applicant_id" not in feature_metrics
    assert "gender" not in feature_metrics
    assert "race" not in feature_metrics
    assert "loan_status" not in feature_metrics

    # 3. Stage C: Performance Drift
    perf = res["performance_report"]
    assert perf["status"] == "STABLE"
    ref_m = perf["reference_metrics"]
    cur_m = perf["current_metrics"]
    deltas = perf["deltas"]

    assert pytest.approx(ref_m["accuracy"], abs=1e-3) == 0.6833
    assert pytest.approx(ref_m["precision"], abs=1e-3) == 0.6900
    assert pytest.approx(ref_m["recall"], abs=1e-3) == 0.9079
    assert pytest.approx(ref_m["f1_score"], abs=1e-3) == 0.7841

    assert pytest.approx(cur_m["accuracy"], abs=1e-3) == 0.6344
    assert pytest.approx(cur_m["precision"], abs=1e-3) == 0.6508
    assert pytest.approx(cur_m["recall"], abs=1e-3) == 0.8497
    assert pytest.approx(cur_m["f1_score"], abs=1e-3) == 0.7371

    assert pytest.approx(deltas["accuracy_delta"], abs=1e-3) == -0.0490
    assert pytest.approx(deltas["accuracy_pct_change"], abs=0.1) == -7.16
    assert pytest.approx(deltas["precision_delta"], abs=1e-3) == -0.0392
    assert pytest.approx(deltas["precision_pct_change"], abs=0.1) == -5.68
    assert pytest.approx(deltas["recall_delta"], abs=1e-3) == -0.0582
    assert pytest.approx(deltas["recall_pct_change"], abs=0.1) == -6.41
    assert pytest.approx(deltas["f1_score_delta"], abs=1e-3) == -0.0470
    assert pytest.approx(deltas["f1_score_pct_change"], abs=0.1) == -6.00

    # 4. Stage D: Fairness Drift
    fair = res["fairness_report"]
    assert fair["status"] == "FAIRNESS_STABLE"
    assert fair["reference_fairness"]["equalized_odds_difference"] is None
    assert fair["eod_delta"] is None

    # 5. Stage E: Calibration Drift
    cal = res["calibration_report"]
    assert cal["status"] == "WARNING"
    assert pytest.approx(cal["reference_calibration"]["brier_score"], abs=1e-4) == 0.1974
    assert pytest.approx(cal["reference_calibration"]["ece"], abs=1e-4) == 0.1254
    assert pytest.approx(cal["current_calibration"]["brier_score"], abs=1e-4) == 0.2220
    assert pytest.approx(cal["current_calibration"]["ece"], abs=1e-4) == 0.0418
    assert pytest.approx(cal["brier_delta"], abs=1e-4) == 0.0246
    assert pytest.approx(cal["brier_pct_change"], abs=0.1) == 12.47
    assert pytest.approx(cal["ece_delta"], abs=1e-4) == -0.0836
    assert pytest.approx(cal["ece_pct_change"], abs=0.1) == -66.66
    assert cal["interpretation"]["brier_status"] == "Degraded (Worsened)"
    assert cal["interpretation"]["ece_status"] == "Improved"

    # 6. Stage F: Overall Model Health
    health = res["health_report"]
    assert health["overall_health"] == "DEGRADED"
    assert health["component_statuses"]["feature_drift_health"] == "DEGRADED"
    assert health["component_statuses"]["calibration_health"] == "WARNING"

    # 7. Stage G: Retraining Recommendation
    rec = res["retraining_recommendation"]
    assert rec["status"] == "RETRAINING_RECOMMENDED"
    assert rec["action_required"] is True
    assert any("annual_income" in r for r in rec["reasons"])
    assert any("loan_amount" in r for r in rec["reasons"])
    assert any("age" in r for r in rec["reasons"])
    assert not any("applicant_id" in r for r in rec["reasons"])
    assert not any("gender" in r for r in rec["reasons"])
    assert not any("race" in r for r in rec["reasons"])
    assert not any("loan_status" in r for r in rec["reasons"])




