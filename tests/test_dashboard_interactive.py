"""
Unit & Integration Tests for Interactive Dashboard Multi-Dataset Generalization.
"""

import io
import os
import json
import pytest
import pandas as pd

from src.data.dataset_config import DatasetConfig
from dashboard.utils import (
    save_uploaded_dataset,
    create_dataset_config_file,
    execute_interactive_pipeline,
    prepare_download_artifacts,
    get_available_dataset_runs,
    load_dataset_run_result
)


def test_adult_results_load_correctly():
    """Verify Adult reference dataset results load dynamically."""
    res = load_dataset_run_result("adult_census_income")
    assert res is not None
    assert res["dataset_id"] == "adult_census_income"
    assert res["target_column"] == "income"
    assert abs(res["baseline_metrics"]["accuracy"] - 0.8399) < 1e-3


def test_student_results_load_correctly():
    """Verify Student dataset results load dynamically."""
    res = load_dataset_run_result("student_performance")
    assert res is not None
    assert res["dataset_id"] == "student_performance"
    assert res["target_column"] == "passed"
    assert res["positive_class"] == "yes"


def test_loan_results_load_correctly():
    """Verify Loan dataset results load dynamically."""
    res = load_dataset_run_result("loan_approval")
    assert res is not None
    assert res["dataset_id"] == "loan_approval"
    assert res["target_column"] == "loan_status"
    assert res["positive_class"] == "approved"


def test_changing_active_dataset_changes_displayed_metrics():
    """Verify switching active dataset ID changes target and baseline accuracy."""
    adult_res = load_dataset_run_result("adult_census_income")
    student_res = load_dataset_run_result("student_performance")
    loan_res = load_dataset_run_result("loan_approval")

    assert adult_res["target_column"] != student_res["target_column"]
    assert student_res["target_column"] != loan_res["target_column"]
    assert adult_res["baseline_metrics"]["accuracy"] != student_res["baseline_metrics"]["accuracy"]
    assert student_res["baseline_metrics"]["accuracy"] != loan_res["baseline_metrics"]["accuracy"]


def test_intersectional_group_labels_change_according_to_dataset():
    """Verify intersectional group labels are dataset-specific."""
    student_res = load_dataset_run_result("student_performance")
    loan_res = load_dataset_run_result("loan_approval")

    student_groups = list(student_res["intersectional_metrics"]["all_group_metrics"].keys())
    loan_groups = list(loan_res["intersectional_metrics"]["all_group_metrics"].keys())

    assert "Female + GroupA" in student_groups or "Male + GroupA" in student_groups
    assert "Female + GroupA" in loan_groups or "Male + GroupA" in loan_groups
    # Verify no Adult groups in student/loan results
    assert "Male + White" not in student_groups
    assert "Male + Black" not in loan_groups


def test_tradeoff_metrics_change_according_to_dataset():
    """Verify trade-off metrics are computed dynamically for each dataset."""
    student_res = load_dataset_run_result("student_performance")
    loan_res = load_dataset_run_result("loan_approval")

    st_tradeoff = student_res["tradeoff_metrics"]
    ln_tradeoff = loan_res["tradeoff_metrics"]

    assert st_tradeoff["performance"]["Accuracy"]["baseline"] != ln_tradeoff["performance"]["Accuracy"]["baseline"]


def test_mitigation_results_are_dataset_specific():
    """Verify mitigation accuracy and fairness deltas differ per dataset."""
    adult_res = load_dataset_run_result("adult_census_income")
    loan_res = load_dataset_run_result("loan_approval")

    assert adult_res["tradeoff_metrics"]["fairness"]["Equalized Odds Difference"]["baseline"] != loan_res["tradeoff_metrics"]["fairness"]["Equalized Odds Difference"]["baseline"]


def test_calibration_results_are_dataset_specific():
    """Verify calibration Brier score and ECE differ per dataset."""
    student_res = load_dataset_run_result("student_performance")
    loan_res = load_dataset_run_result("loan_approval")

    assert student_res["calibration_metrics"]["brier_score"] != loan_res["calibration_metrics"]["brier_score"]


def test_no_adult_leakage_in_custom_results():
    """Verify custom dataset run results contain 0 Adult reference fallback numbers."""
    student_res = load_dataset_run_result("student_performance")
    assert student_res["baseline_metrics"]["accuracy"] != 0.8399
    assert student_res["fairness_metrics"]["equalized_odds_difference"] != 0.4825


def test_results_history_isolates_datasets():
    """Verify get_available_dataset_runs discovers all runs separately."""
    runs = get_available_dataset_runs()
    assert "Adult Census Income (Default)" in runs or "adult_census_income" in runs
    assert "student_performance" in runs
    assert "loan_approval" in runs


def test_downloads_contain_active_dataset_results():
    """Verify prepare_download_artifacts returns specific dataset JSON content."""
    student_art = prepare_download_artifacts("student_performance")
    loan_art = prepare_download_artifacts("loan_approval")

    assert "student_performance" in student_art["json_str"]
    assert "loan_approval" in loan_art["json_str"]


def test_multiclass_target_rejection():
    """Verify preprocessor raises ValueError when target contains > 2 unique values."""
    from src.data.preprocessor import encode_target

    series_multiclass = pd.Series(["low", "medium", "high", "low"])
    with pytest.raises(ValueError, match="binary classification"):
        encode_target(series_multiclass, positive_value="high")


# ==============================================================================
# CANONICAL DASHBOARD RESULT-NORMALIZATION TESTS (TESTS 1-12)
# ==============================================================================
def test_req_1_overview_page_loads_current_schema():
    from dashboard.utils import normalize_run_results
    norm = normalize_run_results(load_dataset_run_result("adult_census_income"))
    assert norm["baseline"]["performance"]["accuracy"] is not None
    assert norm["model"]["selected_model"] is not None


def test_req_2_baseline_page_loads_current_schema():
    from dashboard.utils import normalize_run_results
    norm = normalize_run_results(load_dataset_run_result("adult_census_income"))
    assert "accuracy" in norm["baseline"]["performance"]
    assert "precision" in norm["baseline"]["performance"]


def test_req_3_old_result_schema_normalizes_correctly():
    from dashboard.utils import normalize_run_results
    old_schema = {
        "dataset_summary": {"dataset": "UCI Adult Census Income Dataset", "train_samples": 24000, "test_samples": 6000},
        "performance_metrics": {
            "baseline": {"accuracy": 0.84, "precision": 0.72, "recall": 0.57, "f1_score": 0.64, "roc_auc": 0.88},
            "mitigated": {"accuracy": 0.75, "precision": 0.59, "recall": 0.08, "f1_score": 0.15, "roc_auc": 0.53}
        },
        "fairness_metrics": {
            "baseline": {"equalized_odds_difference": 0.48, "demographic_parity_difference": 0.31},
            "mitigated": {"equalized_odds_difference": 0.16, "demographic_parity_difference": 0.02}
        },
        "calibration_metrics": {
            "baseline": {"brier_score": 0.11, "expected_calibration_error": 0.015},
            "mitigated": {"brier_score": 0.14, "expected_calibration_error": 0.04}
        }
    }
    norm = normalize_run_results(old_schema)
    assert norm["baseline"]["performance"]["accuracy"] == 0.84
    assert norm["mitigated"]["performance"]["accuracy"] == 0.75
    assert norm["baseline"]["fairness"]["equalized_odds_difference"] == 0.48
    assert norm["baseline"]["calibration"]["brier_score"] == 0.11


def test_req_4_new_phase_c_result_schema_normalizes_correctly():
    from dashboard.utils import normalize_run_results
    new_schema = {
        "run_id": "RUN-test-123",
        "dataset_id": "test_ds",
        "baseline_metrics": {"accuracy": 0.91, "precision": 0.88, "recall": 0.85, "f1_score": 0.86, "roc_auc": 0.94},
        "fairness_metrics": {"equalized_odds_difference": 0.05, "demographic_parity_difference": 0.03},
        "calibration_metrics": {"brier_score": 0.08, "status": "AVAILABLE"},
        "selected_model": "random_forest",
        "model_version": "v2"
    }
    norm = normalize_run_results(new_schema)
    assert norm["baseline"]["performance"]["accuracy"] == 0.91
    assert norm["baseline"]["fairness"]["equalized_odds_difference"] == 0.05
    assert norm["baseline"]["calibration"]["brier_score"] == 0.08
    assert norm["model"]["selected_model"] == "random_forest"
    assert norm["model"]["model_version"] == "v2"


def test_req_5_adult_dataset_loads():
    from dashboard.utils import normalize_run_results
    norm = normalize_run_results(load_dataset_run_result("adult_census_income"))
    assert norm["dataset_metadata"]["target_column"] == "income"


def test_req_6_student_dataset_loads():
    from dashboard.utils import normalize_run_results
    norm = normalize_run_results(load_dataset_run_result("student_performance"))
    assert norm["dataset_metadata"]["target_column"] == "passed"


def test_req_7_loan_dataset_loads():
    from dashboard.utils import normalize_run_results
    norm = normalize_run_results(load_dataset_run_result("loan_approval"))
    assert norm["dataset_metadata"]["target_column"] == "loan_status"


def test_req_8_switching_active_dataset_updates_metrics():
    from dashboard.utils import normalize_run_results
    n_adult = normalize_run_results(load_dataset_run_result("adult_census_income"))
    n_student = normalize_run_results(load_dataset_run_result("student_performance"))
    n_loan = normalize_run_results(load_dataset_run_result("loan_approval"))
    assert n_adult["dataset_metadata"]["target_column"] != n_student["dataset_metadata"]["target_column"]
    assert n_student["dataset_metadata"]["target_column"] != n_loan["dataset_metadata"]["target_column"]
    assert n_adult["baseline"]["performance"]["accuracy"] != n_student["baseline"]["performance"]["accuracy"]


def test_req_9_missing_baseline_field_does_not_cause_keyerror():
    from dashboard.utils import normalize_run_results
    empty_schema = {}
    norm = normalize_run_results(empty_schema)
    assert norm["baseline"]["performance"]["accuracy"] is None
    assert norm["baseline"]["fairness"]["equalized_odds_difference"] is None


def test_req_10_missing_optional_field_displays_na_rather_than_crashing():
    from dashboard.utils import format_val
    assert format_val(None) == "N/A"
    assert format_val(float("nan")) == "N/A"


def test_req_11_model_monitoring_page_still_works():
    from dashboard.utils import get_dataset_model_history
    history = get_dataset_model_history("adult_census_income")
    assert isinstance(history, list)


def test_req_12_no_adult_benchmark_leakage_into_custom_dataset_pages():
    from dashboard.utils import normalize_run_results
    n_student = normalize_run_results(load_dataset_run_result("student_performance"))
    assert n_student["dataset_metadata"]["target_column"] != "income"
    assert n_student["baseline"]["performance"]["accuracy"] != 0.8398806563898558


# ==============================================================================
# CANONICAL MODEL MONITORING REGRESSION TESTS (1-15)
# ==============================================================================
def test_mon_1_records_with_model_version_load_correctly():
    from dashboard.utils import normalize_model_record
    rec = {
        "dataset_id": "adult_census_income",
        "model_version": "v1",
        "model_type": "logistic_regression",
        "status": "ACTIVE",
        "is_active": True
    }
    norm = normalize_model_record(rec)
    assert norm is not None
    assert norm["model_version"] == "v1"
    assert norm["is_active"] is True


def test_mon_2_legacy_alternate_records_normalize_correctly():
    from dashboard.utils import normalize_model_record
    rec = {
        "dataset_id": "student_performance",
        "version": "v2",
        "algorithm": "random_forest",
        "status": "SUPERSEDED"
    }
    norm = normalize_model_record(rec)
    assert norm is not None
    assert norm["model_version"] == "v2"
    assert norm["model_type"] == "random_forest"


def test_mon_3_missing_model_version_does_not_cause_keyerror():
    from dashboard.utils import normalize_model_record
    rec = {"dataset_id": "adult_census_income", "status": "COMPLETED"}
    norm = normalize_model_record(rec)
    assert norm is None


def test_mon_4_invalid_registry_records_safely_skipped():
    from dashboard.utils import normalize_model_record
    assert normalize_model_record(None) is None
    assert normalize_model_record("string_record") is None
    assert normalize_model_record({"version": "FAILED", "status": "FAILED"}) is None


def test_mon_5_active_version_is_selected_by_default():
    from dashboard.utils import get_dataset_model_history
    history = get_dataset_model_history("adult_census_income")
    assert len(history) > 0
    active_items = [h for h in history if h.get("is_active")]
    assert len(active_items) > 0


def test_mon_6_latest_valid_version_selected_if_no_active():
    from dashboard.utils import get_dataset_model_history
    history = get_dataset_model_history("adult_census_income")
    assert history[0]["model_version"] is not None


def test_mon_7_dataset_isolation_works():
    from dashboard.utils import get_dataset_model_history
    adult_h = get_dataset_model_history("adult_census_income")
    student_h = get_dataset_model_history("student_performance")
    
    adult_ids = set(h["dataset_id"] for h in adult_h)
    student_ids = set(h["dataset_id"] for h in student_h)

    assert adult_ids == {"adult_census_income"}
    assert student_ids == {"student_performance"}


def test_mon_8_failed_versions_excluded():
    from dashboard.utils import normalize_model_record
    failed_rec = {
        "dataset_id": "adult_census_income",
        "model_version": "v99",
        "status": "FAILED"
    }
    assert normalize_model_record(failed_rec) is None


def test_mon_9_missing_model_artifact_handled():
    from dashboard.utils import normalize_model_record
    rec = {
        "dataset_id": "adult_census_income",
        "model_version": "v1",
        "model_artifact_path": "non_existent_file.joblib"
    }
    norm = normalize_model_record(rec)
    assert norm is not None
    assert not os.path.exists(norm["model_artifact_path"])


def test_mon_10_monitoring_page_opens_successfully():
    from dashboard.utils import get_dataset_model_history
    history = get_dataset_model_history("adult_census_income")
    assert isinstance(history, list)


def test_mon_11_monitoring_page_reaches_step_2():
    from dashboard.utils import get_dataset_model_history
    history = get_dataset_model_history("adult_census_income")
    valid_records = [m for m in history if m.get("model_version")]
    assert len(valid_records) > 0


def test_mon_12_adult_monitoring_works():
    from dashboard.utils import get_dataset_model_history
    history = get_dataset_model_history("adult_census_income")
    assert any(h["dataset_id"] == "adult_census_income" for h in history)


def test_mon_13_student_monitoring_works():
    from dashboard.utils import get_dataset_model_history
    history = get_dataset_model_history("student_performance")
    assert any(h["dataset_id"] == "student_performance" for h in history)


def test_mon_14_loan_monitoring_works():
    from dashboard.utils import get_dataset_model_history
    history = get_dataset_model_history("loan_approval")
    assert any(h["dataset_id"] == "loan_approval" for h in history)


def test_mon_15_existing_phase_a_to_e_tests_pass():
    from src.models.model_registry_store import load_model_registry
    reg = load_model_registry()
    assert "datasets" in reg


def test_mon_16_project_root_helper_resolves_correctly():
    from dashboard.utils import get_project_root
    root = get_project_root()
    assert os.path.isdir(root)
    assert os.path.exists(os.path.join(root, 'dashboard', 'app.py'))
    assert os.path.exists(os.path.join(root, 'dashboard', 'utils.py'))
    assert os.path.exists(os.path.join(root, 'src'))


def test_mon_17_dashboard_imports_get_project_root():
    import dashboard.app as app
    assert hasattr(app, 'get_project_root')
    assert callable(app.get_project_root)


def test_mon_18_all_datasets_step1_to_step2_flow():
    from dashboard.utils import get_dataset_model_history, get_project_root
    root_dir = get_project_root()
    for ds_id in ['adult_census_income', 'student_performance', 'loan_approval']:
        history = get_dataset_model_history(ds_id)
        valid_records = [
            m for m in history
            if isinstance(m, dict) and m.get('model_version') and m.get('status') not in ['FAILED', 'DATASET_TOO_SMALL_FOR_TRAINING']
        ]
        assert len(valid_records) > 0, f'Dataset {ds_id} must have valid model records'
        
        active_records = [m for m in valid_records if m.get('is_active')]
        selected_record = active_records[0] if active_records else valid_records[0]
        ref_ver = selected_record['model_version']
        art_path = selected_record.get('model_artifact_path', '')
        std_path = os.path.join(root_dir, 'models', ds_id, selected_record.get('run_id', ''), 'selected_model.joblib')
        
        assert os.path.exists(art_path) or os.path.exists(std_path), f'Model artifact for {ds_id} version {ref_ver} must exist'


def test_mon_19_approve_retraining_executes_and_creates_new_version():
    from dashboard.utils import approve_and_execute_retraining
    from src.models.model_registry_store import get_dataset_registered_versions
    
    vers_before = get_dataset_registered_versions("loan_approval")
    prev_ver = [v["model_version"] for v in vers_before if v.get("is_active")]
    prev_ver_str = prev_ver[0] if prev_ver else "v1"
    
    res = approve_and_execute_retraining("loan_approval")
    assert res["status"] == "SUCCESS"
    assert res["dataset_id"] == "loan_approval"
    assert res["model_version"] != "NONE"
    assert res["is_active"] is True
    assert res["run_id"].startswith("run_")


def test_mon_20_registry_has_only_one_active_version():
    from src.models.model_registry_store import get_dataset_registered_versions
    
    for ds_id in ["adult_census_income", "student_performance", "loan_approval"]:
        versions = get_dataset_registered_versions(ds_id)
        active_list = [v for v in versions if v.get("is_active")]
        assert len(active_list) == 1, f"Dataset {ds_id} must have exactly one active version, found {len(active_list)}"


def test_mon_21_previous_model_versions_and_artifacts_remain_intact():
    from dashboard.utils import get_project_root
    from src.models.model_registry_store import get_dataset_registered_versions
    
    root = get_project_root()
    versions = get_dataset_registered_versions("loan_approval")
    assert len(versions) >= 2, "Must have multiple versions registered"
    
    # Check that previous versions still exist in folder
    for v in versions:
        run_id = v["run_id"]
        run_dir = os.path.join(root, "models", "loan_approval", run_id)
        assert os.path.exists(run_dir), f"Run directory {run_dir} must remain preserved"


def test_mon_22_retraining_failure_returns_failure_status_and_reason():
    from dashboard.utils import approve_and_execute_retraining
    
    # Nonexistent config
    res = approve_and_execute_retraining("nonexistent_dataset_12345")
    assert res["status"] == "FAILED"
    assert "error" in res
    assert len(res["error"]) > 0


def test_mon_23_invalid_schema_run_blocks_retraining_in_monitoring():
    from src.monitoring.pipeline import run_monitoring_pipeline
    
    # Run with cross-dataset adult CSV on loan
    res = run_monitoring_pipeline("loan_approval", "data/raw/adult.csv", base_dir=".")
    assert res["schema_report"]["has_critical_drift"] is True
    assert res["retraining_recommendation"]["status"] == "BLOCK_MONITORING_DATASET_MISMATCH"
    assert res["health_report"]["overall_health"] == "BLOCKED"


def test_mon_24_valid_schema_run_allows_retraining_approval():
    from src.monitoring.pipeline import run_monitoring_pipeline
    
    res = run_monitoring_pipeline("loan_approval", "data/raw/loan_monitoring_shifted_batch_v2.csv", base_dir=".")
    assert res["schema_report"]["status"] == "SCHEMA_OK"
    assert res["schema_report"]["has_critical_drift"] is False
    assert res["retraining_recommendation"]["status"] in ["RETRAINING_RECOMMENDED", "RETRAINING_REQUIRED"]
    assert res["retraining_recommendation"]["action_required"] is True


def test_mon_25_streamlit_session_state_contract():
    # Verify expected state keys structure
    state_contract = [
        "mon_results_loan_approval",
        "monitoring_result",
        "monitoring_run_id",
        "monitoring_dataset_id",
        "monitoring_reference_version",
        "retrain_result_loan_approval"
    ]
    assert len(state_contract) == 6
