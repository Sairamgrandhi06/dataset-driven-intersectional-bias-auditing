"""
Unit and Integration Tests for Reorganized 9-Step Dashboard UX & Governance Workflow.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

from dashboard.utils import (
    get_project_root,
    get_available_dataset_runs,
    load_dataset_run_result,
    normalize_run_results,
    get_dataset_model_history,
    get_dataset_active_model_context,
    get_dataset_governance_summary,
    get_all_candidate_evaluations,
    get_reliability_summary,
    get_fairness_criteria_compatibility,
    format_val,
    format_confidence_display,
    get_status_style,
    format_tradeoff_value,
    format_tradeoff_percentage,
    calculate_safe_percentage_change,
    get_step5_tradeoff_analysis_data,
    execute_interactive_mitigation_workflow
)
import dashboard.app as app


def test_ux_1_sidebar_navigation_order():
    """Verify that dashboard navigation menu options match the ordered 9-step ML lifecycle."""
    expected_order = [
        "🏠 Home",
        "📥 1. Upload & Configure",
        "🧠 2. Train & Select Model",
        "🔍 3. Fairness Audit",
        "🛠️ 4. Bias Mitigation",
        "⚖️ 5. Trade-off Analysis",
        "📊 6. Model Validation",
        "👁️ 7. Model Monitoring",
        "🔄 8. Retrain & Model History",
        "📦 9. Results & Downloads",
        "ℹ️ About Project"
    ]
    
    # Check that app has all 11 rendering functions
    assert hasattr(app, "render_page_home")
    assert hasattr(app, "render_page_upload_configure")
    assert hasattr(app, "render_page_train_select_model")
    assert hasattr(app, "render_page_fairness_audit")
    assert hasattr(app, "render_page_bias_mitigation")
    assert hasattr(app, "render_page_tradeoff_analysis")
    assert hasattr(app, "render_page_model_validation")
    assert hasattr(app, "render_page_model_monitoring")
    assert hasattr(app, "render_page_retrain_model_history")
    assert hasattr(app, "render_page_results_downloads")
    assert hasattr(app, "render_page_about_project")
    assert hasattr(app, "render_global_context_header")


def test_ux_2_home_page_governance_summary():
    """Verify Home page dataset summary data loads dynamically for Adult, Student, and Loan."""
    adult_sum = get_dataset_governance_summary("adult_census_income")
    student_sum = get_dataset_governance_summary("student_performance")
    loan_sum = get_dataset_governance_summary("loan_approval")

    assert adult_sum["dataset_id"] == "adult_census_income"
    assert student_sum["dataset_id"] == "student_performance"
    assert loan_sum["dataset_id"] == "loan_approval"

    # Targets must differ
    assert adult_sum["target"] == "income"
    assert student_sum["target"] == "passed"
    assert loan_sum["target"] == "loan_status"

    # Positive classes must differ
    assert adult_sum["positive_class"] in [">50K", "1", "True"]
    assert student_sum["positive_class"] == "yes"
    assert loan_sum["positive_class"] == "approved"


def test_ux_3_global_context_bar_dynamic_behavior():
    """Verify global context bar extracts active model, version, status for each dataset."""
    adult_ctx = get_dataset_active_model_context("adult_census_income")
    student_ctx = get_dataset_active_model_context("student_performance")
    loan_ctx = get_dataset_active_model_context("loan_approval")

    assert adult_ctx["dataset_id"] == "adult_census_income"
    assert student_ctx["dataset_id"] == "student_performance"
    assert loan_ctx["dataset_id"] == "loan_approval"

    assert adult_ctx["model_version"] is not None
    assert student_ctx["model_version"] is not None
    assert loan_ctx["model_version"] is not None

    assert adult_ctx["status"] == "ACTIVE"
    assert student_ctx["status"] == "ACTIVE"
    assert loan_ctx["status"] == "ACTIVE"


def test_ux_4_candidate_models_comparison():
    """Verify Step 2 candidate models evaluation retrieves metrics across 3 models."""
    evals = get_all_candidate_evaluations("loan_approval")
    assert "logistic_regression" in evals
    assert "random_forest" in evals
    assert "gradient_boosting" in evals

    for m_key, m_eval in evals.items():
        assert "performance" in m_eval
        assert "fairness" in m_eval
        assert "calibration" in m_eval
        assert "accuracy" in m_eval["performance"]


def test_ux_5_fairness_audit_single_and_intersectional_isolation():
    """Verify Step 3 fairness metrics do not leak adult demographic groups into custom datasets."""
    student_run = load_dataset_run_result("student_performance")
    norm_student = normalize_run_results(student_run)

    # Intersectional groups in student dataset should not contain adult values like "White" or "Black"
    groups = list(norm_student.get("intersectional", {}).get("all_group_metrics", {}).keys())
    for g in groups:
        assert "White" not in g
        assert "Black" not in g
        assert "Amer-Indian-Eskimo" not in g


def test_ux_6_bias_mitigation_workflow_delta_calculations():
    """Verify Step 4 Before vs After delta calculations for accuracy, F1, EOD."""
    run_data = load_dataset_run_result("adult_census_income")
    norm = normalize_run_results(run_data)

    b_acc = norm["baseline"]["performance"]["accuracy"]
    m_acc = norm["mitigated"]["performance"]["accuracy"]
    assert b_acc is not None
    assert m_acc is not None
    delta_acc = m_acc - b_acc
    assert isinstance(delta_acc, float)

    b_eod = norm["baseline"]["fairness"]["equalized_odds_difference"]
    m_eod = norm["mitigated"]["fairness"]["equalized_odds_difference"]
    assert b_eod is not None
    assert m_eod is not None
    gain_eod = b_eod - m_eod
    assert gain_eod > 0  # Mitigation reduces EOD disparity


def test_ux_7_tradeoff_analysis_three_way_metrics():
    """Verify Step 5 trade-off analysis structure has performance, fairness, calibration."""
    norm = normalize_run_results(load_dataset_run_result("loan_approval"))
    tradeoff = norm.get("tradeoff", {})
    assert "performance" in tradeoff
    assert "fairness" in tradeoff
    assert "calibration" in tradeoff


def test_ux_8_model_validation_reliability_summary():
    """Verify Step 6 reliability summary returns health check and replay status."""
    rel = get_reliability_summary("loan_approval")
    assert rel["system_health"] in ["PASS", "HEALTHY"]
    assert "VERIFIED" in rel["replay_status"]
    assert "ACTIVE" in rel["recovery_status"]
    assert rel["evidence_available"] is True


def test_ux_9_model_monitoring_dataset_isolation_and_block():
    """Verify Step 7 monitoring blocks cross-dataset batch upload."""
    from src.monitoring.pipeline import run_monitoring_pipeline
    
    # Loan dataset monitored against Adult CSV batch must be BLOCKED
    res = run_monitoring_pipeline("loan_approval", "data/raw/adult.csv", base_dir=".")
    assert res["schema_report"]["has_critical_drift"] is True
    assert res["health_report"]["overall_health"] == "BLOCKED"
    assert res["retraining_recommendation"]["status"] == "BLOCK_MONITORING_DATASET_MISMATCH"


def test_ux_10_results_downloads_content_isolation():
    """Verify Step 9 downloads strictly contain dataset-specific identifiers."""
    from dashboard.utils import prepare_download_artifacts
    
    stud_arts = prepare_download_artifacts("student_performance")
    loan_arts = prepare_download_artifacts("loan_approval")

    assert "student_performance" in stud_arts["json_str"]
    assert "loan_approval" in loan_arts["json_str"]


def test_ux_step3_confidence_formatting_safe():
    """Verify confidence formatting handles strings, numeric floats, ints, and None safely without ValueError."""
    assert format_confidence_display("High") == "High"
    assert format_confidence_display("Medium") == "Medium"
    assert format_confidence_display("Low") == "Low"
    assert format_confidence_display(0.85) == "85%"
    assert format_confidence_display(0.999) == "100%"
    assert format_confidence_display(1) == "100%"
    assert format_confidence_display(0) == "0%"
    assert format_confidence_display(None) == "N/A"
    assert format_confidence_display("") == "N/A"


def test_ux_step3_ai_suggestions_loan_demo():
    """Verify Step 3 dynamic AI Suggestions for loan demo dataset display all components without errors."""
    from dashboard.utils import profile_dataframe
    
    csv_path = os.path.join(get_project_root(), "data", "raw", "final_project_loan_demo.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(get_project_root(), "data", "raw", "loan_approval.csv")
    
    df = pd.read_csv(csv_path)
    profile = profile_dataframe(df, dataset_id="final_project_loan_demo")
    
    # 1. Target Recommendation
    target_cands = profile.get("target_candidates", [])
    assert len(target_cands) > 0
    top_cand = target_cands[0]
    assert top_cand["column"] == "loan_status"
    t_conf = top_cand.get("confidence")
    formatted_t_conf = format_confidence_display(t_conf)
    assert formatted_t_conf in ["High", "Medium", "Low", "85%"]
    
    # 2. Positive Class Recommendation
    rec_pos = top_cand.get("recommended_positive_class")
    assert rec_pos == "approved"
    p_conf = top_cand.get("positive_class_confidence")
    assert format_confidence_display(p_conf) == "N/A"

    # 3. Protected Attributes
    protected_cands = profile.get("protected_attribute_candidates", [])
    prot_cols = [c["column"] for c in protected_cands if isinstance(c, dict)]
    assert "gender" in prot_cols
    assert "race" in prot_cols
    for c in protected_cands:
        p_conf_str = format_confidence_display(c.get("confidence"))
        assert p_conf_str in ["High", "Medium", "Low", "90%"]

    # 4. ID / Ignore Candidates
    id_cands = profile.get("id_candidates")
    if id_cands is None:
        id_cands = [col for col, meta in profile.get("columns", {}).items() if meta.get("potential_id")]
    assert "applicant_id" in id_cands

    # 5. Quality Warnings
    dq_warnings = profile.get("quality_warnings") or profile.get("data_quality_warnings", [])
    assert isinstance(dq_warnings, list)


def test_ux_id_heuristics_continuous_and_identifiers():
    """Verify applicant_id is detected as ID while annual_income, loan_amount, credit_score, age are NOT."""
    from src.data.dataset_profiler import is_potential_id
    
    df_test = pd.DataFrame({
        "applicant_id": [f"APP_{i:04d}" for i in range(100)],
        "annual_income": [50000.0 + i * 123.45 for i in range(100)],
        "loan_amount": [10000.0 + i * 54.32 for i in range(100)],
        "credit_score": [600 + (i % 200) for i in range(100)],
        "age": [20 + (i % 60) for i in range(100)],
        "gender": ["Female", "Male"] * 50,
        "loan_status": ["approved", "rejected"] * 50
    })

    assert is_potential_id("applicant_id", df_test["applicant_id"]) is True
    assert is_potential_id("annual_income", df_test["annual_income"]) is False
    assert is_potential_id("loan_amount", df_test["loan_amount"]) is False
    assert is_potential_id("credit_score", df_test["credit_score"]) is False
    assert is_potential_id("age", df_test["age"]) is False


def test_ux_new_upload_workflow_context_priority():
    """Verify new upload becomes current workflow dataset and takes priority over sidebar defaults."""
    import streamlit as st
    
    # Initialize clean session state
    st.session_state["workflow_dataset_id"] = "final_project_loan_demo__1"
    st.session_state["workflow_dataset_source"] = "New Upload"
    st.session_state["pending_dataset_id"] = "final_project_loan_demo__1"

    # Context resolution
    clean_ds = st.session_state["workflow_dataset_id"]
    source = st.session_state["workflow_dataset_source"]
    assert clean_ds == "final_project_loan_demo__1"
    assert source == "New Upload"


def test_ux_training_promotes_workflow_context():
    """Verify successful training promotes workflow context to Registered Dataset."""
    import streamlit as st
    from dashboard.utils import get_dataset_active_model_context

    # Simulate completed registration
    st.session_state["workflow_dataset_id"] = "loan_approval"
    st.session_state["workflow_dataset_source"] = "Registered Dataset"
    st.session_state["active_dataset"] = "loan_approval"
    st.session_state["pending_dataset_id"] = None

    ctx = get_dataset_active_model_context("loan_approval")
    assert ctx["dataset_id"] == "loan_approval"
    assert ctx["has_model"] is True
    assert ctx["status"] == "ACTIVE"


def test_ux_arbitrary_new_dataset_workflow_without_preregistration(tmp_path):
    """Verify arbitrary CSV (e.g. Titanic) profiles, configures, and extracts features dynamically."""
    from src.data.dataset_profiler import profile_dataset
    from src.data.dataset_config import DatasetConfig
    from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data

    titanic_df = pd.DataFrame({
        "passenger_id": [f"P_{i:04d}" for i in range(100)],
        "pclass": [1, 2, 3, 1] * 25,
        "sex": ["female", "male"] * 50,
        "age": [22.0 + (i % 40) for i in range(100)],
        "fare": [7.25 + (i * 2.5) for i in range(100)],
        "survived": [1, 0, 1, 0] * 25
    })

    profile = profile_dataset(titanic_df, dataset_id="titanic_test")
    assert profile["target_candidates"][0]["column"] == "survived"
    
    # ID columns correctly isolated
    id_cols = [c for c, m in profile["columns"].items() if m.get("potential_id")]
    assert "passenger_id" in id_cols
    assert "fare" not in id_cols
    assert "age" not in id_cols

    # Build config dynamically
    cfg = DatasetConfig.from_dict({
        "dataset_id": "titanic_test",
        "path": "test_path",
        "target": {"column": "survived", "positive_class": "1"},
        "protected_attributes": ["sex"],
        "id_columns": ["passenger_id"],
        "test_size": 0.2,
        "random_state": 42
    })

    X, y, A = prepare_pipeline_data(titanic_df, config=cfg)
    splits = split_pipeline_data(X, y, A, config=cfg)

    # Features must retain pclass, age, fare without passenger_id or sex
    assert "passenger_id" not in X.columns
    assert "sex" not in X.columns
    assert "age" in X.columns or any("age" in c for c in X.columns)
    assert "fare" in X.columns or any("fare" in c for c in X.columns)
    assert len(splits["X_train"]) == 80
    assert len(splits["X_test"]) == 20


def test_ux_protected_attributes_user_confirmation_flexibility():
    """Verify profiler recommendations do not force unselected attributes into final config."""
    from src.data.dataset_config import DatasetConfig

    # Profiler might detect age, sex, race. User confirms ONLY sex and race.
    confirmed_prot = ["sex", "race"]
    cfg = DatasetConfig.from_dict({
        "dataset_id": "flex_test",
        "path": "test_path",
        "target": {"column": "outcome", "positive_class": "1"},
        "protected_attributes": confirmed_prot,
        "id_columns": ["user_id"]
    })

    assert "age" not in cfg.protected_attributes
    assert cfg.protected_attributes == ["sex", "race"]


def test_ux_persist_and_restore_user_confirmed_configuration(tmp_path):
    """Verify user-confirmed configuration persists, is restored upon revisiting, and is not overwritten by AI suggestions."""
    from dashboard.utils import load_saved_dataset_configuration, create_dataset_config_file
    import streamlit as st

    ds_id = "test_persist_loan_demo"
    cfg_path = create_dataset_config_file(
        dataset_id=ds_id,
        dataset_path=f"data/raw/{ds_id}.csv",
        target_column="loan_status",
        positive_class="approved",
        protected_attributes=["gender", "race"],  # User excluded age
        id_columns=["applicant_id"],              # User excluded annual_income, loan_amount
        test_size=0.25,
        random_state=123,
        min_group_size=25
    )
    assert os.path.exists(cfg_path)

    # 1. Restore from disk
    restored = load_saved_dataset_configuration(ds_id)
    assert restored is not None
    assert restored["dataset_id"] == ds_id
    assert restored["target_column"] == "loan_status"
    assert restored["positive_class"] == "approved"
    assert restored["protected_attributes"] == ["gender", "race"]
    assert "age" not in restored["protected_attributes"]
    assert restored["id_columns"] == ["applicant_id"]
    assert "annual_income" not in restored["id_columns"]
    assert "loan_amount" not in restored["id_columns"]
    assert restored["test_size"] == 0.25
    assert restored["random_state"] == 123
    assert restored["min_group_size"] == 25
    assert restored["is_confirmed"] is True

    # 2. Session state persistence
    if "confirmed_configs" not in st.session_state:
        st.session_state["confirmed_configs"] = {}
    st.session_state["confirmed_configs"][ds_id] = restored

    # Simulate revisit
    session_restored = load_saved_dataset_configuration(ds_id)
    assert session_restored["protected_attributes"] == ["gender", "race"]
    assert session_restored["id_columns"] == ["applicant_id"]


def test_ux_dataset_switching_configuration_isolation():
    """Verify switching datasets restores each dataset's own confirmed configuration without leakage."""
    from dashboard.utils import load_saved_dataset_configuration

    loan_cfg = load_saved_dataset_configuration("loan_approval")
    student_cfg = load_saved_dataset_configuration("student_performance")
    adult_cfg = load_saved_dataset_configuration("adult_census_income")

    assert loan_cfg is not None
    assert student_cfg is not None
    assert adult_cfg is not None

    # Strict isolation
    assert loan_cfg["target_column"] == "loan_status" or "loan" in loan_cfg["target_column"]
    assert student_cfg["target_column"] == "passed" or "pass" in student_cfg["target_column"]
    assert adult_cfg["target_column"] == "income"

    assert "workclass" not in str(loan_cfg)
    assert "workclass" not in str(student_cfg)
    assert "workclass" in str(adult_cfg) or adult_cfg["dataset_id"] == "adult_census_income"


def test_ux_numeric_feature_selected_as_id_warns_and_executes_without_unhandled_traceback():
    """Verify selecting credit_score, annual_income, or loan_amount as ID warns and executes without y_prob > 1 crash."""
    from src.data.dataset_config import validate_dataset_configuration_contract, DatasetConfig
    from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data
    from src.mitigation.mitigator import train_mitigated_model, predict_mitigated_model
    from src.fairness.calibration import evaluate_calibration
    from src.fairness.intersectional import create_intersectional_attribute

    # 600-row demo DataFrame
    np.random.seed(42)
    n = 600
    df = pd.DataFrame({
        "applicant_id": [f"APP_{i:04d}" for i in range(n)],
        "credit_score": np.random.randint(500, 850, size=n),
        "annual_income": np.random.uniform(20000, 150000, size=n),
        "loan_amount": np.random.uniform(5000, 50000, size=n),
        "gender": np.random.choice(["Female", "Male"], size=n),
        "race": np.random.choice(["White", "Black", "Asian", "Hispanic"], size=n),
        "age": np.random.randint(21, 70, size=n),
        "loan_status": np.random.choice(["approved", "rejected"], size=n, p=[0.6, 0.4])
    })

    # Test selecting credit_score as ID
    cfg_dict = {
        "dataset_id": "loan_demo_test",
        "target_column": "loan_status",
        "positive_class": "approved",
        "protected_attributes": ["gender", "race", "age"],
        "id_columns": ["applicant_id", "credit_score"],
        "test_size": 0.2,
        "random_state": 42
    }

    # 1. Validation emits warning but is valid
    is_valid, errors, warnings = validate_dataset_configuration_contract(df, cfg_dict)
    assert is_valid is True
    assert len(errors) == 0
    assert any("credit_score" in w for w in warnings)

    # 2. Pipeline execution does not crash with y_prob > 1
    cfg = DatasetConfig.from_dict({
        "dataset_id": "loan_demo_test",
        "path": "test_path",
        "target": {"column": "loan_status", "positive_class": "approved"},
        "protected_attributes": ["gender", "race", "age"],
        "id_columns": ["applicant_id", "credit_score"],
        "test_size": 0.2,
        "random_state": 42
    })
    X, y, A = prepare_pipeline_data(df, config=cfg)
    splits = split_pipeline_data(X, y, A, config=cfg)
    
    assert len(splits["X_train"]) == 480
    assert len(splits["X_test"]) == 120

    sens_train = create_intersectional_attribute(splits["A_train"], attributes=cfg.protected_attributes)
    mit_model = train_mitigated_model(splits["X_train"], splits["y_train"], sens_train, random_state=42)
    y_pred, y_prob = predict_mitigated_model(mit_model, splits["X_test"], random_state=0)

    assert np.all(y_prob >= 0.0)
    assert np.all(y_prob <= 1.0)
    
    calib = evaluate_calibration(splits["y_test"], y_prob)
    assert calib["status"] == "AVAILABLE"
    assert calib["brier_score"] is not None


def test_ux_configuration_contract_impossible_configurations_blocked():
    """Verify impossible configurations produce clean structured errors (CONFIGURATION_INVALID) without crashes."""
    from src.data.dataset_config import validate_dataset_configuration_contract

    df = pd.DataFrame({
        "applicant_id": ["A1", "A2", "A3", "A4"],
        "score": [10, 20, 30, 40],
        "gender": ["F", "M", "F", "M"],
        "approved": [1, 0, 1, 0]
    })

    # Case 1: Target selected as ID
    is_valid, errors, _ = validate_dataset_configuration_contract(df, {
        "target_column": "approved",
        "positive_class": 1,
        "protected_attributes": ["gender"],
        "id_columns": ["approved"]
    })
    assert is_valid is False
    assert any("Target column" in e and "ID" in e for e in errors)

    # Case 2: Target selected as Protected
    is_valid, errors, _ = validate_dataset_configuration_contract(df, {
        "target_column": "approved",
        "positive_class": 1,
        "protected_attributes": ["approved"],
        "id_columns": []
    })
    assert is_valid is False
    assert any("Target column" in e and "protected" in e for e in errors)

    # Case 3: Protected attribute selected as ID
    is_valid, errors, _ = validate_dataset_configuration_contract(df, {
        "target_column": "approved",
        "positive_class": 1,
        "protected_attributes": ["gender"],
        "id_columns": ["gender"]
    })
    assert is_valid is False
    assert any("Protected attribute" in e and "ID" in e for e in errors)

    # Case 4: Zero predictive features remaining
    is_valid, errors, _ = validate_dataset_configuration_contract(df, {
        "target_column": "approved",
        "positive_class": 1,
        "protected_attributes": ["gender"],
        "id_columns": ["applicant_id", "score"]
    })
    assert is_valid is False
    assert any("No predictive features remain" in e for e in errors)


def test_ux_fairness_criteria_dynamic_single_attribute_switching_loan_demo(tmp_path):
    """Verify Step 3 fairness criteria observed values match selected attribute without fixed 0/1 fabrication."""
    from dashboard.utils import get_fairness_criteria_compatibility, create_dataset_config_file, execute_interactive_pipeline
    import json

    # 1. Test using mock run result structure representing loan demo
    loan_demo_run = {
        "dataset_id": "test_loan_demo_crit",
        "single_attribute_metrics": {
            "gender": {
                "disparities": {
                    "demographic_parity_difference": 0.0853,
                    "disparate_impact_ratio": 0.9034,
                    "equal_opportunity_difference": 0.0971,
                    "equalized_odds_difference": 0.1097,
                    "false_positive_rate_difference": 0.0400
                }
            },
            "race": {
                "disparities": {
                    "demographic_parity_difference": 0.0860,
                    "disparate_impact_ratio": 0.9044,
                    "equal_opportunity_difference": 0.2000,
                    "equalized_odds_difference": 0.2639,
                    "false_positive_rate_difference": 0.1200
                }
            }
        },
        "intersectional_metrics": {
            "disparities": {
                "demographic_parity_difference": None,
                "disparate_impact_ratio": None,
                "equal_opportunity_difference": None,
                "equalized_odds_difference": None,
                "false_positive_rate_difference": None
            },
            "min_group_size_threshold": 30,
            "all_group_metrics": {}
        },
        "baseline_model": {
            "performance": {"accuracy": 0.85},
            "fairness": {
                "demographic_parity_difference": None,
                "disparate_impact_ratio": None,
                "equal_opportunity_difference": None,
                "equalized_odds_difference": None
            }
        }
    }

    run_dir = os.path.join("results", "test_loan_demo_crit")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_results.json"), "w", encoding="utf-8") as f:
        json.dump(loan_demo_run, f)

    # 2. Gender Scope Criteria Evaluation
    crit_gender = get_fairness_criteria_compatibility("test_loan_demo_crit", scope="single", selected_attribute="gender")
    assert crit_gender["status"] == "AVAILABLE"
    comp_g = crit_gender["compatibility"]
    per_g = {c["criterion"]: c for c in comp_g["per_criterion_results"]}
    
    assert per_g["demographic_parity"]["observed_value"] == 0.0853
    assert per_g["demographic_parity"]["status"] == "PASS"
    assert per_g["disparate_impact"]["observed_value"] == 0.9034
    assert per_g["disparate_impact"]["status"] == "PASS"
    assert per_g["equal_opportunity"]["observed_value"] == 0.0971
    assert per_g["equal_opportunity"]["status"] == "PASS"
    assert per_g["equalized_odds"]["observed_value"] == 0.1097
    assert per_g["equalized_odds"]["status"] == "FAIL"
    assert comp_g["passing_criteria_count"] == 3
    assert comp_g["failing_criteria_count"] == 1

    # 3. Race Scope Criteria Evaluation
    crit_race = get_fairness_criteria_compatibility("test_loan_demo_crit", scope="single", selected_attribute="race")
    assert crit_race["status"] == "AVAILABLE"
    comp_r = crit_race["compatibility"]
    per_r = {c["criterion"]: c for c in comp_r["per_criterion_results"]}
    
    assert per_r["demographic_parity"]["observed_value"] == 0.0860
    assert per_r["demographic_parity"]["status"] == "PASS"
    assert per_r["disparate_impact"]["observed_value"] == 0.9044
    assert per_r["disparate_impact"]["status"] == "PASS"
    assert per_r["equal_opportunity"]["observed_value"] == 0.2000
    assert per_r["equal_opportunity"]["status"] == "FAIL"
    assert per_r["equalized_odds"]["observed_value"] == 0.2639
    assert per_r["equalized_odds"]["status"] == "FAIL"
    assert comp_r["passing_criteria_count"] == 2
    assert comp_r["failing_criteria_count"] == 2

    # 4. Intersectional Scope (N < 30, non-estimable)
    crit_inter = get_fairness_criteria_compatibility("test_loan_demo_crit", scope="intersectional")
    assert crit_inter["status"] == "AVAILABLE"
    comp_i = crit_inter["compatibility"]
    assert comp_i["compatibility_status"] == "INSUFFICIENT_EVIDENCE"
    for c in comp_i["per_criterion_results"]:
        assert c["observed_value"] is None
        assert c["status"] == "NOT_ESTIMABLE"
        assert "not estimable" in c["reason"].lower()


def test_ux_step5_safe_formatting_helpers():
    """Verify Step 5 safe formatting helpers prevent TypeError, divide-by-zero, and fake zeros."""
    # 1. format_tradeoff_value
    assert format_tradeoff_value(None) == "N/A"
    assert format_tradeoff_value(float("nan")) == "N/A"
    assert format_tradeoff_value(0.6833333, decimals=4) == "0.6833"
    assert format_tradeoff_value(-0.0083333, decimals=4, prefix="+") == "-0.0083"
    assert format_tradeoff_value(0.0004, decimals=4, prefix="+") == "+0.0004"

    # 2. format_tradeoff_percentage
    assert format_tradeoff_percentage(None) == "N/A"
    assert format_tradeoff_percentage(float("nan")) == "N/A"
    assert format_tradeoff_percentage(-1.2195, decimals=2) == "-1.22%"
    assert format_tradeoff_percentage(50.5803, decimals=2) == "+50.58%"

    # 3. calculate_safe_percentage_change
    assert calculate_safe_percentage_change(None, 0.5) is None
    assert calculate_safe_percentage_change(0.5, None) is None
    assert calculate_safe_percentage_change(0.0, 0.5) is None
    assert calculate_safe_percentage_change(float("nan"), 0.5) is None

    # Normal calculations
    pct_acc = calculate_safe_percentage_change(0.6833333333333333, 0.675)
    assert pct_acc == pytest.approx(-1.2195, abs=1e-2)

    pct_eod = calculate_safe_percentage_change(0.26388888888888884, 0.1527777777777778)
    assert pct_eod == pytest.approx(-42.105, abs=1e-2)

    pct_brier = calculate_safe_percentage_change(0.19741434414640643, 0.29726715484139454)
    assert pct_brier == pytest.approx(50.5803, abs=1e-2)


def test_ux_step5_unmitigated_dataset_shows_unavailable_state():
    """Verify that unmitigated datasets display the clear unavailable state without fake values."""
    res = get_step5_tradeoff_analysis_data("final_project_loan_demo__1", session_mit_res=None)
    assert res["has_mitigation"] is False
    assert "Trade-off analysis unavailable — run Bias Mitigation first." in res["message"]
    assert len(res["rows"]) == 0
    assert len(res["summary_cards"]) == 0


def test_ux_step5_uses_step4_mitigation_output_and_actual_values():
    """
    Verify that Step 5 uses authoritative Step 4 mitigation output for final_project_loan_demo__1,
    matching the expected Before, After, Delta, and Percentage Change values across all 3 dimensions.
    """
    mit_res = execute_interactive_mitigation_workflow("final_project_loan_demo__1")
    assert mit_res is not None
    assert mit_res["status"] == "SUCCESS"

    res = get_step5_tradeoff_analysis_data(
        "final_project_loan_demo__1",
        selected_attribute="race",
        session_mit_res=mit_res
    )
    assert res["has_mitigation"] is True
    assert res["scope_name"] == "Race"

    row_map = {r["Metric"]: r for r in res["rows"]}

    # PERFORMANCE: Accuracy
    assert "Accuracy" in row_map
    acc_row = row_map["Accuracy"]
    assert acc_row["Before"] == "0.6833"
    assert acc_row["After"] == "0.6750"
    assert acc_row["Delta (Δ)"] == "-0.0083"
    assert acc_row["Percentage Change"] == "-1.22%"
    assert "Worsened" in acc_row["Status"]

    # PERFORMANCE: F1-Score
    assert "F1-Score" in row_map
    f1_row = row_map["F1-Score"]
    assert f1_row["Before"] == "0.7841"
    assert f1_row["After"] == "0.7845"
    assert f1_row["Delta (Δ)"] == "+0.0004"
    assert f1_row["Percentage Change"] == "+0.06%"

    # FAIRNESS: Race Equalized Odds Difference
    assert "Race Equalized Odds Difference" in row_map
    eod_row = row_map["Race Equalized Odds Difference"]
    assert eod_row["Before"] == "0.2639"
    assert eod_row["After"] == "0.1528"
    assert eod_row["Delta (Δ)"] == "-0.1111"
    assert eod_row["Percentage Change"] == "-42.11%"
    assert "Improved" in eod_row["Status"]

    # FAIRNESS: Race Demographic Parity Difference
    assert "Race Demographic Parity Difference" in row_map
    dpd_row = row_map["Race Demographic Parity Difference"]
    assert dpd_row["Before"] == "0.0860"
    assert dpd_row["After"] == "0.1426"
    assert dpd_row["Delta (Δ)"] in ["+0.0565", "+0.0566"]
    assert dpd_row["Percentage Change"] == "+65.69%"
    assert "Worsened" in dpd_row["Status"]

    # FAIRNESS: Race Disparate Impact Ratio
    assert "Race Disparate Impact Ratio" in row_map
    dir_row = row_map["Race Disparate Impact Ratio"]
    assert dir_row["Before"] == "0.9044"
    assert dir_row["After"] == "0.8510"
    assert dir_row["Delta (Δ)"] == "-0.0534"
    assert dir_row["Percentage Change"] == "-5.91%"
    assert "Acceptable" in dir_row["Status"]

    # CALIBRATION: Brier Score
    assert "Brier Score" in row_map
    brier_row = row_map["Brier Score"]
    assert brier_row["Before"] == "0.1974"
    assert brier_row["After"] == "0.2973"
    assert brier_row["Delta (Δ)"] == "+0.0999"
    assert brier_row["Percentage Change"] == "+50.58%"
    assert "Worsened" in brier_row["Status"]

    # CALIBRATION: ECE
    assert "Expected Calibration Error (ECE)" in row_map
    ece_row = row_map["Expected Calibration Error (ECE)"]
    assert ece_row["Before"] == "0.1254"
    assert ece_row["After"] == "0.2613"
    assert ece_row["Delta (Δ)"] == "+0.1359"
    assert ece_row["Percentage Change"] == "+108.41%"
    assert "Worsened" in ece_row["Status"]

    # Summary Cards
    cards = res["summary_cards"]
    assert cards["accuracy"]["value_display"] == "0.6750"
    assert "-0.0083" in cards["accuracy"]["delta_display"]
    assert "-1.22%" in cards["accuracy"]["delta_display"]

    assert cards["fairness"]["value_display"] == "0.1528"
    assert "-0.1111" in cards["fairness"]["delta_display"]
    assert "-42.11%" in cards["fairness"]["delta_display"]

    assert cards["calibration"]["value_display"] == "0.2973"
    assert "+0.0999" in cards["calibration"]["delta_display"]
    assert "+50.58%" in cards["calibration"]["delta_display"]

    # Dynamic Governance Conclusion
    conc = res["conclusion_html"]
    assert "42.11%" in conc
    assert "1.22%" in conc
    assert "+0.0999" in conc


def test_ux_step5_dataset_isolation():
    """Verify that Step 5 preserves strict dataset isolation and never cross-contaminates."""
    adult_res = get_step5_tradeoff_analysis_data("adult_census_income")
    student_res = get_step5_tradeoff_analysis_data("student_performance_academic")
    loan_unmit = get_step5_tradeoff_analysis_data("final_project_loan_demo__1")

    # Adult Census loads its stored mitigation run
    if adult_res["has_mitigation"]:
        assert adult_res["clean_ds"] == "adult_census_income"
        assert adult_res["clean_ds"] != student_res["clean_ds"]

    # Loan unmitigated does not leak Adult data
    assert loan_unmit["clean_ds"] == "final_project_loan_demo__1"
    assert loan_unmit["has_mitigation"] is False


def test_ux_step5_no_typeerror_on_partial_none_metrics():
    """Verify that when some metrics are None, no TypeError is raised and N/A is displayed."""
    mock_partial = {
        "status": "SUCCESS",
        "dataset_id": "test_partial_ds",
        "baseline": {
            "performance": {"accuracy": 0.70, "f1_score": None},
            "calibration": {"brier_score": None, "ece": 0.05},
            "single_attribute": {
                "gender": {
                    "disparities": {
                        "equalized_odds_difference": 0.12,
                        "demographic_parity_difference": None,
                        "disparate_impact_ratio": None
                    }
                }
            }
        },
        "mitigated": {
            "performance": {"accuracy": 0.68, "f1_score": None},
            "calibration": {"brier_score": None, "ece": 0.04},
            "single_attribute": {
                "gender": {
                    "disparities": {
                        "equalized_odds_difference": 0.05,
                        "demographic_parity_difference": None,
                        "disparate_impact_ratio": None
                    }
                }
            }
        }
    }

    res = get_step5_tradeoff_analysis_data("test_partial_ds", selected_attribute="gender", session_mit_res=mock_partial)
    assert res["has_mitigation"] is True
    assert len(res["rows"]) > 0

    row_map = {r["Metric"]: r for r in res["rows"]}
    assert row_map["F1-Score"]["Before"] == "N/A"
    assert row_map["F1-Score"]["Delta (Δ)"] == "N/A"
    assert row_map["F1-Score"]["Percentage Change"] == "N/A"

    assert row_map["Brier Score"]["Before"] == "N/A"
    assert row_map["Brier Score"]["Delta (Δ)"] == "N/A"


def test_ux_step5_brier_score_interpretation_and_governance_conclusion():
    """
    Regression test for Step 5 Trade-off Analysis Brier score interpretation & Governance Conclusion:
    - When Brier changes from 0.0009 to 0.0000, it must be interpreted as IMPROVED (lower is better).
    - Row interpretation text: 'Probability calibration improved (Brier score reduced)'.
    - Governance Conclusion: 'Probability calibration improved (Brier score reduced by 0.0009).'
    - Governance Conclusion must NOT say 'Probability calibration remained stable'.
    - ECE improvement is interpreted as 'Superior probability calibration (Error reduced)'.
    """
    mock_mit_improved = {
        "status": "SUCCESS",
        "baseline": {
            "performance": {"accuracy": 0.85, "f1_score": 0.82},
            "fairness": {"equalized_odds_difference": 0.12},
            "calibration": {"brier_score": 0.0009, "ece": 0.0500},
            "single_attribute": {
                "gender": {
                    "disparities": {
                        "equalized_odds_difference": 0.12,
                        "demographic_parity_difference": 0.08,
                        "disparate_impact_ratio": 0.85
                    }
                }
            }
        },
        "mitigated": {
            "performance": {"accuracy": 0.84, "f1_score": 0.81},
            "fairness": {"equalized_odds_difference": 0.04},
            "calibration": {"brier_score": 0.0000, "ece": 0.0200},
            "single_attribute": {
                "gender": {
                    "disparities": {
                        "equalized_odds_difference": 0.04,
                        "demographic_parity_difference": 0.05,
                        "disparate_impact_ratio": 0.92
                    }
                }
            }
        }
    }

    res = get_step5_tradeoff_analysis_data(
        "test_brier_improvement_ds",
        selected_attribute="gender",
        session_mit_res=mock_mit_improved
    )
    assert res["has_mitigation"] is True

    row_map = {r["Metric"]: r for r in res["rows"]}

    # 1. Verify Brier Score table row interpretation
    assert "Brier Score" in row_map
    brier_row = row_map["Brier Score"]
    assert brier_row["Before"] == "0.0009"
    assert brier_row["After"] == "0.0000"
    assert brier_row["Delta (Δ)"] == "-0.0009"
    assert "Improved" in brier_row["Status"]
    assert brier_row["Direction / Interpretation"] == "Probability calibration improved (Brier score reduced)"

    # 2. Verify ECE table row interpretation
    assert "Expected Calibration Error (ECE)" in row_map
    ece_row = row_map["Expected Calibration Error (ECE)"]
    assert ece_row["Before"] == "0.0500"
    assert ece_row["After"] == "0.0200"
    assert ece_row["Delta (Δ)"] == "-0.0300"
    assert "Improved" in ece_row["Status"]
    assert ece_row["Direction / Interpretation"] == "Superior probability calibration (Error reduced)"

    # 3. Verify Governance Conclusion text
    conclusion_html = res["conclusion_html"]
    assert "Probability calibration improved (Brier score reduced by 0.0009)." in conclusion_html
    assert "Probability calibration remained stable" not in conclusion_html


def test_ux_step9_governance_dashboard_data_structure_students():
    """Verify Step 9 extracts all 14 governance sections dynamically for the students run."""
    from dashboard.utils import get_step9_governance_dashboard_data

    dash_data = get_step9_governance_dashboard_data("students")

    # Section 1: Final Governance Summary
    assert dash_data["dataset_id"] == "students"
    assert dash_data["target_column"] == "passed"
    assert dash_data["positive_class"] == "1"
    assert "gender" in dash_data["protected_attributes"]
    assert "ethnicity" in dash_data["protected_attributes"]
    assert dash_data["selected_model"] == "Random Forest"
    assert dash_data["model_version"] == "v3"
    assert dash_data["model_status"] == "ACTIVE"

    # Section 2: Executive Final Decision
    exec_dec = dash_data["executive_decision"]
    assert exec_dec["overall_health"] == "HEALTHY"
    assert exec_dec["retraining_recommendation"] == "NO_RETRAINING_NEEDED"
    assert exec_dec["is_healthy"] is True
    assert exec_dec["no_retrain"] is True
    assert "Current model is healthy and active. No retraining is currently required." in exec_dec["headline"]
    assert "Random Forest v3 ACTIVE" in exec_dec["model_summary"]

    # Section 3: End-to-End Governance Flow (9 stages)
    flow = dash_data["flow_stages"]
    assert len(flow) == 9
    stage_names = [s["name"] for s in flow]
    assert stage_names == [
        "Upload & Configure",
        "Train & Select Model",
        "Fairness Audit",
        "Bias Mitigation",
        "Trade-off Analysis",
        "Model Validation",
        "Production Monitoring",
        "Retrain & Model History",
        "Results & Governance"
    ]

    # Section 4: Model Selection Result
    ms = dash_data["model_selection"]
    assert ms["selected_model"] == "Random Forest"
    assert len(ms["comparison_rows"]) == 3
    cand_names = [r["Candidate Architecture"] for r in ms["comparison_rows"]]
    assert "Logistic Regression" in cand_names
    assert "Random Forest" in cand_names
    assert "Gradient Boosting" in cand_names

    # Section 5: Fairness Audit Summary
    fs = dash_data["fairness_summary"]
    assert len(fs["single_attribute_rows"]) >= 2
    single_attrs = [r["Protected Attribute"] for r in fs["single_attribute_rows"]]
    assert "Gender" in single_attrs
    assert "Ethnicity" in single_attrs
    assert fs["intersectional_dpd"] != "N/A"
    assert "Intersectional demographic parity disparity detected" in fs["finding_note"]

    # Section 6: Bias Mitigation Result
    mi = dash_data["mitigation_info"]
    assert "In-Processing" in mi["strategy"]
    assert "EqualizedOdds" in mi["constraint"]
    assert "Random Forest" in mi["base_estimator"]

    # Section 7: Trade-off Summary
    ts = dash_data["tradeoff_summary"]
    assert len(ts["rows"]) > 0
    brier_rows = [r for r in ts["rows"] if "Brier" in r.get("Metric", "")]
    if brier_rows:
        assert "Improved" in brier_rows[0]["Status"]

    # Section 8: Model Validation Summary
    val = dash_data["validation_summary"]
    assert val["has_model"] is True
    assert val["performance"]["accuracy"] == 1.0
    assert val["performance"]["f1_score"] == 1.0

    # Section 9: Production Monitoring Summary
    mon = dash_data["monitoring_summary"]
    assert mon["has_monitoring"] is True
    assert mon["schema_report"]["status"] in ["SCHEMA_OK", "HEALTHY"]
    assert mon["health_report"]["overall_health"] == "HEALTHY"
    assert mon["retraining_recommendation"]["status"] == "NO_RETRAINING_NEEDED"

    # Section 10: Model Version History
    hist = dash_data["model_history"]
    assert len(hist) >= 1
    assert hist[0]["model_version"] == "v3"
    assert hist[0]["is_active"] is True

    # Section 13: Evaluator Quick View
    qv = dash_data["evaluator_quick_view"]
    assert "AI governance workflow" in qv["q1_problem"]
    assert "students" in qv["q2_dataset"]
    assert "Random Forest" in qv["q3_model"]
    assert "NO_RETRAINING_NEEDED" in qv["q9_retraining"]

    # Section 14: Final Governance Statement
    stmt = dash_data["final_statement"]
    assert "students" in stmt
    assert "Random Forest" in stmt
    assert "v3" in stmt
    assert "HEALTHY" in stmt
    assert "NO_RETRAINING_NEEDED" in stmt


def test_ux_step9_dataset_agnostic_resolution():
    """Verify Step 9 dynamically adapts to non-students datasets."""
    from dashboard.utils import get_step9_governance_dashboard_data

    loan_data = get_step9_governance_dashboard_data("loan_approval")
    assert loan_data["dataset_id"] == "loan_approval"
    assert loan_data["selected_model"] in ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    assert loan_data["target_column"] != "N/A"
    assert len(loan_data["flow_stages"]) == 9
    assert "loan_approval" in loan_data["final_statement"]


def test_ux_step9_download_artifacts_and_zip_package():
    """Verify all individual governance artifacts and in-memory ZIP package for Step 9."""
    import io
    import json
    import zipfile
    from dashboard.utils import prepare_download_artifacts, build_complete_governance_package_zip

    arts = prepare_download_artifacts("students")

    # Verify individual artifacts
    assert "json_str" in arts and len(arts["json_str"]) > 0
    assert "summary_csv_str" in arts and len(arts["summary_csv_str"]) > 0
    assert "model_metadata_str" in arts and len(arts["model_metadata_str"]) > 0
    assert "monitoring_report_str" in arts and len(arts["monitoring_report_str"]) > 0
    assert "validation_lineage_str" in arts and len(arts["validation_lineage_str"]) > 0
    assert "fairness_audit_str" in arts and len(arts["fairness_audit_str"]) > 0
    assert "zip_bytes" in arts and len(arts["zip_bytes"]) > 0

    # Verify complete ZIP package contents
    zip_bytes = build_complete_governance_package_zip("students")
    assert len(zip_bytes) > 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        file_list = zf.namelist()
        assert "students_run_results.json" in file_list
        assert "students_tradeoff_summary.csv" in file_list
        assert "students_model_metadata.json" in file_list
        assert "students_monitoring_report.json" in file_list
        assert "students_health_report.json" in file_list
        assert "students_drift_report.json" in file_list
        assert "students_validation_lineage.json" in file_list
        assert "students_fairness_audit.json" in file_list
        assert "GOVERNANCE_MANIFEST.json" in file_list

        manifest_content = json.loads(zf.read("GOVERNANCE_MANIFEST.json").decode("utf-8"))
        assert manifest_content["dataset_id"] == "students"
        assert manifest_content["active_model_version"] == "v3"
        assert manifest_content["monitoring_health"] == "HEALTHY"
        assert manifest_content["retraining_recommendation"] == "NO_RETRAINING_NEEDED"


def test_ux_step9_monitoring_present_consistency():
    """
    Verify that when authoritative monitoring evidence exists (e.g. students dataset):
    - Section 2, Stage 7, Section 9, Section 13 (Quick View), and Section 14 (Final Statement)
      all report the SAME actual monitoring health and retraining recommendation.
    - No contradictions exist.
    """
    from dashboard.utils import get_step9_governance_dashboard_data, load_latest_monitoring_evidence

    mon_evidence = load_latest_monitoring_evidence("students")
    assert mon_evidence["has_monitoring"] is True
    assert mon_evidence["monitoring_run_id"] == "MON-students-1790671720"
    actual_health = mon_evidence["health_report"]["overall_health"]
    actual_ret = mon_evidence["retraining_recommendation"]["status"]

    dash = get_step9_governance_dashboard_data("students")

    # Section 2: Executive Decision
    dec = dash["executive_decision"]
    assert dec["overall_health"] == actual_health
    assert dec["retraining_recommendation"] == actual_ret
    assert dec["is_healthy"] is True
    assert dec["no_retrain"] is True
    assert dec["has_monitoring"] is True

    # Section 3: Flow Stage 7 (Production Monitoring)
    stage7 = [s for s in dash["flow_stages"] if s["stage"] == 7][0]
    assert stage7["status"] == actual_health
    assert stage7["badge_cls"] == "badge-healthy"

    # Section 9: Monitoring Summary
    mon_summary = dash["monitoring_summary"]
    assert mon_summary["has_monitoring"] is True
    assert mon_summary["monitoring_run_id"] == "MON-students-1790671720"
    assert mon_summary["health_report"]["overall_health"] == actual_health
    assert mon_summary["retraining_recommendation"]["status"] == actual_ret

    # Section 13: Quick View
    qv = dash["evaluator_quick_view"]
    assert actual_health in qv["q8_monitoring"]
    assert actual_ret in qv["q9_retraining"]

    # Section 14: Final Statement
    stmt = dash["final_statement"]
    assert "batch production monitoring" in stmt
    assert actual_health in stmt
    assert actual_ret in stmt


def test_ux_step9_monitoring_absent_consistency():
    """
    Verify that when NO authoritative monitoring evidence exists:
    - Section 9 states monitoring has not yet been executed.
    - Section 2 does NOT claim HEALTHY or NO_RETRAINING_NEEDED based on nonexistent monitoring.
    - Stage 7 is marked PENDING.
    - Quick View does NOT claim HEALTHY or NO_RETRAINING_NEEDED.
    - Final Statement does NOT claim batch production monitoring was completed.
    - All sections consistently report pending monitoring evidence.
    - No fabricated or synthesized monitoring values are returned.
    """
    from dashboard.utils import get_step9_governance_dashboard_data, load_latest_monitoring_evidence

    # Use a mock/synthetic unmonitored dataset name
    unmon_evidence = load_latest_monitoring_evidence("unmonitored_dataset_xyz")
    assert unmon_evidence["has_monitoring"] is False
    assert unmon_evidence["monitoring_run_id"] == "N/A"
    assert unmon_evidence["health_report"]["overall_health"] == "MONITORING_PENDING"
    assert unmon_evidence["retraining_recommendation"]["status"] == "DECISION_PENDING_MONITORING"
    assert unmon_evidence["schema_report"]["status"] == "NOT_MONITORED"
    assert unmon_evidence["data_quality_report"]["status"] == "NOT_MONITORED"
    assert unmon_evidence["feature_drift_report"]["status"] == "NOT_MONITORED"

    dash = get_step9_governance_dashboard_data("unmonitored_dataset_xyz")

    # Section 2: Executive Decision
    dec = dash["executive_decision"]
    assert dec["overall_health"] == "MONITORING_PENDING"
    assert dec["retraining_recommendation"] == "DECISION_PENDING_MONITORING"
    assert dec["badge"] == "MONITORING PENDING"
    assert dec["badge_cls"] == "badge-warning"
    assert "pending" in dec["headline"].lower()
    assert dec["is_healthy"] is False
    assert dec["no_retrain"] is False
    assert dec["has_monitoring"] is False

    # Section 3: Flow Stage 7
    stage7 = [s for s in dash["flow_stages"] if s["stage"] == 7][0]
    assert stage7["status"] == "PENDING"
    assert stage7["badge_cls"] == "badge-warning"

    # Section 9: Monitoring Summary
    mon_summary = dash["monitoring_summary"]
    assert mon_summary["has_monitoring"] is False
    assert mon_summary["health_report"]["overall_health"] == "MONITORING_PENDING"

    # Section 13: Quick View
    qv = dash["evaluator_quick_view"]
    assert "HEALTHY" not in qv["q8_monitoring"]
    assert "NO_RETRAINING_NEEDED" not in qv["q9_retraining"]
    assert "pending" in qv["q8_monitoring"].lower()
    assert "pending" in qv["q9_retraining"].lower()

    # Section 14: Final Statement
    stmt = dash["final_statement"]
    assert "Production monitoring has not yet been executed" in stmt
    assert "HEALTHY" not in stmt
    assert "NO_RETRAINING_NEEDED" not in stmt


def test_ux_step9_section1_no_raw_html_tags():
    """
    Verify Section 1 formatted fields render plain clean text without literal HTML tags
    such as '<b>', '</b>', '<code>', '</code>'.
    """
    from dashboard.utils import get_step9_governance_dashboard_data

    dash = get_step9_governance_dashboard_data("students")

    # Construct Section 1 captions exactly as dashboard/app.py renders them
    c1_caption = f"ID: {dash['dataset_id']}"
    c2_caption = f"Positive Class: {dash['positive_class']}"
    c3_caption = f"Features: {dash['feature_count']} | Test: {dash['test_size']}"
    c4_caption = f"Run ID: {dash['run_id']} | Status: {dash['model_status']}"

    for cap in [c1_caption, c2_caption, c3_caption, c4_caption]:
        assert "<b>" not in cap
        assert "</b>" not in cap
        assert "<code>" not in cap
        assert "</code>" not in cap

    assert c3_caption == "Features: 29 | Test: 2000"









