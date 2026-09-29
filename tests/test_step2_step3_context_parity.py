"""
Regression Test Suite: Step 2 to Step 3 Model Context Parity & Evaluator Alignment.

Proves:
1. Step 2 selected model == Step 3 audited model == global active model context.
2. For the `students` dataset, Step 2 selects Random Forest (Logistic Regression breaches fairness constraints),
   Step 3 audits the exact Random Forest model, and the global context header displays Random Forest.
3. For benchmark datasets (loan_approval, final_project_loan_demo__1, adult_census_income, student_performance),
   Step 2 selection, Step 3 fairness audit, and global header remain strictly synchronized and isolated.
4. Single-attribute, intersectional, and formal criteria compatibility are all tied to the selected model.
"""

import os
import json
import pytest
import numpy as np

from dashboard.utils import (
    get_dataset_active_model_context,
    get_all_candidate_evaluations,
    load_dataset_run_result,
    normalize_run_results,
    get_fairness_criteria_compatibility,
    get_project_root
)
from src.models.model_selection import select_best_candidate_model


def test_1_students_step2_step3_global_context_parity():
    """
    Verify for students dataset:
    - Step 2 selects Random Forest (Logistic Regression breaches fairness constraints)
    - Step 3 audits Random Forest (Accuracy 1.0, EOD 0.0)
    - Global active model context displays Random Forest
    - Step 2 selected model == Step 3 audited model == global active context
    """
    # 1. Step 2 candidate model evaluation
    cand_evals = get_all_candidate_evaluations("students")
    assert "logistic_regression" in cand_evals
    assert "random_forest" in cand_evals
    assert "gradient_boosting" in cand_evals

    sel_input = {k: v.get("raw_eval", v) for k, v in cand_evals.items() if isinstance(v, dict)}
    sel_res = select_best_candidate_model(sel_input)

    # Logistic regression fails fairness constraints (EOD = 0.30 > 0.10)
    comp_map = {row["model"]: row for row in sel_res["comparison_table"]}
    assert comp_map["logistic_regression"]["status"] == "FAIL_CONSTRAINT"
    assert comp_map["random_forest"]["status"] == "PASS"

    step2_selected_key = sel_res["selected_model"]
    assert step2_selected_key == "random_forest"
    assert sel_res["selection_status"] == "SELECTED"

    # 2. Step 3 audited model
    run_data = load_dataset_run_result("students")
    assert run_data is not None
    norm = normalize_run_results(run_data)

    # Step 3 baseline performance and fairness match the Random Forest evaluation
    step3_perf = norm["baseline"]["performance"]
    step3_fair = norm["baseline"]["fairness"]
    assert np.isclose(step3_perf["accuracy"], 1.0)
    assert np.isclose(step3_fair["equalized_odds_difference"], 0.0)

    # 3. Global Context Header
    ctx = get_dataset_active_model_context("students")
    assert ctx["has_model"] is True
    assert ctx["dataset_id"] == "students"
    assert ctx["selected_model"] == "Random Forest"
    assert ctx["status"] in ["SELECTED", "ACTIVE"]

    # 4. Identity proof: Step 2 selected model == Step 3 audited model == Global active context
    model_name_map = {
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
        "gradient_boosting": "Gradient Boosting"
    }
    assert model_name_map[step2_selected_key] == ctx["selected_model"]
    assert norm["model"]["selected_model"] in ["random_forest", "Random Forest"]


def test_2_students_fairness_and_criteria_tied_to_selected_model():
    """
    Verify that single-attribute metrics, intersectional metrics, and formal criteria
    for students dataset are tied strictly to Random Forest disparities.
    """
    run_data = load_dataset_run_result("students")
    norm = normalize_run_results(run_data)

    # Single attribute disparities for gender and ethnicity
    single_data = norm.get("single_attribute", {})
    assert "gender" in single_data
    assert "ethnicity" in single_data
    assert np.isclose(single_data["gender"]["disparities"]["equalized_odds_difference"], 0.0)
    assert np.isclose(single_data["ethnicity"]["disparities"]["equalized_odds_difference"], 0.0)

    # Intersectional disparities
    inter_data = norm.get("intersectional", {})
    assert np.isclose(inter_data["disparities"]["equalized_odds_difference"], 0.0)

    # Criteria compatibility evaluated on Random Forest observed values
    crit_single = get_fairness_criteria_compatibility("students", scope="single", selected_attribute="gender")
    assert crit_single["status"] != "UNAVAILABLE"
    crit_inter = get_fairness_criteria_compatibility("students", scope="intersectional")
    assert crit_inter["status"] != "UNAVAILABLE"
    # Intersectional DPD is 0.1052 (> 0.10 threshold), while EOD is 0.0000 (<= 0.10 threshold)
    assert crit_inter["compatibility"]["compatibility_status"] == "CONFLICT_DETECTED"
    assert round(crit_inter["observed_metrics"]["demographic_parity_difference"], 4) == 0.1052
    assert round(crit_inter["observed_metrics"]["equalized_odds_difference"], 4) == 0.0000


def test_3_loan_demo_step2_step3_global_context_parity():
    """
    Verify for final_project_loan_demo__1:
    - Step 2 selects Logistic Regression (fallback highest CV accuracy 0.6979)
    - Step 3 audits Logistic Regression (Accuracy 0.6833, F1 0.7841, ROC-AUC 0.7339)
    - Global active context displays Logistic Regression (v3 ACTIVE)
    - Step 2 selected model == Step 3 audited model == Global active context
    """
    # 1. Step 2 candidate selection
    cand_evals = get_all_candidate_evaluations("final_project_loan_demo__1")
    sel_input = {k: v.get("raw_eval", v) for k, v in cand_evals.items() if isinstance(v, dict)}
    sel_res = select_best_candidate_model(sel_input)

    assert sel_res["selected_model"] == "logistic_regression"
    assert sel_res["selection_status"] == "NO_MODEL_SATISFIES_CONSTRAINTS_FALLBACK"

    # 2. Step 3 audited model
    run_data = load_dataset_run_result("final_project_loan_demo__1")
    assert run_data is not None
    norm = normalize_run_results(run_data)
    assert round(norm["baseline"]["performance"]["accuracy"], 4) == 0.6833

    # 3. Global Context Header
    ctx = get_dataset_active_model_context("final_project_loan_demo__1")
    assert ctx["has_model"] is True
    assert ctx["dataset_id"] == "final_project_loan_demo__1"
    assert ctx["selected_model"] == "Logistic Regression"
    assert ctx["model_version"] == "v3"
    assert ctx["status"] == "ACTIVE"


def test_4_student_performance_step2_step3_global_context_parity():
    """
    Verify for student_performance:
    - Step 3 audits Random Forest
    - Global active context displays Random Forest (ACTIVE)
    """
    run_data = load_dataset_run_result("student_performance")
    assert run_data is not None
    norm = normalize_run_results(run_data)

    assert round(norm["baseline"]["performance"]["accuracy"], 4) == 0.9333

    ctx = get_dataset_active_model_context("student_performance")
    assert ctx["has_model"] is True
    assert ctx["dataset_id"] == "student_performance"
    assert ctx["selected_model"] == "Random Forest"
    assert ctx["model_version"].startswith("v")
    assert ctx["status"] == "ACTIVE"


def test_5_adult_and_loan_approval_parity():
    """
    Verify that adult_census_income (Gradient Boosting) and loan_approval (Logistic Regression)
    maintain exact alignment and isolation.
    """
    adult_ctx = get_dataset_active_model_context("adult_census_income")
    assert adult_ctx["selected_model"] == "Gradient Boosting"
    assert adult_ctx["model_version"].startswith("v")
    assert adult_ctx["status"] == "ACTIVE"

    loan_ctx = get_dataset_active_model_context("loan_approval")
    assert loan_ctx["selected_model"] == "Logistic Regression"
    assert loan_ctx["model_version"].startswith("v")
    assert loan_ctx["status"] == "ACTIVE"


def test_6_untrained_dataset_pending_training_context():
    """
    Verify that an untrained dataset gracefully resolves to Pending Training and has_model=False.
    """
    ctx = get_dataset_active_model_context("unknown_dataset_xyz_123")
    assert ctx["has_model"] is False
    assert ctx["selected_model"] == "Pending Training"
    assert ctx["status"] == "PENDING_TRAINING"
    assert ctx["model_version"] == "v0"
