"""
Regression Test Suite: Step 2 -> Step 3 -> Step 4 Model Context Propagation & Base Estimator Alignment.

Proves:
A. Step 2 selected model = Step 4 mitigation base model.
B. Random Forest selection causes Step 4 to use RandomForestClassifier (with authoritative hyperparameters).
C. Step 4 baseline accuracy/F1/fairness exactly match Step 3 for the selected model.
D. Logistic Regression is not silently substituted when Random Forest is selected.
E. Existing loan dataset behavior remains unchanged.
F. Different datasets can select different base estimators correctly.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from dashboard.utils import (
    execute_interactive_mitigation_workflow,
    execute_pareto_frontier_run,
    resolve_dataset_selected_model_key,
    normalize_model_key,
    get_dataset_active_model_context,
    load_dataset_run_result,
    normalize_run_results
)
from src.mitigation.mitigator import get_base_estimator, train_mitigated_model_with_logging
from src.models.model_registry import get_candidate_model


def test_regression_a_step2_selected_model_equals_step4_mitigation_base_model():
    """
    TEST A: Step 2 selected model == Step 4 mitigation base model.
    Verifies that the model selected in Step 2 is propagated directly into Step 4 mitigation.
    """
    # 1. Students dataset: Step 2 selected Random Forest
    students_ctx = get_dataset_active_model_context("students")
    assert students_ctx["selected_model"] == "Random Forest"
    students_resolved_key = resolve_dataset_selected_model_key("students")
    assert students_resolved_key == "random_forest"

    # 2. Loan dataset: Step 2 selected Logistic Regression
    loan_ctx = get_dataset_active_model_context("final_project_loan_demo__1")
    assert loan_ctx["selected_model"] == "Logistic Regression"
    loan_resolved_key = resolve_dataset_selected_model_key("final_project_loan_demo__1")
    assert loan_resolved_key == "logistic_regression"

    # 3. Adult dataset: Step 2 selected Gradient Boosting
    adult_ctx = get_dataset_active_model_context("adult_census_income")
    assert adult_ctx["selected_model"] == "Gradient Boosting"
    adult_resolved_key = resolve_dataset_selected_model_key("adult_census_income")
    assert adult_resolved_key == "gradient_boosting"


def test_regression_b_random_forest_selection_causes_step4_to_use_random_forest_classifier():
    """
    TEST B: Random Forest selection causes Step 4 to use RandomForestClassifier with authoritative hyperparameters.
    """
    logs = []
    mit_res = execute_interactive_mitigation_workflow(
        dataset_id="students",
        strategy_type="in_processing",
        constraint_type="EqualizedOdds",
        eps=0.01,
        max_iter=10,
        logger_callback=lambda m: logs.append(m)
    )

    # Mitigation result metadata
    assert mit_res["status"] == "SUCCESS"
    assert mit_res["selected_model"] == "random_forest"
    assert mit_res["model_type"] == "random_forest"
    assert mit_res["baseline"]["model_key"] == "random_forest"
    assert mit_res["baseline"]["model_name"] == "Random Forest"
    assert mit_res["mitigated"]["base_estimator"] == "random_forest"
    assert "Random Forest" in mit_res["mitigated"]["algorithm"]

    # Check live logs for authoritative base estimator
    full_logs_text = " ".join(logs + mit_res.get("logs", []))
    assert "[CONFIG] Base Estimator: RandomForestClassifier" in full_logs_text
    assert "LogisticRegression" not in full_logs_text or "LogisticRegression" not in [l for l in logs if "Base Estimator:" in l]

    # Verify authoritative hyperparameters (n_estimators=100 from model_registry)
    base_est = get_base_estimator("random_forest", random_state=42)
    assert isinstance(base_est, RandomForestClassifier)
    assert base_est.n_estimators == 100


def test_regression_c_step4_baseline_matches_step3_for_selected_model():
    """
    TEST C: Step 4 baseline accuracy/F1/fairness exactly match Step 3 for the selected model.
    For students dataset with Random Forest:
    - Accuracy = 1.0000
    - F1 = 1.0000
    - Gender EOD = 0.0000
    - Gender DPD = 0.0623
    - Gender DIR = 0.9354
    """
    # Step 3 audited results
    raw_run = load_dataset_run_result("students")
    norm_step3 = normalize_run_results(raw_run)
    step3_perf = norm_step3["baseline"]["performance"]
    step3_single = norm_step3["single_attribute"]["gender"]["disparities"]

    # Step 4 execution
    step4_res = execute_interactive_mitigation_workflow(
        dataset_id="students",
        strategy_type="in_processing",
        constraint_type="EqualizedOdds",
        eps=0.01,
        max_iter=5
    )

    step4_base_perf = step4_res["baseline"]["performance"]
    step4_base_single = step4_res["baseline"]["single_attribute"]["gender"]["disparities"]

    # Strict mathematical match between Step 3 and Step 4 Baseline
    assert np.isclose(step4_base_perf["accuracy"], step3_perf["accuracy"])
    assert np.isclose(step4_base_perf["accuracy"], 1.0000)
    assert np.isclose(step4_base_perf["f1_score"], step3_perf["f1_score"])
    assert np.isclose(step4_base_perf["f1_score"], 1.0000)

    assert np.isclose(step4_base_single["equalized_odds_difference"], step3_single["equalized_odds_difference"])
    assert np.isclose(step4_base_single["equalized_odds_difference"], 0.0000)
    assert np.isclose(step4_base_single["demographic_parity_difference"], step3_single["demographic_parity_difference"])
    assert np.isclose(step4_base_single["demographic_parity_difference"], 0.0622787, atol=1e-4)
    assert np.isclose(step4_base_single["disparate_impact_ratio"], step3_single["disparate_impact_ratio"])
    assert np.isclose(step4_base_single["disparate_impact_ratio"], 0.93544, atol=1e-4)


def test_regression_d_logistic_regression_not_silently_substituted():
    """
    TEST D: Logistic Regression is not silently substituted when Random Forest is selected.
    Verifies that the old buggy baseline metrics (Acc=0.9920, F1=0.9958, EOD=0.0857) from Logistic Regression
    are NOT reported on students.
    """
    step4_res = execute_interactive_mitigation_workflow(
        dataset_id="students",
        strategy_type="in_processing",
        constraint_type="EqualizedOdds",
        eps=0.01,
        max_iter=5
    )

    step4_acc = step4_res["baseline"]["performance"]["accuracy"]
    step4_f1 = step4_res["baseline"]["performance"]["f1_score"]
    step4_eod = step4_res["baseline"]["single_attribute"]["gender"]["disparities"]["equalized_odds_difference"]

    # Must NOT equal Logistic Regression numbers (0.9920, 0.9958, 0.0857)
    assert not np.isclose(step4_acc, 0.9920, atol=1e-3)
    assert not np.isclose(step4_f1, 0.9958, atol=1e-3)
    assert not np.isclose(step4_eod, 0.0857, atol=1e-3)

    # Must equal Random Forest numbers (1.0, 1.0, 0.0)
    assert np.isclose(step4_acc, 1.0000)
    assert np.isclose(step4_f1, 1.0000)
    assert np.isclose(step4_eod, 0.0000)


def test_regression_e_existing_loan_dataset_behavior_remains_unchanged():
    """
    TEST E: Existing loan dataset behavior remains unchanged.
    Verifies that datasets where Step 2 selected Logistic Regression (final_project_loan_demo__1)
    continue using LogisticRegression as base estimator with unchanged baseline metrics.
    """
    logs = []
    loan_res = execute_interactive_mitigation_workflow(
        dataset_id="final_project_loan_demo__1",
        strategy_type="in_processing",
        constraint_type="EqualizedOdds",
        eps=0.01,
        max_iter=5,
        logger_callback=lambda m: logs.append(m)
    )

    assert loan_res["status"] == "SUCCESS"
    assert loan_res["selected_model"] == "logistic_regression"
    assert loan_res["baseline"]["model_key"] == "logistic_regression"
    assert loan_res["mitigated"]["base_estimator"] == "logistic_regression"

    # Verify baseline metrics
    b_perf = loan_res["baseline"]["performance"]
    assert np.isclose(b_perf["accuracy"], 0.6833, atol=1e-3)
    assert np.isclose(b_perf["f1_score"], 0.7841, atol=1e-3)

    # Verify logs
    log_text = " ".join(logs)
    assert "[CONFIG] Base Estimator: LogisticRegression" in log_text


def test_regression_f_different_datasets_select_different_base_estimators():
    """
    TEST F: Different datasets can select different base estimators correctly.
    """
    datasets_expected = {
        "students": "random_forest",
        "student_performance": "random_forest",
        "final_project_loan_demo__1": "logistic_regression",
        "loan_approval": "logistic_regression",
        "adult_census_income": "gradient_boosting"
    }

    for ds_id, expected_model in datasets_expected.items():
        resolved_key = resolve_dataset_selected_model_key(ds_id)
        assert resolved_key == expected_model, f"Dataset '{ds_id}' resolved to '{resolved_key}', expected '{expected_model}'"

        base_est = get_base_estimator(resolved_key, random_state=42)
        if expected_model == "random_forest":
            assert isinstance(base_est, RandomForestClassifier)
        elif expected_model == "gradient_boosting":
            assert isinstance(base_est, GradientBoostingClassifier)
        elif expected_model == "logistic_regression":
            assert isinstance(base_est, LogisticRegression)


def test_regression_g_mitigation_result_explicitly_tied_to_dataset_model_version():
    """
    TEST G: The mitigation result is explicitly tied to dataset + selected model + model version.
    """
    active_ctx = get_dataset_active_model_context("students")
    expected_version = active_ctx.get("model_version", "v1") if active_ctx else "v1"

    students_res = execute_interactive_mitigation_workflow(
        dataset_id="students",
        strategy_type="in_processing",
        constraint_type="EqualizedOdds",
        eps=0.01,
        max_iter=5
    )

    assert students_res["dataset_id"] == "students"
    assert students_res["selected_model"] == "random_forest"
    assert students_res["model_type"] == "random_forest"
    assert students_res["model_version"] == expected_version
    assert students_res["baseline"]["model_key"] == "random_forest"
    assert students_res["mitigated"]["base_estimator"] == "random_forest"
