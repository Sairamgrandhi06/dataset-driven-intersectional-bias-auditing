"""
Regression Test Suite: Step 2 Candidate Model Evaluation & Authoritative Subsystem Parity.

Verifies:
1. Step 2 candidate models use the identical estimator definitions and hyperparameters as src/models/model_registry.py.
2. Step 2 executes the standard 5-fold StratifiedKFold CV on training data.
3. Step 2 candidate metrics on final_project_loan_demo__1 mathematically match authoritative candidate evaluation.
4. Step 2 model selection aligns with configured selection policy and transparent reason reporting.
5. Strict dataset isolation across candidate model evaluations.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.models.model_registry import get_candidate_model, list_supported_models
from src.models.model_trainer import train_and_cross_validate_candidate
from src.models.model_evaluator import evaluate_candidate_model
from src.models.model_selection import select_best_candidate_model
from src.data.dataset_config import load_dataset_config_by_id
from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data
from dashboard.utils import (
    get_candidate_model_audit_metrics,
    get_all_candidate_evaluations,
    get_raw_dataset_dataframe,
    get_project_root
)


def test_1_step2_uses_exact_same_estimator_configurations():
    """
    Verify that Step 2 candidate model evaluation utilizes the exact estimator classes,
    hyperparameters, and solver definitions from src/models/model_registry.py.
    """
    supported = list_supported_models()
    assert set(supported) == {"logistic_regression", "random_forest", "gradient_boosting"}

    # 1. Logistic Regression
    lr = get_candidate_model("logistic_regression", random_seed=42)
    assert isinstance(lr, LogisticRegression)
    assert lr.max_iter == 1000
    assert lr.random_state == 42
    # Ensure default solver is lbfgs (not liblinear)
    assert lr.solver == "lbfgs"

    # 2. Random Forest
    rf = get_candidate_model("random_forest", random_seed=42)
    assert isinstance(rf, RandomForestClassifier)
    assert rf.n_estimators == 100
    assert rf.random_state == 42

    # 3. Gradient Boosting
    gb = get_candidate_model("gradient_boosting", random_seed=42)
    assert isinstance(gb, GradientBoostingClassifier)
    assert gb.n_estimators == 100
    assert gb.random_state == 42


def test_2_step2_executes_5_fold_stratified_cv_and_evaluates_held_out_test():
    """
    Verify that get_candidate_model_audit_metrics runs 5-fold StratifiedKFold CV
    and held-out test evaluation on the active dataset.
    """
    res = get_candidate_model_audit_metrics("final_project_loan_demo__1", model_key="logistic_regression")
    
    assert res["model_key"] == "logistic_regression"
    assert "cv_metrics" in res
    cv = res["cv_metrics"]
    assert cv["folds_completed"] == 5
    assert cv["cv_accuracy_mean"] > 0.0
    assert "cv_f1_mean" in cv
    assert "cv_roc_auc_mean" in cv

    assert "performance" in res
    perf = res["performance"]
    assert perf["accuracy"] > 0.0
    assert perf["f1_score"] > 0.0
    assert "confusion_matrix" in perf
    assert sum([perf["confusion_matrix"]["TN"], perf["confusion_matrix"]["FP"], perf["confusion_matrix"]["FN"], perf["confusion_matrix"]["TP"]]) == 120

    assert "calibration" in res
    cal = res["calibration"]
    assert cal["status"] == "AVAILABLE"
    assert cal["brier_score"] is not None
    assert cal["ece"] is not None


def test_3_step2_metrics_mathematically_match_authoritative_subsystem():
    """
    Verify mathematical identity between get_all_candidate_evaluations
    and direct pipeline candidate training/evaluation on final_project_loan_demo__1.
    """
    clean_id = "final_project_loan_demo__1"
    root = get_project_root()
    df = get_raw_dataset_dataframe(clean_id)
    cfg = load_dataset_config_by_id(clean_id, base_dir=root)

    X, y, A = prepare_pipeline_data(df, config=cfg)
    splits = split_pipeline_data(X, y, A, config=cfg)

    # 1. Authoritative candidate evaluation directly
    cv_res_lr = train_and_cross_validate_candidate("logistic_regression", splits["X_train"], splits["y_train"], cv_folds=5, random_seed=42)
    eval_res_lr = evaluate_candidate_model(
        "logistic_regression",
        cv_res_lr["fitted_model"],
        cv_res_lr["cv_metrics"],
        splits["X_test"],
        splits["y_test"],
        splits["A_test"],
        min_group_size=30
    )

    # 2. Step 2 dynamic candidate evaluation
    step2_evals = get_all_candidate_evaluations(clean_id)
    step2_lr = step2_evals["logistic_regression"]

    # Verify identical values
    assert np.isclose(step2_lr["cv_metrics"]["cv_accuracy_mean"], cv_res_lr["cv_metrics"]["cv_accuracy_mean"])
    assert np.isclose(step2_lr["performance"]["accuracy"], eval_res_lr["test_performance"]["accuracy"])
    assert np.isclose(step2_lr["performance"]["f1_score"], eval_res_lr["test_performance"]["f1_score"])
    assert np.isclose(step2_lr["performance"]["roc_auc"], eval_res_lr["test_performance"]["roc_auc"])
    assert np.isclose(step2_lr["calibration"]["brier_score"], eval_res_lr["calibration_metrics"]["brier_score"])
    assert np.isclose(step2_lr["calibration"]["ece"], eval_res_lr["calibration_metrics"]["ece"])

    # Baseline matches known recorded benchmark values
    assert round(step2_lr["performance"]["accuracy"], 4) == 0.6833
    assert round(step2_lr["performance"]["f1_score"], 4) == 0.7841
    assert round(step2_lr["performance"]["roc_auc"], 4) == 0.7339
    assert round(step2_lr["cv_metrics"]["cv_accuracy_mean"], 4) == 0.6979
    assert round(step2_lr["calibration"]["brier_score"], 4) == 0.1974
    assert round(step2_lr["calibration"]["ece"], 4) == 0.1254


def test_4_step2_selection_policy_alignment():
    """
    Verify that model selection evaluated on Step 2 candidates
    accurately reflects constraint compliance and selection status.
    """
    step2_evals = get_all_candidate_evaluations("final_project_loan_demo__1")
    sel_input = {k: v.get("raw_eval", v) for k, v in step2_evals.items()}
    sel_res = select_best_candidate_model(sel_input)

    assert sel_res["selected_model"] == "logistic_regression"
    assert sel_res["selection_status"] == "NO_MODEL_SATISFIES_CONSTRAINTS_FALLBACK"
    assert "highest CV accuracy: 0.6979" in sel_res["reason"]


def test_5_strict_candidate_dataset_isolation():
    """
    Verify that candidate evaluations for loan vs adult maintain strict data and feature isolation.
    """
    loan_evals = get_all_candidate_evaluations("final_project_loan_demo__1")
    adult_evals = get_all_candidate_evaluations("adult_census_income")

    loan_acc = loan_evals["logistic_regression"]["performance"]["accuracy"]
    adult_acc = adult_evals["logistic_regression"]["performance"]["accuracy"]

    assert loan_acc != adult_acc
    assert round(loan_acc, 4) == 0.6833
    assert round(adult_acc, 4) == 0.8399
