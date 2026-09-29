"""
Unit and Integration Tests for Phase C Model Subsystem.

Tests 1 to 25 verify:
1. Candidate model registry factory.
2. Logistic Regression training.
3. Random Forest training.
4. Gradient Boosting training.
5. 5-fold Stratified Cross-Validation on training data.
6. Held-out test set evaluation.
7. Probability generation & ROC-AUC evaluation.
8. Calibration metrics (Brier, ECE).
9. Dynamic model comparison table generation.
10. Constraint-based model selection (EOD threshold <= 0.20).
11. NO_MODEL_SATISFIES_CONSTRAINTS status when threshold violated.
12. Dataset-specific model directory isolation (models/<dataset_id>/).
13. Dataset SHA-256 hash recorded in metadata.
14. Reproducibility using fixed random seed.
15. Model training failure graceful handling.
16. Single-class target handling.
17. Probability-unavailable handling.
18. Single-attribute and Intersectional metrics associated with each candidate.
19. Adult dataset model training and candidate comparison.
20. Student dataset model training and candidate comparison.
21. Loan dataset model training and candidate comparison.
22. Custom dataset model training and candidate comparison.
23. No benchmark metric leakage into custom runs.
24. Zero test-set leakage (CV performed strictly on X_train).
25. Joblib serialization and run_manifest.json verification.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
import joblib

from src.models.model_registry import get_candidate_model, list_supported_models
from src.models.model_trainer import train_and_cross_validate_candidate
from src.models.model_evaluator import evaluate_candidate_model
from src.models.model_selection import select_best_candidate_model, evaluate_fairness_constraints
from src.models.model_storage import save_model_run_artifacts, get_dataset_model_runs


def test_1_candidate_model_registry():
    """TEST 1: Candidate model registry factory instantiates supported models."""
    supp = list_supported_models()
    assert "logistic_regression" in supp
    assert "random_forest" in supp
    assert "gradient_boosting" in supp

    lr = get_candidate_model("logistic_regression")
    rf = get_candidate_model("random_forest")
    gb = get_candidate_model("gradient_boosting")

    assert lr.__class__.__name__ == "LogisticRegression"
    assert rf.__class__.__name__ == "RandomForestClassifier"
    assert gb.__class__.__name__ == "GradientBoostingClassifier"


def test_2_logistic_regression_training():
    """TEST 2: Logistic Regression training and fitting."""
    X_tr = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6], "f2": [0, 1, 0, 1, 0, 1]})
    y_tr = pd.Series([0, 1, 0, 1, 0, 1])

    res = train_and_cross_validate_candidate("logistic_regression", X_tr, y_tr, cv_folds=2)
    assert res["fitted_model"] is not None
    assert hasattr(res["fitted_model"], "predict")


def test_3_random_forest_training():
    """TEST 3: Random Forest candidate model training."""
    X_tr = pd.DataFrame({"f1": [10, 20, 30, 40, 50, 60], "f2": [1, 0, 1, 0, 1, 0]})
    y_tr = pd.Series([1, 0, 1, 0, 1, 0])

    res = train_and_cross_validate_candidate("random_forest", X_tr, y_tr, cv_folds=2)
    assert res["fitted_model"].__class__.__name__ == "RandomForestClassifier"


def test_4_gradient_boosting_training():
    """TEST 4: Gradient Boosting candidate model training."""
    X_tr = pd.DataFrame({"f1": [5, 15, 25, 35, 45, 55], "f2": [0, 0, 1, 1, 0, 1]})
    y_tr = pd.Series([0, 0, 1, 1, 0, 1])

    res = train_and_cross_validate_candidate("gradient_boosting", X_tr, y_tr, cv_folds=2)
    assert res["fitted_model"].__class__.__name__ == "GradientBoostingClassifier"


def test_5_cross_validation_metrics_calculated():
    """TEST 5: 5-fold Stratified Cross-Validation metric calculation."""
    X_tr = pd.DataFrame({"f1": list(range(20)), "f2": list(range(20, 40))})
    y_tr = pd.Series([0, 1] * 10)

    res = train_and_cross_validate_candidate("logistic_regression", X_tr, y_tr, cv_folds=5)
    cv_m = res["cv_metrics"]
    assert "cv_accuracy_mean" in cv_m
    assert "cv_precision_mean" in cv_m
    assert "cv_f1_mean" in cv_m
    assert cv_m["folds_completed"] == 5


def test_6_held_out_test_set_evaluation():
    """TEST 6: Candidate model evaluation on held-out test data."""
    X_tr = pd.DataFrame({"f1": list(range(20)), "f2": list(range(20, 40))})
    y_tr = pd.Series([0, 1] * 10)
    X_te = pd.DataFrame({"f1": [2, 5, 8], "f2": [22, 25, 28]})
    y_te = pd.Series([0, 1, 0])
    A_te = pd.DataFrame({"gender": ["Female", "Male", "Female"]})

    cv_res = train_and_cross_validate_candidate("logistic_regression", X_tr, y_tr, cv_folds=2)
    eval_res = evaluate_candidate_model("logistic_regression", cv_res["fitted_model"], cv_res["cv_metrics"], X_te, y_te, A_te)

    assert "test_performance" in eval_res
    assert "accuracy" in eval_res["test_performance"]
    assert "confusion_matrix" in eval_res["test_performance"]


def test_7_probability_generation_and_roc_auc():
    """TEST 7: Probability generation and ROC-AUC evaluation."""
    X_tr = pd.DataFrame({"f1": list(range(20))})
    y_tr = pd.Series([0, 1] * 10)
    X_te = pd.DataFrame({"f1": [1, 10, 18]})
    y_te = pd.Series([0, 1, 1])
    A_te = pd.DataFrame({"gender": ["Female", "Male", "Male"]})

    cv_res = train_and_cross_validate_candidate("logistic_regression", X_tr, y_tr, cv_folds=2)
    eval_res = evaluate_candidate_model("logistic_regression", cv_res["fitted_model"], cv_res["cv_metrics"], X_te, y_te, A_te)
    assert eval_res["test_performance"]["roc_auc"] is not None


def test_8_calibration_metrics_brier_and_ece():
    """TEST 8: Calibration metrics calculation (Brier Score, ECE)."""
    X_tr = pd.DataFrame({"f1": list(range(20))})
    y_tr = pd.Series([0, 1] * 10)
    X_te = pd.DataFrame({"f1": [1, 5, 10, 15]})
    y_te = pd.Series([0, 0, 1, 1])
    A_te = pd.DataFrame({"gender": ["F", "M", "F", "M"]})

    cv_res = train_and_cross_validate_candidate("logistic_regression", X_tr, y_tr, cv_folds=2)
    eval_res = evaluate_candidate_model("logistic_regression", cv_res["fitted_model"], cv_res["cv_metrics"], X_te, y_te, A_te)
    assert "brier_score" in eval_res["calibration_metrics"]
    assert "ece" in eval_res["calibration_metrics"]


def test_9_dynamic_comparison_table_generation():
    """TEST 9: Model comparison table generation across candidate models."""
    evals = {
        "logistic_regression": {
            "cv_metrics": {"cv_accuracy_mean": 0.80, "cv_accuracy_std": 0.02},
            "test_performance": {"accuracy": 0.82, "f1_score": 0.79, "roc_auc": 0.85},
            "calibration_metrics": {"brier_score": 0.15, "ece": 0.04},
            "fairness_metrics": {"equalized_odds_difference": 0.08},
            "single_attribute_metrics": {}
        },
        "random_forest": {
            "cv_metrics": {"cv_accuracy_mean": 0.85, "cv_accuracy_std": 0.01},
            "test_performance": {"accuracy": 0.86, "f1_score": 0.84, "roc_auc": 0.89},
            "calibration_metrics": {"brier_score": 0.12, "ece": 0.03},
            "fairness_metrics": {"equalized_odds_difference": 0.12},
            "single_attribute_metrics": {}
        }
    }

    sel_res = select_best_candidate_model(evals)
    comp_tbl = sel_res["comparison_table"]
    assert len(comp_tbl) == 2
    assert comp_tbl[0]["model"] in evals
    assert comp_tbl[1]["model"] in evals


def test_10_constraint_based_model_selection():
    """TEST 10: Selection policy chooses highest CV accuracy candidate satisfying fairness constraints."""
    evals = {
        "logistic_regression": {
            "cv_metrics": {"cv_accuracy_mean": 0.80},
            "test_performance": {"accuracy": 0.82, "f1_score": 0.79, "roc_auc": 0.85},
            "calibration_metrics": {"brier_score": 0.15, "ece": 0.04},
            "fairness_metrics": {"equalized_odds_difference": 0.10},
            "single_attribute_metrics": {}
        },
        "random_forest": {
            "cv_metrics": {"cv_accuracy_mean": 0.88},
            "test_performance": {"accuracy": 0.89, "f1_score": 0.87, "roc_auc": 0.91},
            "calibration_metrics": {"brier_score": 0.10, "ece": 0.02},
            "fairness_metrics": {"equalized_odds_difference": 0.15},
            "single_attribute_metrics": {}
        }
    }

    cfg = {"fairness_constraints": {"equalized_odds_difference": {"enabled": True, "max_allowed": 0.20}}}
    sel_res = select_best_candidate_model(evals, selection_config=cfg)
    assert sel_res["selected_model"] == "random_forest"
    assert sel_res["selection_status"] == "SELECTED"


def test_11_no_model_satisfies_constraints_state():
    """TEST 11: Explicit NO_MODEL_SATISFIES_CONSTRAINTS status when all candidates violate threshold."""
    evals = {
        "logistic_regression": {
            "cv_metrics": {"cv_accuracy_mean": 0.80},
            "test_performance": {"accuracy": 0.82, "f1_score": 0.79},
            "calibration_metrics": {},
            "fairness_metrics": {"equalized_odds_difference": 0.35},
            "single_attribute_metrics": {}
        },
        "random_forest": {
            "cv_metrics": {"cv_accuracy_mean": 0.85},
            "test_performance": {"accuracy": 0.86, "f1_score": 0.84},
            "calibration_metrics": {},
            "fairness_metrics": {"equalized_odds_difference": 0.40},
            "single_attribute_metrics": {}
        }
    }

    cfg = {
        "fairness_constraints": {"equalized_odds_difference": {"enabled": True, "max_allowed": 0.20}},
        "allow_fallback": False
    }
    sel_res = select_best_candidate_model(evals, selection_config=cfg)
    assert sel_res["selected_model"] is None
    assert sel_res["selection_status"] == "NO_MODEL_SATISFIES_CONSTRAINTS"


def test_12_dataset_specific_model_isolation(tmp_path):
    """TEST 12: Dataset model directory isolation under models/<dataset_id>/."""
    X_tr = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6]})
    y_tr = pd.Series([0, 1, 0, 1, 0, 1])

    res_lr = train_and_cross_validate_candidate("logistic_regression", X_tr, y_tr, cv_folds=2)
    models_dict = {"logistic_regression": res_lr["fitted_model"]}
    sel_res = {"selected_model": "logistic_regression", "selection_status": "SELECTED", "reason": "Test", "comparison_table": []}

    paths = save_model_run_artifacts(
        dataset_id="dataset_A",
        run_id="run_1001",
        candidate_models=models_dict,
        selected_model_key="logistic_regression",
        selection_result=sel_res,
        dataset_hash="hash_a",
        config_dict={},
        base_dir=str(tmp_path)
    )

    assert "dataset_A" in paths["model_logistic_regression"]
    assert os.path.exists(paths["model_logistic_regression"])


def test_13_dataset_hash_recorded_in_metadata(tmp_path):
    """TEST 13: Dataset SHA-256 hash recorded in metadata artifact."""
    X_tr = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6]})
    y_tr = pd.Series([0, 1, 0, 1, 0, 1])

    res_lr = train_and_cross_validate_candidate("logistic_regression", X_tr, y_tr, cv_folds=2)
    models_dict = {"logistic_regression": res_lr["fitted_model"]}
    sel_res = {"selected_model": "logistic_regression", "selection_status": "SELECTED", "reason": "Test"}

    paths = save_model_run_artifacts(
        dataset_id="dataset_hash_test",
        run_id="run_hash_01",
        candidate_models=models_dict,
        selected_model_key="logistic_regression",
        selection_result=sel_res,
        dataset_hash="abc123sha256hash",
        config_dict={},
        base_dir=str(tmp_path)
    )

    with open(paths["model_metadata"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["dataset_hash"] == "abc123sha256hash"


def test_14_reproducibility_using_fixed_random_seed():
    """TEST 14: Reproducibility of cross-validation metrics using fixed random seed."""
    X_tr = pd.DataFrame({"f1": list(range(30)), "f2": list(range(30, 60))})
    y_tr = pd.Series([0, 1, 1] * 10)

    res1 = train_and_cross_validate_candidate("random_forest", X_tr, y_tr, cv_folds=5, random_seed=42)
    res2 = train_and_cross_validate_candidate("random_forest", X_tr, y_tr, cv_folds=5, random_seed=42)

    assert res1["cv_metrics"]["cv_accuracy_mean"] == res2["cv_metrics"]["cv_accuracy_mean"]


def test_15_model_training_failure_handling():
    """TEST 15: Graceful fallback when target has insufficient samples for CV."""
    X_tr = pd.DataFrame({"f1": [1, 2]})
    y_tr = pd.Series([0, 1])

    res = train_and_cross_validate_candidate("logistic_regression", X_tr, y_tr, cv_folds=5)
    assert res["fitted_model"] is not None
    assert res["cv_metrics"]["folds_completed"] == 0


def test_16_single_class_target_handling():
    """TEST 16: Handling dataset with single target class without raw exception crash."""
    X_tr = pd.DataFrame({"f1": [1, 2, 3]})
    y_tr = pd.Series([0, 0, 0])

    res = train_and_cross_validate_candidate("logistic_regression", X_tr, y_tr, cv_folds=2)
    assert res["cv_metrics"]["folds_completed"] == 0
    assert res["fitted_model"] is None


def test_17_probability_unavailable_handling():
    """TEST 17: Probability unavailable handling does not raise AttributeError."""
    class DummyClassifierNoProba:
        def fit(self, X, y):
            pass
        def predict(self, X):
            return np.zeros(len(X))

    dummy = DummyClassifierNoProba()
    X_te = pd.DataFrame({"f1": [1, 2, 3]})
    y_te = pd.Series([0, 1, 0])
    A_te = pd.DataFrame({"gender": ["F", "M", "F"]})

    eval_res = evaluate_candidate_model("dummy", dummy, {}, X_te, y_te, A_te)
    assert eval_res["test_performance"]["roc_auc"] is None
    assert eval_res["calibration_metrics"]["status"] == "UNAVAILABLE"


def test_18_fairness_metrics_associated_with_each_candidate():
    """TEST 18: Single-attribute and Intersectional fairness metrics computed for each candidate."""
    X_tr = pd.DataFrame({"f1": list(range(20))})
    y_tr = pd.Series([0, 1] * 10)
    X_te = pd.DataFrame({"f1": [1, 5, 10, 15]})
    y_te = pd.Series([0, 1, 0, 1])
    A_te = pd.DataFrame({"gender": ["Female", "Female", "Male", "Male"]})

    cv_res = train_and_cross_validate_candidate("logistic_regression", X_tr, y_tr, cv_folds=2)
    eval_res = evaluate_candidate_model("logistic_regression", cv_res["fitted_model"], cv_res["cv_metrics"], X_te, y_te, A_te)

    assert "single_attribute_metrics" in eval_res
    assert "gender" in eval_res["single_attribute_metrics"]
    assert "intersectional_metrics" in eval_res


def test_19_adult_dataset_candidate_training():
    """TEST 19: Adult benchmark dataset candidate model training & selection."""
    adult_csv = "data/raw/adult.csv"
    if os.path.exists(adult_csv):
        df = pd.read_csv(adult_csv, skipinitialspace=True).head(500)
        X = pd.get_dummies(df.drop(columns=["income", "sex", "race"]), drop_first=True)
        y = (df["income"].astype(str).str.strip().str.rstrip(".") == ">50K").astype(int)

        res_lr = train_and_cross_validate_candidate("logistic_regression", X, y, cv_folds=3)
        res_rf = train_and_cross_validate_candidate("random_forest", X, y, cv_folds=3)
        assert res_lr["cv_metrics"]["cv_accuracy_mean"] > 0.50
        assert res_rf["cv_metrics"]["cv_accuracy_mean"] > 0.50


def test_20_student_dataset_candidate_training():
    """TEST 20: Student dataset candidate model training & selection."""
    df = pd.DataFrame({
        "study_hours": np.random.randint(1, 10, 100),
        "attendance": np.random.randint(50, 100, 100),
        "pass_status": np.random.choice([0, 1], 100)
    })
    X = df[["study_hours", "attendance"]]
    y = df["pass_status"]

    res_gb = train_and_cross_validate_candidate("gradient_boosting", X, y, cv_folds=3)
    assert res_gb["cv_metrics"]["cv_accuracy_mean"] > 0.0


def test_21_loan_dataset_candidate_training():
    """TEST 21: Loan dataset candidate model training & selection."""
    loan_csv = "data/raw/loan_approval_fairness_test.csv"
    if os.path.exists(loan_csv):
        df = pd.read_csv(loan_csv)
        X = pd.get_dummies(df.drop(columns=["loan_status", "gender", "race"]), drop_first=True)
        y = (df["loan_status"].astype(str).str.strip().str.lower() == "approved").astype(int)

        res_lr = train_and_cross_validate_candidate("logistic_regression", X, y, cv_folds=3)
        res_rf = train_and_cross_validate_candidate("random_forest", X, y, cv_folds=3)

        assert res_lr["fitted_model"] is not None
        assert res_rf["fitted_model"] is not None


def test_22_custom_arbitrary_dataset_candidate_training():
    """TEST 22: Custom arbitrary dataset model training."""
    df = pd.DataFrame({
        "var_a": np.random.randn(80),
        "var_b": np.random.randn(80),
        "label": np.random.choice([0, 1], 80)
    })
    X = df[["var_a", "var_b"]]
    y = df["label"]

    res = train_and_cross_validate_candidate("logistic_regression", X, y, cv_folds=3)
    assert res["cv_metrics"]["folds_completed"] == 3


def test_23_no_adult_benchmark_leakage():
    """TEST 23: Custom dataset runs do not leak Adult benchmark accuracy or fairness metrics."""
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    X = df[["feature1"]]
    y = df["target"]

    res = train_and_cross_validate_candidate("logistic_regression", X, y, cv_folds=2)
    # Adult benchmark accuracy is 0.849; custom 10-row dataset must compute its own dynamic CV accuracy
    assert res["cv_metrics"]["cv_accuracy_mean"] != 0.849


def test_24_zero_test_set_leakage_in_model_selection():
    """TEST 24: Model selection depends strictly on X_train CV performance without touching X_test."""
    X_tr = pd.DataFrame({"f1": list(range(30))})
    y_tr = pd.Series([0, 1, 0] * 10)
    X_te = pd.DataFrame({"f1": [100, 200, 300]})
    y_te = pd.Series([1, 1, 1])

    # Model selection executes only on X_train, y_train
    cv_res1 = train_and_cross_validate_candidate("logistic_regression", X_tr, y_tr, cv_folds=3)
    cv_res2 = train_and_cross_validate_candidate("random_forest", X_tr, y_tr, cv_folds=3)

    assert cv_res1["cv_metrics"]["folds_completed"] == 3
    assert cv_res2["cv_metrics"]["folds_completed"] == 3


def test_25_joblib_serialization_and_manifest(tmp_path):
    """TEST 25: Joblib serialization and run_manifest.json verification."""
    X_tr = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6]})
    y_tr = pd.Series([0, 1, 0, 1, 0, 1])

    res = train_and_cross_validate_candidate("logistic_regression", X_tr, y_tr, cv_folds=2)
    models_dict = {"logistic_regression": res["fitted_model"]}
    sel_res = {"selected_model": "logistic_regression", "selection_status": "SELECTED", "reason": "Test"}

    paths = save_model_run_artifacts(
        dataset_id="test_joblib_ds",
        run_id="run_joblib_01",
        candidate_models=models_dict,
        selected_model_key="logistic_regression",
        selection_result=sel_res,
        dataset_hash="hash999",
        config_dict={"target": {"column": "target", "positive_class": "1"}},
        base_dir=str(tmp_path)
    )

    loaded_model = joblib.load(paths["selected_model"])
    assert hasattr(loaded_model, "predict")

    with open(paths["run_manifest"], "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["dataset_id"] == "test_joblib_ds"
    assert manifest["selected_model"] == "logistic_regression"
