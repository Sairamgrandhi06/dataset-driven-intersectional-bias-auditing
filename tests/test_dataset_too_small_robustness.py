"""
Regression tests for dataset preflight validation, model selection safeguards,
and ultra-small / untrainable dataset handling.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

from src.models.model_trainer import (
    validate_training_data_preflight,
    train_and_cross_validate_candidate
)
from src.models.model_selection import select_best_candidate_model
from src.models.classifier import predict_model
from src.run_dataset import run_dataset_pipeline
from src.models.retraining import retrain_dataset


def test_1_two_row_dataset_preflight():
    """TEST 1: 2-row dataset triggers DATASET_TOO_SMALL_FOR_TRAINING preflight error."""
    X_train = pd.DataFrame({"f1": [1.0]})
    y_train = pd.Series([1])
    X_test = pd.DataFrame({"f1": [2.0]})
    y_test = pd.Series([0])

    is_valid, report = validate_training_data_preflight(X_train, y_train, X_test, y_test, cv_folds=5)

    assert is_valid is False
    assert report["status"] == "DATASET_TOO_SMALL_FOR_TRAINING"
    assert report["actual_rows"] == 2
    assert report["train_rows"] == 1
    assert report["test_rows"] == 1
    assert report["cv_folds"] == 5
    assert report["required_minimum"] >= 10


def test_2_one_training_sample_preflight():
    """TEST 2: 1 training sample triggers DATASET_TOO_SMALL_FOR_TRAINING preflight error."""
    X_train = pd.DataFrame({"f1": [1.0]})
    y_train = pd.Series([1])
    X_test = pd.DataFrame({"f1": [2.0], "f2": [3.0]})
    y_test = pd.Series([0])

    is_valid, report = validate_training_data_preflight(X_train, y_train, X_test, y_test, cv_folds=5)

    assert is_valid is False
    assert report["status"] == "DATASET_TOO_SMALL_FOR_TRAINING"
    assert report["train_rows"] == 1


def test_3_single_class_target_preflight():
    """TEST 3: Single-class target set triggers target class preflight error."""
    X_train = pd.DataFrame({"f1": range(10)})
    y_train = pd.Series([1] * 10)  # Only 1 class
    X_test = pd.DataFrame({"f1": range(2)})
    y_test = pd.Series([1, 1])

    is_valid, report = validate_training_data_preflight(X_train, y_train, X_test, y_test, cv_folds=5)

    assert is_valid is False
    assert report["status"] == "DATASET_TOO_SMALL_FOR_TRAINING"
    assert report["target_classes_found"] == 1


def test_4_insufficient_samples_for_5fold_cv():
    """TEST 4: Insufficient samples per class for 5-fold CV triggers preflight error."""
    X_train = pd.DataFrame({"f1": range(6)})
    y_train = pd.Series([0, 0, 0, 0, 1, 1])  # Class '1' has only 2 samples (< 5 needed for 5-fold SKF)
    X_test = pd.DataFrame({"f1": range(2)})
    y_test = pd.Series([0, 1])

    is_valid, report = validate_training_data_preflight(X_train, y_train, X_test, y_test, cv_folds=5)

    assert is_valid is False
    assert report["status"] == "DATASET_TOO_SMALL_FOR_TRAINING"
    assert "samples per class are required" in report["message"]


def test_5_candidate_training_failure_handling():
    """TEST 5: Single candidate training failure returns is_fitted=False without raising exception."""
    X_train = pd.DataFrame({"f1": [1.0]})
    y_train = pd.Series([1])

    res = train_and_cross_validate_candidate("logistic_regression", X_train, y_train, cv_folds=5)

    assert res["model_key"] == "logistic_regression"
    assert res["is_fitted"] is False
    assert res["fitted_model"] is None
    assert res["failure_reason"] is not None


def test_6_all_candidates_failing_selection():
    """TEST 6: Selection returns NO_VALID_MODEL and selected_model=None when all candidates fail."""
    failed_evals = {
        "logistic_regression": {
            "fitted_model": None,
            "is_fitted": False,
            "failure_reason": "Insufficient samples",
            "cv_metrics": {"cv_accuracy_mean": 0.0},
            "test_performance": {},
            "fairness_metrics": {}
        },
        "random_forest": {
            "fitted_model": None,
            "is_fitted": False,
            "failure_reason": "Insufficient samples",
            "cv_metrics": {"cv_accuracy_mean": 0.0},
            "test_performance": {},
            "fairness_metrics": {}
        }
    }

    selection_res = select_best_candidate_model(failed_evals)

    assert selection_res["selected_model"] is None
    assert selection_res["selection_status"] == "NO_VALID_MODEL"
    assert len(selection_res["comparison_table"]) == 2
    assert selection_res["comparison_table"][0]["status"] == "FAIL_TRAINING"


def test_7_predict_model_never_receives_none():
    """TEST 7: predict_model raises explicit ValueError if passed None instead of throwing AttributeError."""
    with pytest.raises((ValueError, TypeError, AttributeError)):
        predict_model(None, pd.DataFrame({"f1": [1.0]}))


def test_8_run_dataset_pipeline_interactive_small_dataset(tmp_path):
    """TEST 8: Full pipeline execution on test_interactive_ds returns DATASET_TOO_SMALL_FOR_TRAINING without traceback."""
    cfg_path = "config/test_interactive_ds_config.json"
    if not os.path.exists(cfg_path):
        pytest.skip("test_interactive_ds_config.json file missing.")

    result = run_dataset_pipeline(cfg_path)

    assert result is not None
    assert result["status"] == "DATASET_TOO_SMALL_FOR_TRAINING"
    assert result["selected_model"] is None
    assert result["actual_rows"] == 2
    assert result["train_rows"] == 1
    assert result["test_rows"] == 1


def test_9_retrain_dataset_small_dataset(tmp_path):
    """TEST 9: retrain_dataset on small dataset returns DATASET_TOO_SMALL_FOR_TRAINING and registers FAILED run."""
    cfg = {
        "dataset_id": "test_small_retrain",
        "path": "data/raw/test_interactive_ds.csv" if os.path.exists("data/raw/test_interactive_ds.csv") else "dummy.csv",
        "target": {"column": "passed", "positive_class": "yes"},
        "protected_attributes": ["gender"],
        "id_columns": ["student_id"],
        "test_size": 0.5,
        "random_state": 42
    }

    if os.path.exists(cfg["path"]):
        res = retrain_dataset("test_small_retrain", cfg, base_dir=str(tmp_path))
        assert res["status"] == "DATASET_TOO_SMALL_FOR_TRAINING"
        assert res["selected_model"] is None
        assert res["model_version"] == "NONE"
        assert res["model_status"] == "FAILED"
