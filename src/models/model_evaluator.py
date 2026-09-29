"""
Model Evaluator Engine.

Evaluates candidate model performance, calibration, single-attribute fairness, and intersectional
fairness on held-out test data (X_test, y_test, A_test).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.fairness.single_attribute import run_single_attribute_audits
from src.fairness.intersectional import audit_intersectional_attributes
from src.fairness.calibration import evaluate_calibration


def evaluate_candidate_model(
    model_key: str,
    fitted_model: Any,
    cv_metrics: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    A_test: pd.DataFrame,
    min_group_size: int = 30
) -> Dict[str, Any]:
    """
    Evaluate fitted candidate model on held-out test set.

    Parameters:
        model_key (str): Candidate identifier string.
        fitted_model (Any): Trained classifier instance.
        cv_metrics (dict): CV performance metrics on training data.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target labels.
        A_test (pd.DataFrame): Test protected attributes.
        min_group_size (int): Intersectional group threshold (N >= min_group_size).

    Returns:
        dict: Complete candidate model evaluation output.
    """
    if fitted_model is None:
        return {
            "model_key": model_key,
            "fitted_model": None,
            "is_fitted": False,
            "cv_metrics": cv_metrics,
            "test_performance": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "roc_auc": None, "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 0}},
            "calibration_metrics": {"status": "UNAVAILABLE", "brier_score": None, "ece": None},
            "single_attribute_metrics": {},
            "intersectional_metrics": {"coverage_status": {"code": "UNAVAILABLE"}},
            "fairness_metrics": {"equalized_odds_difference": None},
            "status": "FAILED"
        }

    y_test_arr = np.array(y_test)
    y_pred = fitted_model.predict(X_test)

    # Performance metrics
    acc = float(accuracy_score(y_test_arr, y_pred))
    prec = float(precision_score(y_test_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_test_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_test_arr, y_pred, zero_division=0))

    # ROC-AUC calculation
    roc_auc = None
    y_prob = None
    if hasattr(fitted_model, "predict_proba") and len(np.unique(y_test_arr)) == 2:
        try:
            probs = fitted_model.predict_proba(X_test)
            if probs.shape[1] == 2:
                y_prob = probs[:, 1]
                roc_auc = float(roc_auc_score(y_test_arr, y_prob))
        except Exception:
            pass

    # Confusion matrix
    cm = confusion_matrix(y_test_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    perf_metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    }

    # Calibration metrics
    cal_metrics = evaluate_calibration(y_test_arr, y_prob)

    # Single-attribute fairness audits
    single_audits = run_single_attribute_audits(y_test_arr, y_pred, A_test)

    # Intersectional fairness audit
    inter_audit = audit_intersectional_attributes(
        y_test_arr, y_pred, A_test, min_group_size=min_group_size
    )

    return {
        "model_key": model_key,
        "fitted_model": fitted_model,
        "is_fitted": True,
        "cv_metrics": cv_metrics,
        "test_performance": perf_metrics,
        "calibration_metrics": cal_metrics,
        "single_attribute_metrics": single_audits,
        "intersectional_metrics": inter_audit,
        "fairness_metrics": inter_audit["disparities"],
        "status": "EVALUATED"
    }
