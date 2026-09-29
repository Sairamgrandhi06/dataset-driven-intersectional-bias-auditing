"""
Model Performance Monitoring Module.

Evaluates trained baseline/selected model accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
on new monitoring batch data. Calculates performance metric deltas against reference baseline and reports status.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.models.classifier import predict_model


def monitor_performance_drift(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    reference_metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate model performance on current monitoring dataset and compare against reference metrics.

    Parameters:
        model: Trained scikit-learn estimator model object.
        X_test (pd.DataFrame): Monitoring feature matrix.
        y_test (pd.Series): Monitoring target labels.
        reference_metrics (dict): Baseline performance metrics dictionary from model run.
        config (dict, optional): Performance monitoring configuration options.

    Returns:
        dict: Performance monitoring report.
    """
    if y_test is None or y_test.dropna().empty:
        return {
            "status": "PERFORMANCE_UNAVAILABLE",
            "labels_available": False,
            "message": "Ground-truth target labels are unavailable in monitoring dataset."
        }

    config = config or {}
    perf_cfg = config.get("performance_drift", {})
    f1_warn_thresh = float(perf_cfg.get("f1_degradation_warning", 0.05))
    f1_crit_thresh = float(perf_cfg.get("f1_degradation_critical", 0.10))

    y_test_arr = y_test.dropna().to_numpy()
    if len(y_test_arr) == 0 or model is None:
        return {
            "status": "PERFORMANCE_UNAVAILABLE",
            "labels_available": False,
            "message": "Invalid test labels or missing model estimator."
        }

    y_pred, y_prob = predict_model(model, X_test)

    acc = float(accuracy_score(y_test_arr, y_pred))
    prec = float(precision_score(y_test_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_test_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_test_arr, y_pred, zero_division=0))

    roc_auc = None
    if y_prob is not None and len(np.unique(y_test_arr)) == 2:
        try:
            roc_auc = float(roc_auc_score(y_test_arr, y_prob))
        except Exception:
            pass

    cm = confusion_matrix(y_test_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    cur_perf = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    }

    # Extract reference metrics
    ref_acc = float(reference_metrics.get("accuracy", 0.0) or 0.0)
    ref_prec = float(reference_metrics.get("precision", 0.0) or 0.0)
    ref_rec = float(reference_metrics.get("recall", 0.0) or 0.0)
    ref_f1 = float(reference_metrics.get("f1_score", 0.0) or 0.0)

    acc_delta = (acc - ref_acc) if ref_acc > 0 else 0.0
    acc_pct_change = ((acc - ref_acc) / abs(ref_acc) * 100.0) if ref_acc > 0 else 0.0

    prec_delta = (prec - ref_prec) if ref_prec > 0 else 0.0
    prec_pct_change = ((prec - ref_prec) / abs(ref_prec) * 100.0) if ref_prec > 0 else 0.0

    rec_delta = (rec - ref_rec) if ref_rec > 0 else 0.0
    rec_pct_change = ((rec - ref_rec) / abs(ref_rec) * 100.0) if ref_rec > 0 else 0.0

    f1_delta = (f1 - ref_f1) if ref_f1 > 0 else 0.0
    f1_pct_change = ((f1 - ref_f1) / abs(ref_f1) * 100.0) if ref_f1 > 0 else 0.0

    # Determine performance drift status
    if f1_delta <= -f1_crit_thresh:
        status = "DEGRADED"
    elif f1_delta <= -f1_warn_thresh:
        status = "WARNING"
    elif f1_delta >= 0.02:
        status = "IMPROVED"
    else:
        status = "STABLE"

    return {
        "status": status,
        "labels_available": True,
        "current_metrics": cur_perf,
        "reference_metrics": reference_metrics,
        "deltas": {
            "accuracy_delta": acc_delta,
            "accuracy_pct_change": acc_pct_change,
            "precision_delta": prec_delta,
            "precision_pct_change": prec_pct_change,
            "recall_delta": rec_delta,
            "recall_pct_change": rec_pct_change,
            "f1_score_delta": f1_delta,
            "f1_score_pct_change": f1_pct_change
        }
    }
