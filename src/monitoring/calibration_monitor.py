"""
Calibration Monitoring Module.

Evaluates Brier Score and Expected Calibration Error (ECE) drift on new monitoring datasets.
Reports CALIBRATION_UNAVAILABLE when ground-truth labels or predicted probabilities are missing.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from src.fairness.calibration import evaluate_calibration
from src.models.classifier import predict_model


def monitor_calibration_drift(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    reference_calibration: Dict[str, Any],
    n_bins: int = 10,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Monitor Brier score and ECE calibration drift on current monitoring dataset.

    Parameters:
        model: Trained scikit-learn estimator model object.
        X_test (pd.DataFrame): Monitoring feature matrix.
        y_test (pd.Series): Monitoring target labels.
        reference_calibration (dict): Reference calibration metrics.
        n_bins (int): Number of calibration probability bins.
        config (dict, optional): Calibration monitoring config options.

    Returns:
        dict: Calibration monitoring report.
    """
    if y_test is None or y_test.dropna().empty:
        return {
            "status": "CALIBRATION_UNAVAILABLE",
            "reason": "Target labels unavailable for calibration monitoring."
        }

    config = config or {}
    cal_cfg = config.get("calibration_drift", {})
    brier_warn_thresh = float(cal_cfg.get("brier_increase_warning", 0.05))
    ece_warn_thresh = float(cal_cfg.get("ece_increase_warning", 0.05))

    y_test_arr = y_test.dropna().to_numpy()
    _, y_prob = predict_model(model, X_test)

    if y_prob is None:
        return {
            "status": "CALIBRATION_UNAVAILABLE",
            "reason": "Predicted probabilities unavailable for model."
        }

    cur_calib = evaluate_calibration(y_test_arr, y_prob, n_bins=n_bins)

    ref_brier = reference_calibration.get("brier_score")
    cur_brier = cur_calib.get("brier_score")

    ref_ece = reference_calibration.get("ece", reference_calibration.get("expected_calibration_error"))
    cur_ece = cur_calib.get("ece", cur_calib.get("expected_calibration_error"))

    brier_delta = float(cur_brier - ref_brier) if (cur_brier is not None and ref_brier is not None) else None
    brier_pct_change = float((cur_brier - ref_brier) / abs(ref_brier) * 100.0) if (cur_brier is not None and ref_brier is not None and ref_brier > 0) else None

    ece_delta = float(cur_ece - ref_ece) if (cur_ece is not None and ref_ece is not None) else None
    ece_pct_change = float((cur_ece - ref_ece) / abs(ref_ece) * 100.0) if (cur_ece is not None and ref_ece is not None and ref_ece > 0) else None

    # Determine status (for Brier & ECE, lower is better)
    if (brier_delta is not None and brier_delta >= brier_warn_thresh) or (ece_delta is not None and ece_delta >= ece_warn_thresh):
        status = "DEGRADED"
    elif (brier_delta is not None and brier_delta >= 0.02) or (ece_delta is not None and ece_delta >= 0.02):
        status = "WARNING"
    elif (brier_delta is not None and brier_delta <= -0.02) or (ece_delta is not None and ece_delta <= -0.02):
        status = "IMPROVED"
    else:
        status = "STABLE"

    brier_status = "Degraded (Worsened)" if (brier_delta is not None and brier_delta > 0.001) else ("Improved" if (brier_delta is not None and brier_delta < -0.001) else "Stable")
    ece_status = "Degraded (Worsened)" if (ece_delta is not None and ece_delta > 0.001) else ("Improved" if (ece_delta is not None and ece_delta < -0.001) else "Stable")

    return {
        "status": status,
        "current_calibration": cur_calib,
        "reference_calibration": reference_calibration,
        "brier_delta": brier_delta,
        "brier_pct_change": brier_pct_change,
        "ece_delta": ece_delta,
        "ece_pct_change": ece_pct_change,
        "interpretation": {
            "brier_status": brier_status,
            "ece_status": ece_status
        }
    }
