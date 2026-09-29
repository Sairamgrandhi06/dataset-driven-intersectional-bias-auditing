"""
Fairness Monitoring Module.

Monitors single-attribute and intersectional fairness disparity drift on new monitoring datasets.
Respects statistical safety rules, minimum group sizes, and handles missing protected attributes or labels.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.fairness.single_attribute import run_single_attribute_audits
from src.fairness.intersectional import audit_intersectional_attributes
from src.models.classifier import predict_model


def monitor_fairness_drift(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    A_test: pd.DataFrame,
    reference_fairness: Dict[str, Any],
    min_group_size: int = 30,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Monitor fairness metrics on current monitoring data and compare against reference model baseline.

    Parameters:
        model: Trained scikit-learn estimator model object.
        X_test (pd.DataFrame): Monitoring feature matrix.
        y_test (pd.Series): Monitoring target labels.
        A_test (pd.DataFrame): Monitoring protected attributes DataFrame.
        reference_fairness (dict): Reference fairness disparities from baseline run.
        min_group_size (int): Minimum required sample size per protected subgroup.
        config (dict, optional): Monitoring config options.

    Returns:
        dict: Fairness monitoring report.
    """
    if y_test is None or y_test.dropna().empty:
        return {
            "status": "FAIRNESS_UNAVAILABLE",
            "reason": "Target labels unavailable for fairness evaluation."
        }

    if A_test is None or A_test.empty:
        return {
            "status": "FAIRNESS_UNAVAILABLE",
            "reason": "Protected attributes unavailable in monitoring dataset."
        }

    config = config or {}
    fair_cfg = config.get("fairness_drift", {})
    eod_warn_thresh = float(fair_cfg.get("eod_increase_warning", 0.05))
    eod_crit_thresh = float(fair_cfg.get("eod_increase_critical", 0.10))

    y_test_arr = y_test.dropna().to_numpy()
    y_pred, _ = predict_model(model, X_test)

    # Single attribute audits
    single_audits = run_single_attribute_audits(y_test_arr, y_pred, A_test)

    # Intersectional audit
    inter_audit = audit_intersectional_attributes(
        y_test_arr, y_pred, A_test,
        min_group_size=min_group_size
    )

    cov_status = inter_audit.get("coverage_status", {}).get("code", "COVERAGE_OK")
    if cov_status in ["INSUFFICIENT_GROUP_COVERAGE", "INSUFFICIENT_COVERAGE"]:
        return {
            "status": "INSUFFICIENT_GROUP_COVERAGE",
            "coverage_status": inter_audit.get("coverage_status"),
            "single_attribute_metrics": single_audits,
            "intersectional_metrics": inter_audit,
            "fairness_metrics": inter_audit.get("disparities", {})
        }

    cur_eod = inter_audit.get("disparities", {}).get("equalized_odds_difference")
    ref_eod = reference_fairness.get("equalized_odds_difference")

    if cur_eod is None or ref_eod is None:
        status = "FAIRNESS_STABLE"
        eod_delta = None
    else:
        eod_delta = float(cur_eod - ref_eod)
        if eod_delta >= eod_crit_thresh:
            status = "FAIRNESS_DEGRADED"
        elif eod_delta >= eod_warn_thresh:
            status = "FAIRNESS_WARNING"
        elif eod_delta <= -0.02:
            status = "FAIRNESS_IMPROVED"
        else:
            status = "FAIRNESS_STABLE"

    return {
        "status": status,
        "current_fairness": inter_audit.get("disparities", {}),
        "reference_fairness": reference_fairness,
        "eod_delta": eod_delta,
        "single_attribute_metrics": single_audits,
        "intersectional_metrics": inter_audit
    }
