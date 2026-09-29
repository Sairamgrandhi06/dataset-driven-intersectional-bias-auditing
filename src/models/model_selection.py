"""
Model Selection Engine.

Implements transparent, constraint-based model selection policies.
Rejects candidates violating configured fairness thresholds, ranks eligible candidates
by cross-validation performance, and explicitly reports NO_MODEL_SATISFIES_CONSTRAINTS when
violations occur.
"""

import json
import os
from typing import Dict, Any, List, Optional


def load_model_selection_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load model selection policy configuration file or return standard defaults."""
    default_config = {
        "fairness_constraints": {
            "equalized_odds_difference": {
                "enabled": True,
                "max_allowed": 0.20
            }
        },
        "performance_metric": "cv_accuracy",
        "direction": "maximize",
        "allow_fallback": True
    }

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                return {**default_config, **cfg}
        except Exception:
            pass

    # Check default file location
    def_file = os.path.join("config", "model_selection_config.json")
    if os.path.exists(def_file):
        try:
            with open(def_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                return {**default_config, **cfg}
        except Exception:
            pass

    return default_config


def evaluate_fairness_constraints(
    candidate_eval: Dict[str, Any],
    selection_config: Dict[str, Any]
) -> Tuple_Bool_Reason:
    """
    Check whether candidate model satisfies configured fairness constraints.

    Returns:
        (bool, str): (satisfies_all_constraints, reason_string)
    """
    constraints = selection_config.get("fairness_constraints", {})
    eod_cfg = constraints.get("equalized_odds_difference", {})

    if not eod_cfg.get("enabled", True):
        return True, "Fairness constraints disabled."

    max_allowed = float(eod_cfg.get("max_allowed", 0.20))

    # Check top-level fairness metrics if present
    top_fair = candidate_eval.get("fairness_metrics", {})
    top_eod = top_fair.get("equalized_odds_difference")
    if top_eod is not None and top_eod > max_allowed:
        return False, f"Equalized Odds Difference ({top_eod:.4f}) exceeds threshold ({max_allowed:.4f})."

    # Check single-attribute audits
    single_audits = candidate_eval.get("single_attribute_metrics", {})
    for attr, audit in single_audits.items():
        disp = audit.get("disparities", {})
        eod = disp.get("equalized_odds_difference")
        if eod is not None and eod > max_allowed:
            return False, f"Single-attribute '{attr}' Equalized Odds Difference ({eod:.4f}) exceeds threshold ({max_allowed:.4f})."

    # Check intersectional audits if estimable
    inter_audit = candidate_eval.get("intersectional_metrics", {})
    cov_status = inter_audit.get("coverage_status", {})
    if cov_status.get("code") == "AVAILABLE":
        inter_disp = inter_audit.get("disparities", {})
        inter_eod = inter_disp.get("equalized_odds_difference")
        if inter_eod is not None and inter_eod > max_allowed:
            return False, f"Intersectional Equalized Odds Difference ({inter_eod:.4f}) exceeds threshold ({max_allowed:.4f})."

    return True, f"Satisfies all configured fairness constraints (EOD <= {max_allowed:.2f})."


# Simple tuple type alias for annotation
Tuple_Bool_Reason = Any


def select_best_candidate_model(
    candidate_evaluations: Dict[str, Dict[str, Any]],
    selection_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Select transparent candidate model based on cross-validation performance and fairness constraints.

    Parameters:
        candidate_evaluations (dict): Map of model_key -> candidate evaluation dict.
        selection_config (dict, optional): Selection policy config.

    Returns:
        dict: Model selection result containing comparison table and selection status.
    """
    if not selection_config:
        selection_config = load_model_selection_config()

    perf_metric_key = selection_config.get("performance_metric", "cv_accuracy")
    allow_fallback = selection_config.get("allow_fallback", True)

    comparison_table = []
    satisfied_candidates = []
    all_candidates = []

    for model_key, eval_data in candidate_evaluations.items():
        cv_metrics = eval_data.get("cv_metrics", {})
        test_perf = eval_data.get("test_performance", {})
        cal_metrics = eval_data.get("calibration_metrics", {})
        fair_metrics = eval_data.get("fairness_metrics", {})

        has_fitted_key = "fitted_model" in eval_data
        if has_fitted_key:
            fitted_obj = eval_data.get("fitted_model")
            is_fitted = eval_data.get("is_fitted", fitted_obj is not None) and (fitted_obj is not None)
        else:
            fitted_obj = "mock_fitted_model"
            is_fitted = (eval_data.get("status") != "FAILED") and (eval_data.get("is_fitted", True))

        cv_acc = cv_metrics.get("cv_accuracy_mean", 0.0)
        test_acc = test_perf.get("accuracy", 0.0)
        f1 = test_perf.get("f1_score", 0.0)
        roc_auc = test_perf.get("roc_auc")
        brier = cal_metrics.get("brier_score")
        ece = cal_metrics.get("ece")
        eod = fair_metrics.get("equalized_odds_difference")

        if not is_fitted:
            satisfies = False
            reason = eval_data.get("failure_reason") or "Model fitting failed during training or CV."
        else:
            satisfies, reason = evaluate_fairness_constraints(eval_data, selection_config)

        cand_record = {
            "model_key": model_key,
            "fitted_model": fitted_obj,
            "is_fitted": is_fitted,
            "cv_accuracy": cv_acc,
            "cv_accuracy_std": cv_metrics.get("cv_accuracy_std", 0.0),
            "test_accuracy": test_acc,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "brier_score": brier,
            "ece": ece,
            "equalized_odds_difference": eod,
            "satisfies_constraints": satisfies,
            "constraint_reason": reason,
            "evaluation": eval_data
        }

        all_candidates.append(cand_record)
        if satisfies and is_fitted:
            satisfied_candidates.append(cand_record)

        comparison_table.append({
            "model": model_key,
            "cv_accuracy": round(cv_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
            "brier_score": round(brier, 4) if brier is not None else "N/A",
            "ece": round(ece, 4) if ece is not None else "N/A",
            "eod": round(eod, 4) if eod is not None else "N/A",
            "status": "PASS" if (satisfies and is_fitted) else ("FAIL_TRAINING" if not is_fitted else "FAIL_CONSTRAINT"),
            "failure_reason": reason
        })

    valid_fitted_candidates = [c for c in all_candidates if c["is_fitted"]]

    # Sort candidates by CV accuracy descending
    satisfied_candidates.sort(key=lambda x: x["cv_accuracy"], reverse=True)
    valid_fitted_candidates.sort(key=lambda x: x["cv_accuracy"], reverse=True)

    if not valid_fitted_candidates:
        selected_key = None
        status = "NO_VALID_MODEL"
        sel_reason = "All candidate models failed during training. No valid fitted estimator is available."
    elif satisfied_candidates:
        top_cand = satisfied_candidates[0]
        selected_key = top_cand["model_key"]
        status = "SELECTED"
        sel_reason = f"Selected candidate '{selected_key}' with highest CV accuracy ({top_cand['cv_accuracy']:.4f}) satisfying all fairness constraints."
    elif allow_fallback and valid_fitted_candidates:
        top_cand = valid_fitted_candidates[0]
        selected_key = top_cand["model_key"]
        status = "NO_MODEL_SATISFIES_CONSTRAINTS_FALLBACK"
        sel_reason = f"No candidate satisfied fairness constraints. Selected fallback model '{selected_key}' (highest CV accuracy: {top_cand['cv_accuracy']:.4f}) for trade-off audit."
    else:
        selected_key = None
        status = "NO_MODEL_SATISFIES_CONSTRAINTS"
        sel_reason = "No candidate model satisfied configured fairness constraints and fallback is disabled."

    return {
        "selected_model": selected_key,
        "selection_status": status,
        "reason": sel_reason,
        "policy": "performance_with_fairness_constraints",
        "comparison_table": comparison_table,
        "candidates": all_candidates
    }
