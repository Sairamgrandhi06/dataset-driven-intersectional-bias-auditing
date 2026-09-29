"""
Fairness Criteria Compatibility Analysis Engine.

Evaluates whether multiple selected fairness criteria simultaneously satisfy their
project-configured thresholds based on empirical audit outputs. Operates without silent prioritization.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.fairness.criteria import get_criterion_definition, CRITERIA_DEFINITIONS
from src.fairness.conflict_policy import evaluate_conflict_policy


def evaluate_criteria_compatibility(
    observed_metrics: Dict[str, Any],
    selected_criteria: List[str],
    configured_thresholds: Dict[str, float],
    min_group_sample_size: int = 30
) -> Dict[str, Any]:
    """
    Evaluate per-criterion satisfaction and determine overall compatibility status.

    Parameters:
        observed_metrics (dict): Dictionary mapping metric names to observed float values.
        selected_criteria (list[str]): List of requested criteria to evaluate.
        configured_thresholds (dict): Threshold map (e.g. {'equalized_odds_difference_max': 0.10}).
        min_group_sample_size (int): Sample size check.

    Returns:
        dict: Complete compatibility analysis result.
    """
    if not selected_criteria or not isinstance(selected_criteria, list):
        return {
            "compatibility_status": "INVALID_CONFIGURATION",
            "conflict_reason": "No valid selected_criteria list provided.",
            "per_criterion_results": [],
            "policy_decision": evaluate_conflict_policy([], [], "INVALID_CONFIGURATION")
        }

    per_criterion_results = []

    for criterion_name in selected_criteria:
        if criterion_name not in CRITERIA_DEFINITIONS:
            per_criterion_results.append({
                "criterion": criterion_name,
                "display_name": criterion_name,
                "observed_value": None,
                "configured_threshold": None,
                "threshold_key": "unknown",
                "better_direction": "unknown",
                "status": "FAIL",
                "reason": f"Unsupported criterion name '{criterion_name}'"
            })
            continue

        meta = get_criterion_definition(criterion_name)
        field = meta["output_field"]
        thresh_key = meta["threshold_key"]
        direction = meta["better_direction"]

        # Retrieve threshold from configured_thresholds or fallback to default
        threshold = configured_thresholds.get(thresh_key, meta["default_threshold"])
        observed = observed_metrics.get(field, None)

        if observed is None:
            status = "NOT_ESTIMABLE"
            reason = f"Observed metric '{field}' is not estimable from audit output (e.g. insufficient sample size N < 30)"
        else:
            if direction == "lower":
                is_pass = observed <= threshold
            else: # 'higher' (e.g. disparate impact ratio)
                is_pass = observed >= threshold

            status = "PASS" if is_pass else "FAIL"
            if is_pass:
                reason = f"Observed value ({observed:.4f}) satisfies threshold ({threshold:.4f})"
            else:
                reason = f"Observed value ({observed:.4f}) breaches configured threshold ({threshold:.4f})"

        per_criterion_results.append({
            "criterion": criterion_name,
            "display_name": meta["display_name"],
            "observed_value": round(observed, 6) if observed is not None else None,
            "configured_threshold": round(threshold, 6),
            "threshold_key": thresh_key,
            "better_direction": direction,
            "status": status,
            "reason": reason
        })

    # Determine overall compatibility status
    statuses = [c["status"] for c in per_criterion_results]
    if all(s == "PASS" for s in statuses):
        compatibility_status = "COMPATIBLE"
    elif all(s == "NOT_ESTIMABLE" for s in statuses):
        compatibility_status = "INSUFFICIENT_EVIDENCE"
    elif any(s == "FAIL" for s in statuses):
        compatibility_status = "CONFLICT_DETECTED"
    else:
        compatibility_status = "INSUFFICIENT_EVIDENCE"

    policy_decision = evaluate_conflict_policy(selected_criteria, per_criterion_results, compatibility_status)

    return {
        "compatibility_status": compatibility_status,
        "conflict_reason": policy_decision["conflict_reason"],
        "total_criteria_evaluated": len(selected_criteria),
        "passing_criteria_count": policy_decision["passing_criteria_count"],
        "failing_criteria_count": policy_decision["failing_criteria_count"],
        "per_criterion_results": per_criterion_results,
        "policy_decision": policy_decision
    }
