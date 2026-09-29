"""
Explicit No-Silent-Priority Conflict Policy Module.

Implements the 'report_all_do_not_silently_prioritize' policy mandated by FR-4.
Ensures that when multiple fairness criteria exhibit threshold conflicts or failure states,
no criterion is silently prioritized, hidden, or discarded. Sets operator_decision_required = True.
"""

from typing import Any, Dict, List


def evaluate_conflict_policy(
    selected_criteria: List[str],
    per_criterion_results: List[Dict[str, Any]],
    compatibility_status: str
) -> Dict[str, Any]:
    """
    Apply no-silent-priority conflict policy to fairness criteria evaluation outputs.

    Parameters:
        selected_criteria (list[str]): List of requested criteria names.
        per_criterion_results (list[dict]): Evaluated results for each criterion.
        compatibility_status (str): 'COMPATIBLE', 'CONFLICT_DETECTED', 'INSUFFICIENT_EVIDENCE', or 'INVALID_CONFIGURATION'.

    Returns:
        dict: Policy decision structure confirming zero silent prioritization and operator decision requirement.
    """
    failing_criteria = [c for c in per_criterion_results if c.get("status") == "FAIL"]
    passing_criteria = [c for c in per_criterion_results if c.get("status") == "PASS"]
    not_estimable_criteria = [c for c in per_criterion_results if c.get("status") == "NOT_ESTIMABLE"]

    # Determine if operator decision is required
    if compatibility_status in ("CONFLICT_DETECTED", "INVALID_CONFIGURATION"):
        operator_decision_required = True
        if len(failing_criteria) > 0 and len(passing_criteria) > 0:
            reason = f"Criterion threshold conflict: {len(passing_criteria)} criteria PASSED ({[c['criterion'] for c in passing_criteria]}) while {len(failing_criteria)} criteria FAILED ({[c['criterion'] for c in failing_criteria]}). No criterion was silently prioritized."
        elif len(failing_criteria) == len(selected_criteria):
            reason = f"All {len(selected_criteria)} criteria failed their configured thresholds. Simultaneous satisfaction not demonstrated."
        else:
            reason = f"Compatibility analysis returned status '{compatibility_status}'."
    elif compatibility_status == "INSUFFICIENT_EVIDENCE":
        operator_decision_required = True
        reason = "Insufficient statistical sample evidence to demonstrate simultaneous threshold satisfaction (e.g. insufficient subgroup sample size N < 30)."
    else:
        operator_decision_required = False
        reason = "All selected criteria simultaneously satisfied their configured thresholds."

    return {
        "policy": "report_all_do_not_silently_prioritize",
        "selected_criteria": selected_criteria,
        "failing_criteria_count": len(failing_criteria),
        "passing_criteria_count": len(passing_criteria),
        "not_estimable_criteria_count": len(not_estimable_criteria),
        "compatibility_status": compatibility_status,
        "conflict_reason": reason,
        "silent_priority_detected": False,
        "priority_selected": False,
        "operator_decision_required": operator_decision_required
    }
