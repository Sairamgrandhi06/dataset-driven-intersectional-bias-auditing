"""
Unit tests for Explicit No-Silent-Priority Conflict Policy.
"""

import pytest
from src.fairness.conflict_policy import evaluate_conflict_policy


def test_conflict_policy_no_silent_prioritization():
    """Verify policy never silently prioritizes criteria or sets priority_selected = True."""
    per_results = [
        {"criterion": "demographic_parity", "status": "PASS"},
        {"criterion": "equalized_odds", "status": "FAIL"}
    ]
    policy = evaluate_conflict_policy(["demographic_parity", "equalized_odds"], per_results, "CONFLICT_DETECTED")

    assert policy["policy"] == "report_all_do_not_silently_prioritize"
    assert policy["silent_priority_detected"] is False
    assert policy["priority_selected"] is False
    assert policy["operator_decision_required"] is True
    assert policy["failing_criteria_count"] == 1
    assert policy["passing_criteria_count"] == 1


def test_conflict_policy_all_passing():
    """Verify operator_decision_required = False when all criteria pass."""
    per_results = [
        {"criterion": "demographic_parity", "status": "PASS"},
        {"criterion": "equalized_odds", "status": "PASS"}
    ]
    policy = evaluate_conflict_policy(["demographic_parity", "equalized_odds"], per_results, "COMPATIBLE")

    assert policy["operator_decision_required"] is False
    assert policy["silent_priority_detected"] is False
