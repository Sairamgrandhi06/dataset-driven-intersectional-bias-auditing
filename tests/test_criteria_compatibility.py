"""
Unit tests for Fairness Criteria Compatibility Engine.
"""

import pytest
import json
import os

from src.fairness.criteria_compatibility import evaluate_criteria_compatibility
from src.fairness.criteria import get_criterion_definition, list_supported_criteria


def test_supported_criteria_definitions():
    """Verify get_criterion_definition returns valid metadata for supported criteria."""
    defs = list_supported_criteria()
    assert len(defs) == 5
    assert "equalized_odds" in defs
    meta = get_criterion_definition("equalized_odds")
    assert meta["better_direction"] == "lower"
    assert meta["output_field"] == "equalized_odds_difference"


def test_compatibility_engine_all_pass():
    """Verify compatibility engine returns COMPATIBLE when all metrics satisfy thresholds."""
    metrics = {
        "demographic_parity_difference": 0.05,
        "equalized_odds_difference": 0.08
    }
    criteria = ["demographic_parity", "equalized_odds"]
    thresh = {
        "demographic_parity_difference_max": 0.10,
        "equalized_odds_difference_max": 0.10
    }

    res = evaluate_criteria_compatibility(metrics, criteria, thresh)
    assert res["compatibility_status"] == "COMPATIBLE"
    assert res["passing_criteria_count"] == 2
    assert res["failing_criteria_count"] == 0
    assert res["policy_decision"]["operator_decision_required"] is False


def test_compatibility_engine_conflict_detected():
    """Verify compatibility engine returns CONFLICT_DETECTED when thresholds are breached."""
    metrics = {
        "demographic_parity_difference": 0.3171,
        "equalized_odds_difference": 0.4825
    }
    criteria = ["demographic_parity", "equalized_odds"]
    thresh = {
        "demographic_parity_difference_max": 0.10,
        "equalized_odds_difference_max": 0.10
    }

    res = evaluate_criteria_compatibility(metrics, criteria, thresh)
    assert res["compatibility_status"] == "CONFLICT_DETECTED"
    assert res["failing_criteria_count"] == 2
    assert res["policy_decision"]["operator_decision_required"] is True
    assert res["policy_decision"]["silent_priority_detected"] is False


def test_compatibility_fixture_nt3():
    """Verify compatibility engine behavior against the NT-3 controlled conflict fixture."""
    fixture_path = "tests/fixtures/criteria_conflict.json"
    assert os.path.exists(fixture_path)

    with open(fixture_path, "r", encoding="utf-8") as f:
        fx = json.load(f)

    res = evaluate_criteria_compatibility(fx["observed_metrics"], fx["selected_criteria"], fx["configured_thresholds"])
    assert res["compatibility_status"] == fx["expected_compatibility_status"]
    assert res["failing_criteria_count"] == fx["expected_failing_count"]
    assert res["policy_decision"]["silent_priority_detected"] == fx["expected_silent_priority_detected"]
    assert res["policy_decision"]["operator_decision_required"] == fx["expected_operator_decision_required"]
