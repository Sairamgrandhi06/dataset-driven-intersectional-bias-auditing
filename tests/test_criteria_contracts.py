"""
Unit tests for Fairness Criteria Input and Output Contracts.
"""

import pytest
import numpy as np
import pandas as pd

from src.contracts.fairness_criteria_contract import validate_criteria_input_contract
from src.contracts.criteria_output_contract import build_criteria_output_contract
from src.contracts.error_contract import create_structured_error, CriteriaError


def test_valid_input_contract():
    """Verify input contract passes on valid inputs."""
    y_t = np.array([1, 0, 1, 0])
    y_p = np.array([1, 0, 0, 0])
    A_df = pd.DataFrame({"sex": ["Female", "Male", "Female", "Male"]})
    criteria = ["demographic_parity", "equalized_odds"]
    thresh = {"demographic_parity_difference_max": 0.10, "equalized_odds_difference_max": 0.10}

    res = validate_criteria_input_contract(y_t, y_p, A_df, criteria, thresh)
    assert res["is_valid"] is True
    assert len(res["errors"]) == 0


def test_missing_input_contract():
    """Verify input contract fails when y_true or y_pred is missing."""
    res = validate_criteria_input_contract(None, np.array([1, 0]), pd.DataFrame(), ["equalized_odds"], {})
    assert res["is_valid"] is False
    assert any(e["error_code"] == "MISSING_INPUT" for e in res["errors"])


def test_length_mismatch_contract():
    """Verify input contract fails on sample length mismatch."""
    res = validate_criteria_input_contract([1, 0, 1], [1, 0], pd.DataFrame({"sex": ["Female", "Male"]}), ["equalized_odds"], {})
    assert res["is_valid"] is False
    assert any(e["error_code"] == "LENGTH_MISMATCH" for e in res["errors"])


def test_invalid_criterion_contract():
    """Verify input contract fails on unsupported criterion name."""
    y = [1, 0]
    res = validate_criteria_input_contract(y, y, pd.DataFrame({"sex": ["F", "M"]}), ["unsupported_metric_xyz"], {})
    assert res["is_valid"] is False
    assert any(e["error_code"] == "INVALID_CRITERION" for e in res["errors"])


def test_duplicate_criterion_contract():
    """Verify input contract fails on duplicate criteria names."""
    y = [1, 0]
    res = validate_criteria_input_contract(y, y, pd.DataFrame({"sex": ["F", "M"]}), ["equalized_odds", "equalized_odds"], {})
    assert res["is_valid"] is False
    assert any(e["error_code"] == "INVALID_CRITERION" for e in res["errors"])


def test_invalid_threshold_contract():
    """Verify input contract fails on negative threshold."""
    y = [1, 0]
    res = validate_criteria_input_contract(y, y, pd.DataFrame({"sex": ["F", "M"]}), ["equalized_odds"], {"equalized_odds_difference_max": -0.5})
    assert res["is_valid"] is False
    assert any(e["error_code"] == "INVALID_THRESHOLD" for e in res["errors"])


def test_output_contract_builder():
    """Verify build_criteria_output_contract constructs complete output schema."""
    contract = build_criteria_output_contract(
        run_id="RUN-100",
        dataset_id="UCI-ADULT",
        dataset_hash="hash123",
        model_id="BASELINE",
        configuration_version="1.0.0",
        selected_criteria=["equalized_odds"],
        metric_values={"equalized_odds_difference": 0.4825},
        configured_thresholds={"equalized_odds_difference_max": 0.10},
        per_criterion_status=[{"criterion": "equalized_odds", "status": "FAIL"}],
        compatibility_status="CONFLICT_DETECTED",
        conflict_reason="Threshold breached",
        operator_decision_required=True
    )

    assert contract["run_id"] == "RUN-100"
    assert contract["compatibility_status"] == "CONFLICT_DETECTED"
    assert contract["silent_priority_detected"] is False
    assert contract["operator_decision_required"] is True
