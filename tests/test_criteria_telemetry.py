"""
Unit tests for Fairness Criteria Telemetry module.
"""

import os
import json
import pytest

from src.telemetry.fairness_criteria_telemetry import record_criteria_telemetry


def test_record_criteria_telemetry(tmp_path):
    """Verify record_criteria_telemetry logs structured JSONL record."""
    test_jsonl = os.path.join(tmp_path, "test_telemetry.jsonl")

    record = record_criteria_telemetry(
        run_id="RUN-TEST-001",
        dataset_hash="hash123",
        model_id="BASELINE",
        configuration_version="1.0.0",
        selected_criteria=["equalized_odds"],
        measured_metrics={"equalized_odds_difference": 0.4825},
        thresholds={"equalized_odds_difference_max": 0.10},
        compatibility_status="CONFLICT_DETECTED",
        conflict_status="Threshold breached",
        operator_decision_required=True,
        execution_duration_ms=12.5,
        output_path=test_jsonl
    )

    assert os.path.exists(test_jsonl)
    with open(test_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["run_id"] == "RUN-TEST-001"
    assert data["compatibility_status"] == "CONFLICT_DETECTED"
    assert data["operator_decision_required"] is True
