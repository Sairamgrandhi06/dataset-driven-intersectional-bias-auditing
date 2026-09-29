"""
Unit tests for Failure Contract and Registry.
"""

import pytest
from src.reliability.failure_contract import create_failure_event
from src.reliability.failure_registry import register_failure_event


def test_create_failure_event():
    """Verify failure event dictionary contains all standard fields."""
    event = create_failure_event(
        failure_id="FAIL-001",
        test_id="NT-1",
        failure_type="TEST_FAILURE",
        severity="ERROR",
        run_id="RUN-001",
        dataset_hash="hash123",
        component="test_comp",
        condition="Missing item",
        observed="Observed null",
        expected="Expected valid",
        safe_response="BLOCK_RESULT",
        recovery_action="RESTORE"
    )

    assert event["failure_id"] == "FAIL-001"
    assert event["severity"] == "ERROR"
    assert event["status"] == "DETECTED"
    assert "timestamp" in event


def test_register_failure_event(tmp_path):
    """Verify failure event is correctly written to JSONL."""
    test_jsonl = str(tmp_path / "test_failures.jsonl")
    event = create_failure_event(
        failure_id="FAIL-002",
        test_id="NT-2",
        failure_type="METRIC_MISSING",
        severity="ERROR",
        run_id="RUN-002",
        dataset_hash="hash123",
        component="mitigation",
        condition="Missing cost",
        observed="None",
        expected="Full metric",
        safe_response="BLOCK",
        recovery_action="RECOMPUTE"
    )

    registered = register_failure_event(event, output_path=test_jsonl)
    assert registered["failure_id"] == "FAIL-002"
