"""
Unit tests for Negative Test NT-5 (Malformed input schema evolution).
"""

import pytest
from src.reliability.recovery_manager import RecoveryManager


def test_nt5_workflow_execution():
    """Verify NT-5 is detected, blocked, recovered, and reconciled."""
    mgr = RecoveryManager()
    res = mgr.run_nt5_recovery_workflow()

    assert res["test_id"] == "NT-5"
    assert res["status"] == "RECOVERED"
    assert res["failure_detected"] is True
    assert res["reconciliation"] == "PASS"
