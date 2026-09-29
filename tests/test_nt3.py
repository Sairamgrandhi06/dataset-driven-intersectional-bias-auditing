"""
Unit tests for Negative Test NT-3 (Silent criterion selection).
"""

import pytest
from src.reliability.recovery_manager import RecoveryManager


def test_nt3_workflow_execution():
    """Verify NT-3 is detected, blocked, recovered, and reconciled."""
    mgr = RecoveryManager()
    res = mgr.run_nt3_recovery_workflow()

    assert res["test_id"] == "NT-3"
    assert res["status"] == "RECOVERED"
    assert res["failure_detected"] is True
    assert res["reconciliation"] == "PASS"
