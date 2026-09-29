"""
Unit tests for Negative Test NT-1 (Missing intersectional groups).
"""

import pytest
from src.reliability.recovery_manager import RecoveryManager


def test_nt1_workflow_execution():
    """Verify NT-1 is detected, blocked, recovered, and reconciled."""
    mgr = RecoveryManager()
    res = mgr.run_nt1_recovery_workflow()

    assert res["test_id"] == "NT-1"
    assert res["status"] == "RECOVERED"
    assert res["failure_detected"] is True
    assert res["reconciliation"] == "PASS"
