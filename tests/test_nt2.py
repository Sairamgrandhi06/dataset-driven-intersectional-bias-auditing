"""
Unit tests for Negative Test NT-2 (Hidden mitigation cost).
"""

import pytest
from src.reliability.recovery_manager import RecoveryManager


def test_nt2_workflow_execution():
    """Verify NT-2 is detected, blocked, recovered, and reconciled."""
    mgr = RecoveryManager()
    res = mgr.run_nt2_recovery_workflow()

    assert res["test_id"] == "NT-2"
    assert res["status"] == "RECOVERED"
    assert res["failure_detected"] is True
    assert res["reconciliation"] == "PASS"
