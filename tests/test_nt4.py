"""
Unit tests for Negative Test NT-4 (Sensitive evidence exposure).
"""

import pytest
from src.reliability.recovery_manager import RecoveryManager


def test_nt4_workflow_execution():
    """Verify NT-4 is detected, blocked, recovered, and reconciled."""
    mgr = RecoveryManager()
    res = mgr.run_nt4_recovery_workflow()

    assert res["test_id"] == "NT-4"
    assert res["status"] == "RECOVERED"
    assert res["failure_detected"] is True
    assert res["reconciliation"] == "PASS"
