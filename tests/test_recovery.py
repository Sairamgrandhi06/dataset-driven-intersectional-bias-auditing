"""
Unit tests for Recovery Manager overall integration.
"""

import pytest
from src.reliability.recovery_manager import RecoveryManager
from src.reliability.health_check import perform_operating_health_check


def test_health_check():
    """Verify health check returns PASS on healthy workspace."""
    health = perform_operating_health_check()
    assert health["overall_status"] == "PASS"
    assert health["checks_passed"] == health["total_checks"]


def test_all_negative_tests_recovery():
    """Verify RecoveryManager executes all 5 recovery workflows successfully."""
    mgr = RecoveryManager()
    results = [
        mgr.run_nt1_recovery_workflow(),
        mgr.run_nt2_recovery_workflow(),
        mgr.run_nt3_recovery_workflow(),
        mgr.run_nt4_recovery_workflow(),
        mgr.run_nt5_recovery_workflow()
    ]

    for r in results:
        assert r["status"] == "RECOVERED"
        assert r["reconciliation"] == "PASS"
