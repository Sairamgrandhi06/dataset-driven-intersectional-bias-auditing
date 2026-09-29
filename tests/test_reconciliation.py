"""
Unit tests for Reconciliation Engine.
"""

import pytest
from src.reliability.reconciliation import reconcile_recovered_output


def test_reconciliation_pass():
    """Verify reconciliation returns PASS when metrics match within tolerance."""
    exp = {"accuracy": 0.8399, "equalized_odds": 0.4825}
    rec = {"accuracy": 0.83991, "equalized_odds": 0.48249}

    res = reconcile_recovered_output(exp, rec, tolerance=1e-4)
    assert res["status"] == "PASS"
    assert res["mismatched_count"] == 0


def test_reconciliation_mismatch():
    """Verify reconciliation returns MISMATCH when metrics differ beyond tolerance."""
    exp = {"accuracy": 0.8399}
    rec = {"accuracy": 0.7578}

    res = reconcile_recovered_output(exp, rec, tolerance=1e-4)
    assert res["status"] == "MISMATCH"
    assert res["mismatched_count"] == 1
