"""
Unit tests for System Operational State Manager.
"""

import pytest
from src.reliability.degraded_mode import SystemStateManager


def test_system_state_transitions():
    """Verify state manager transitions correctly and enforces publish policy."""
    mgr = SystemStateManager(initial_state="NORMAL")
    assert mgr.current_state == "NORMAL"
    assert mgr.can_publish_result() is True

    mgr.transition_to("BLOCKED", reason="Failure detected")
    assert mgr.current_state == "BLOCKED"
    assert mgr.can_publish_result() is False

    mgr.transition_to("RECOVERING", reason="Recovery in progress")
    assert mgr.current_state == "RECOVERING"
    assert mgr.can_publish_result() is False

    mgr.transition_to("RECOVERED", reason="Reconciliation passed")
    assert mgr.current_state == "RECOVERED"
    assert mgr.can_publish_result() is True


def test_invalid_state_transition():
    """Verify state manager raises ValueError on invalid state name."""
    mgr = SystemStateManager()
    with pytest.raises(ValueError):
        mgr.transition_to("INVALID_STATE_NAME")
