"""
System Operational State & Degraded Mode Machine.

Tracks system states: NORMAL, DEGRADED, BLOCKED, RECOVERING, RECOVERED, FAILED.
Enforces rules preventing invalid or blocked results from reporting PASS.
"""

from typing import Any, Dict, List, Optional

VALID_OPERATIONAL_STATES = {
    "NORMAL",
    "DEGRADED",
    "BLOCKED",
    "RECOVERING",
    "RECOVERED",
    "FAILED"
}


class SystemStateManager:
    """Manages system operational state transitions and safety enforcement."""

    def __init__(self, initial_state: str = "NORMAL"):
        self._current_state = initial_state if initial_state in VALID_OPERATIONAL_STATES else "NORMAL"
        self._history: List[Dict[str, str]] = [{"state": self._current_state, "reason": "System Initialized"}]

    @property
    def current_state(self) -> str:
        return self._current_state

    def transition_to(self, new_state: str, reason: str = "") -> str:
        """
        Transition system to a new state and record state history.
        """
        if new_state not in VALID_OPERATIONAL_STATES:
            raise ValueError(f"Invalid operational state '{new_state}'. Must be one of {VALID_OPERATIONAL_STATES}")

        self._current_state = new_state
        self._history.append({"state": new_state, "reason": reason})
        return self._current_state

    def can_publish_result(self) -> bool:
        """
        Return True only if system is in NORMAL or RECOVERED state.
        Results MUST NOT be published if state is BLOCKED, DEGRADED, RECOVERING, or FAILED.
        """
        return self._current_state in ("NORMAL", "RECOVERED")

    def get_status_summary(self) -> Dict[str, Any]:
        return {
            "current_state": self._current_state,
            "can_publish": self.can_publish_result(),
            "state_history": self._history
        }
