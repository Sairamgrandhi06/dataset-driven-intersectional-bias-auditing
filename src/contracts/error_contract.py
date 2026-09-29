"""
Fairness Criteria Structured Error Contract Module.

Defines standard error codes, severity levels, and recovery actions for input validation,
configuration parsing, and compatibility analysis errors.
"""

from typing import Any, Dict, Optional


class CriteriaError(Exception):
    """Base exception for fairness criteria contract errors."""
    def __init__(self, error_code: str, message: str, field: str = "general", severity: str = "ERROR", recovery_action: str = ""):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.field = field
        self.severity = severity
        self.recovery_action = recovery_action

    def to_dict(self) -> Dict[str, str]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "field": self.field,
            "severity": self.severity,
            "recovery_action": self.recovery_action
        }


def create_structured_error(
    error_code: str,
    message: str,
    field: str = "general",
    severity: str = "ERROR",
    recovery_action: str = "Check parameter settings and documentation."
) -> Dict[str, str]:
    """
    Construct a standardized error contract dictionary.

    Parameters:
        error_code (str): Standardized error identifier.
        message (str): Human-readable error description.
        field (str): Parameter or component causing error.
        severity (str): 'ERROR', 'WARNING', or 'INFO'.
        recovery_action (str): Recommended action to resolve error.

    Returns:
        dict: Standardized error dictionary.
    """
    valid_codes = {
        "INVALID_CRITERION",
        "INVALID_THRESHOLD",
        "MISSING_INPUT",
        "LENGTH_MISMATCH",
        "MISSING_PROTECTED_ATTRIBUTE",
        "INVALID_TARGET",
        "INVALID_CONFIGURATION",
        "INSUFFICIENT_GROUP_SIZE",
        "INSUFFICIENT_EVIDENCE"
    }

    code = error_code if error_code in valid_codes else "INVALID_CONFIGURATION"

    return {
        "error_code": code,
        "message": message,
        "field": field,
        "severity": severity,
        "recovery_action": recovery_action
    }
