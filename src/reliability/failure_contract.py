"""
Standard Failure Event Contract Module.

Defines standardized schema for recording failure events, test identifiers, severity levels,
safe responses, recovery actions, and evidence paths.
"""

import datetime
from typing import Any, Dict, Optional


def create_failure_event(
    failure_id: str,
    test_id: str,
    failure_type: str,
    severity: str,
    run_id: str,
    dataset_hash: str,
    component: str,
    condition: str,
    observed: str,
    expected: str,
    safe_response: str,
    recovery_action: str,
    recovery_status: str = "PENDING",
    evidence_path: str = "",
    status: str = "DETECTED"
) -> Dict[str, Any]:
    """
    Construct standardized failure event contract dictionary.

    Returns:
        dict: Standard failure event dictionary.
    """
    valid_severities = {"WARNING", "ERROR", "CRITICAL"}
    sev = severity.upper() if severity.upper() in valid_severities else "ERROR"

    return {
        "failure_id": failure_id,
        "test_id": test_id,
        "failure_type": failure_type,
        "severity": sev,
        "status": status,
        "timestamp": datetime.datetime.now().isoformat(),
        "run_id": run_id,
        "dataset_hash": dataset_hash,
        "component": component,
        "condition": condition,
        "observed": observed,
        "expected": expected,
        "safe_response": safe_response,
        "recovery_action": recovery_action,
        "recovery_status": recovery_status,
        "evidence_path": evidence_path
    }
