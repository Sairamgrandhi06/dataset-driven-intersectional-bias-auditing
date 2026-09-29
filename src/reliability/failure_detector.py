"""
Failure Detector Module for NT-1 through NT-5 Failure Conditions.

Detects missing intersectional groups (NT-1), missing trade-off metrics (NT-2),
fairness criteria threshold conflicts (NT-3), sensitive data exposure attempts (NT-4),
and malformed input schemas (NT-5).
"""

import json
import os
import pandas as pd
from typing import Any, Dict, List, Optional

from src.reliability.failure_contract import create_failure_event

EXPECTED_INTERSECTIONAL_GROUPS = [
    "Female + Amer-Indian-Eskimo",
    "Female + Asian-Pac-Islander",
    "Female + Black",
    "Female + Other",
    "Female + White",
    "Male + Amer-Indian-Eskimo",
    "Male + Asian-Pac-Islander",
    "Male + Black",
    "Male + Other",
    "Male + White"
]

MANDATORY_MITIGATION_METRICS = [
    "accuracy",
    "recall",
    "equalized_odds_difference",
    "brier_score",
    "expected_calibration_error"
]

PROHIBITED_PII_FIELDS = {
    "raw_ssn",
    "raw_name",
    "medical_record_number",
    "social_security_number"
}


def detect_nt1_missing_intersections(audit_result: Dict[str, Any], dataset_hash: str = "") -> Optional[Dict[str, Any]]:
    """
    NT-1 Failure Detection: Single-attribute audits missing intersections.
    """
    discovered_groups = audit_result.get("discovered_groups", list(audit_result.get("group_metrics", {}).keys()))
    missing = [g for g in EXPECTED_INTERSECTIONAL_GROUPS if g not in discovered_groups]

    if len(missing) > 0:
        return create_failure_event(
            failure_id="FAIL-NT1-001",
            test_id="NT-1",
            failure_type="MISSING_INTERSECTIONAL_GROUP",
            severity="ERROR",
            run_id="RUN-NT1-001",
            dataset_hash=dataset_hash,
            component="intersectional_audit",
            condition=f"Expected {len(EXPECTED_INTERSECTIONAL_GROUPS)} intersectional groups, missing {len(missing)}",
            observed=f"Discovered groups: {discovered_groups}",
            expected=f"All {len(EXPECTED_INTERSECTIONAL_GROUPS)} expected intersectional groups present",
            safe_response="BLOCK_INTERSECTIONAL_RESULT",
            recovery_action="RESTORE_APPROVED_DATASET_AND_REAUDIT",
            evidence_path="evidence/reliability/failure_events.jsonl"
        )
    return None


def detect_nt2_hidden_mitigation_cost(tradeoff_report: Dict[str, Any], dataset_hash: str = "") -> Optional[Dict[str, Any]]:
    """
    NT-2 Failure Detection: Hidden mitigation cost.
    """
    perf = tradeoff_report.get("performance_metrics", {}).get("mitigated", {})
    fair = tradeoff_report.get("fairness_metrics", {}).get("mitigated", {})
    calib = tradeoff_report.get("calibration_metrics", {}).get("mitigated", {})

    all_observed = {**perf, **fair, **calib}
    missing_metrics = [m for m in MANDATORY_MITIGATION_METRICS if m not in all_observed]

    if len(missing_metrics) > 0:
        return create_failure_event(
            failure_id="FAIL-NT2-001",
            test_id="NT-2",
            failure_type="HIDDEN_MITIGATION_COST",
            severity="ERROR",
            run_id="RUN-NT2-001",
            dataset_hash=dataset_hash,
            component="tradeoff_reporting",
            condition=f"Mitigation report missing required trade-off cost metrics: {missing_metrics}",
            observed=f"Observed metrics keys: {list(all_observed.keys())}",
            expected=f"All required metrics present: {MANDATORY_MITIGATION_METRICS}",
            safe_response="BLOCK_MITIGATION_CLAIM",
            recovery_action="RECOMPUTE_COMPLETE_TRADEOFF_METRICS",
            evidence_path="evidence/reliability/failure_events.jsonl"
        )
    return None


def detect_nt3_criteria_conflict(compatibility_report: Dict[str, Any], dataset_hash: str = "") -> Optional[Dict[str, Any]]:
    """
    NT-3 Failure Detection: Incompatible fairness criteria threshold conflict.
    """
    status = compatibility_report.get("compatibility_status", "UNKNOWN")

    if status in ("CONFLICT_DETECTED", "INVALID_CONFIGURATION"):
        return create_failure_event(
            failure_id="FAIL-NT3-001",
            test_id="NT-3",
            failure_type="CRITERIA_THRESHOLD_CONFLICT",
            severity="WARNING",
            run_id="RUN-NT3-001",
            dataset_hash=dataset_hash,
            component="fairness_criteria_compatibility",
            condition=f"Criteria threshold conflict detected: {compatibility_report.get('conflict_reason', '')}",
            observed=f"Compatibility status = {status}, Priority selected = False",
            expected="All selected criteria simultaneously satisfy configured thresholds",
            safe_response="ENTER_BLOCKED_OR_DEFERRED_DECISION_STATE",
            recovery_action="REQUIRE_EXPLICIT_OPERATOR_POLICY_DECISION",
            evidence_path="evidence/reliability/failure_events.jsonl"
        )
    return None


def detect_nt4_sensitive_evidence(export_payload: Dict[str, Any], dataset_hash: str = "") -> Optional[Dict[str, Any]]:
    """
    NT-4 Failure Detection: Sensitive evidence exposure attempt.
    """
    found_pii = [k for k in export_payload.keys() if k in PROHIBITED_PII_FIELDS]

    if len(found_pii) > 0:
        return create_failure_event(
            failure_id="FAIL-NT4-001",
            test_id="NT-4",
            failure_type="SENSITIVE_EVIDENCE_EXPOSURE",
            severity="CRITICAL",
            run_id="RUN-NT4-001",
            dataset_hash=dataset_hash,
            component="evidence_export_authorization",
            condition=f"Export payload contains prohibited PII fields: {found_pii}",
            observed=f"Attempted export of fields: {list(export_payload.keys())}",
            expected="Export payload contains safe aggregated metrics only",
            safe_response="ACCESS_DENIED_BLOCK_EXPORT",
            recovery_action="SANITIZE_PAYLOAD_TO_PERMITTED_FIELDS_ONLY",
            evidence_path="evidence/reliability/failure_events.jsonl"
        )
    return None


def detect_nt5_schema_evolution(df: pd.DataFrame, dataset_hash: str = "") -> Optional[Dict[str, Any]]:
    """
    NT-5 Failure Detection: Malformed / schema evolution input.
    """
    req_cols = {"income", "sex", "race"}
    missing_cols = list(req_cols - set(df.columns))

    if len(missing_cols) > 0:
        return create_failure_event(
            failure_id="FAIL-NT5-001",
            test_id="NT-5",
            failure_type="MALFORMED_SCHEMA_EVOLUTION",
            severity="ERROR",
            run_id="RUN-NT5-001",
            dataset_hash=dataset_hash,
            component="data_ingestion_pipeline",
            condition=f"Input dataset missing required schema columns: {missing_cols}",
            observed=f"Observed dataset columns: {list(df.columns)}",
            expected=f"Input dataset must contain columns: {sorted(list(req_cols))}",
            safe_response="REJECT_INPUT_BLOCK_TRAINING",
            recovery_action="RESTORE_APPROVED_SCHEMA_FIXTURE",
            evidence_path="evidence/reliability/failure_events.jsonl"
        )
    return None
