"""
Structured Output Contract for Fairness Criteria Compatibility Analysis.

Constructs standardized, machine-readable output objects containing run IDs, dataset hashes,
compatibility statuses, per-criterion results, and no-silent-priority flags.
"""

import datetime
from typing import Any, Dict, List, Optional


def build_criteria_output_contract(
    run_id: str,
    dataset_id: str,
    dataset_hash: str,
    model_id: str,
    configuration_version: str,
    selected_criteria: List[str],
    metric_values: Dict[str, Any],
    configured_thresholds: Dict[str, float],
    per_criterion_status: List[Dict[str, Any]],
    compatibility_status: str,
    conflict_reason: str,
    operator_decision_required: bool,
    timestamp: str | None = None
) -> Dict[str, Any]:
    """
    Construct standardized output contract dictionary.

    Returns:
        dict: Standardized criteria output contract.
    """
    ts = timestamp or datetime.datetime.now().isoformat()

    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "model_id": model_id,
        "configuration_version": configuration_version,
        "selected_criteria": selected_criteria,
        "metric_values": metric_values,
        "configured_thresholds": configured_thresholds,
        "per_criterion_status": per_criterion_status,
        "compatibility_status": compatibility_status,
        "conflict_reason": conflict_reason,
        "silent_priority_detected": False,
        "operator_decision_required": operator_decision_required,
        "timestamp": ts
    }
