"""
Fairness Criteria Telemetry Module.

Logs structured JSONL telemetry records to 'evidence/telemetry/fairness_criteria.jsonl'
for operational monitoring and audit compliance.
"""

import datetime
import json
import os
import time
from typing import Any, Dict, List, Optional


def get_telemetry_file_path() -> str:
    """Return path to telemetry JSONL file."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root, "evidence", "telemetry", "fairness_criteria.jsonl")


def record_criteria_telemetry(
    run_id: str,
    dataset_hash: str,
    model_id: str,
    configuration_version: str,
    selected_criteria: List[str],
    measured_metrics: Dict[str, Any],
    thresholds: Dict[str, float],
    compatibility_status: str,
    conflict_status: str,
    operator_decision_required: bool,
    error_code: Optional[str] = None,
    execution_duration_ms: float = 0.0,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Construct and append a telemetry event record to JSONL evidence file.

    Returns:
        dict: The serialized telemetry record.
    """
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "run_id": run_id,
        "dataset_hash": dataset_hash,
        "model_id": model_id,
        "configuration_version": configuration_version,
        "selected_criteria": selected_criteria,
        "measured_metrics": measured_metrics,
        "thresholds": thresholds,
        "compatibility_status": compatibility_status,
        "conflict_status": conflict_status,
        "operator_decision_required": operator_decision_required,
        "error_code": error_code,
        "execution_duration_ms": round(execution_duration_ms, 2)
    }

    target_path = output_path or get_telemetry_file_path()
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    try:
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        record["telemetry_write_error"] = str(e)

    return record
