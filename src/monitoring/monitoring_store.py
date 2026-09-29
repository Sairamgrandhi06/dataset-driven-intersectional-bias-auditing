"""
Monitoring Persistence & Evidence Registry Module.

Saves monitoring execution runs into central registry 'registry/monitoring_registry.json'
and serializes immutable structured evidence reports under 'evidence/monitoring/<dataset_id>/<monitoring_run_id>/'.
"""

import json
import os
import time
from typing import Dict, Any, List, Optional


def get_monitoring_registry_path(base_dir: str = ".") -> str:
    """Return path to central monitoring registry JSON file."""
    return os.path.join(base_dir, "registry", "monitoring_registry.json")


def load_monitoring_registry(base_dir: str = ".") -> Dict[str, Any]:
    """Load monitoring registry dictionary."""
    path = get_monitoring_registry_path(base_dir)
    if not os.path.exists(path):
        return {"datasets": {}, "runs": []}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"datasets": {}, "runs": []}


def save_monitoring_registry(registry_data: Dict[str, Any], base_dir: str = ".") -> None:
    """Safely save monitoring registry dictionary."""
    path = get_monitoring_registry_path(base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2)
    os.replace(temp_path, path)


def save_monitoring_run_evidence(
    dataset_id: str,
    monitoring_run_id: str,
    reference_version: str,
    current_data_hash: str,
    schema_report: Dict[str, Any],
    quality_report: Dict[str, Any],
    feature_drift_report: Dict[str, Any],
    target_drift_report: Dict[str, Any],
    performance_report: Dict[str, Any],
    fairness_report: Dict[str, Any],
    calibration_report: Dict[str, Any],
    health_report: Dict[str, Any],
    retraining_recommendation: Dict[str, Any],
    reference_dataset_id: Optional[str] = None,
    reference_dataset_hash: Optional[str] = None,
    reference_run_id: Optional[str] = None,
    base_dir: str = "."
) -> Dict[str, str]:
    """
    Save all structured monitoring evidence files under 'evidence/monitoring/<dataset_id>/<monitoring_run_id>/'
    and register the run in 'registry/monitoring_registry.json'.

    Returns:
        dict: Mapping of artifact name to relative file path.
    """
    run_dir = os.path.join(base_dir, "evidence", "monitoring", dataset_id, monitoring_run_id)
    os.makedirs(run_dir, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    artifacts = {
        "drift_report.json": {
            "schema_drift": schema_report,
            "data_quality_drift": quality_report,
            "feature_drift": feature_drift_report,
            "target_drift": target_drift_report
        },
        "performance_monitoring.json": performance_report,
        "fairness_monitoring.json": fairness_report,
        "calibration_monitoring.json": calibration_report,
        "health_report.json": health_report,
        "retraining_recommendation.json": retraining_recommendation
    }

    saved_paths = {}
    for filename, content in artifacts.items():
        file_path = os.path.join(run_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
        saved_paths[filename] = file_path.replace("\\", "/")

    summary_report = {
        "monitoring_run_id": monitoring_run_id,
        "dataset_id": dataset_id,
        "monitoring_dataset_id": dataset_id,
        "monitoring_dataset_hash": current_data_hash,
        "reference_dataset_id": reference_dataset_id or dataset_id,
        "reference_dataset_hash": reference_dataset_hash or "",
        "reference_model_version": reference_version,
        "reference_run_id": reference_run_id or "",
        "current_data_hash": current_data_hash,
        "timestamp": timestamp,
        "schema_status": schema_report.get("status"),
        "feature_drift_status": feature_drift_report.get("status"),
        "performance_status": performance_report.get("status"),
        "fairness_status": fairness_report.get("status"),
        "calibration_status": calibration_report.get("status"),
        "overall_health": health_report.get("overall_health"),
        "retraining_recommendation": retraining_recommendation.get("status"),
        "artifact_paths": saved_paths
    }

    summary_path = os.path.join(run_dir, "monitoring_report.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    saved_paths["monitoring_report.json"] = summary_path.replace("\\", "/")

    # Update central monitoring registry
    reg = load_monitoring_registry(base_dir)

    ds_entry = reg["datasets"].get(dataset_id, {
        "dataset_id": dataset_id,
        "total_monitoring_runs": 0,
        "latest_run_id": None
    })
    ds_entry["total_monitoring_runs"] += 1
    ds_entry["latest_run_id"] = monitoring_run_id
    ds_entry["latest_health"] = health_report.get("overall_health")
    ds_entry["latest_recommendation"] = retraining_recommendation.get("status")
    reg["datasets"][dataset_id] = ds_entry

    run_record = {
        "monitoring_run_id": monitoring_run_id,
        "dataset_id": dataset_id,
        "monitoring_dataset_id": dataset_id,
        "monitoring_dataset_hash": current_data_hash,
        "reference_dataset_id": reference_dataset_id or dataset_id,
        "reference_dataset_hash": reference_dataset_hash or "",
        "reference_model_version": reference_version,
        "reference_run_id": reference_run_id or "",
        "current_data_hash": current_data_hash,
        "timestamp": timestamp,
        "schema_status": schema_report.get("status"),
        "feature_drift_status": feature_drift_report.get("status"),
        "performance_status": performance_report.get("status"),
        "fairness_status": fairness_report.get("status"),
        "calibration_status": calibration_report.get("status"),
        "health_status": health_report.get("overall_health"),
        "retraining_recommendation": retraining_recommendation.get("status"),
        "evidence_directory": run_dir.replace("\\", "/")
    }

    reg["runs"].append(run_record)
    save_monitoring_registry(reg, base_dir)

    return saved_paths
