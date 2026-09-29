"""
Model Registry Store Engine.

Provides machine-readable model lifecycle tracking, deterministic versioning (v1, v2, v3...),
run identification, and dataset-isolated model registry persistence under registry/model_registry.json.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


def get_registry_path(base_dir: str = ".") -> str:
    """Return absolute path to registry/model_registry.json."""
    return os.path.abspath(os.path.join(base_dir, "registry", "model_registry.json"))


def load_model_registry(base_dir: str = ".") -> Dict[str, Any]:
    """
    Load model registry JSON from disk, initializing an empty structure if missing.

    Returns:
        dict: Model registry database structure.
    """
    reg_path = get_registry_path(base_dir=base_dir)
    if not os.path.exists(reg_path):
        return {"datasets": {}, "runs": []}

    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"datasets": {}, "runs": []}


def save_model_registry(registry_data: Dict[str, Any], base_dir: str = ".") -> str:
    """
    Safely persist model registry dictionary to registry/model_registry.json.

    Parameters:
        registry_data (dict): Registry data dictionary.
        base_dir (str): Base project directory path.

    Returns:
        str: Path to saved registry file.
    """
    reg_path = get_registry_path(base_dir=base_dir)
    os.makedirs(os.path.dirname(reg_path), exist_ok=True)

    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2)

    return reg_path


def get_next_model_version(dataset_id: str, base_dir: str = ".") -> str:
    """
    Calculate the next deterministic model version (v1, v2, v3...) for dataset_id.
    Failed training runs do NOT increment the model version number.

    Parameters:
        dataset_id (str): Dataset identifier string.
        base_dir (str): Project root directory path.

    Returns:
        str: Version string (e.g. 'v1', 'v2').
    """
    reg = load_model_registry(base_dir=base_dir)
    runs = reg.get("runs", [])

    completed_version_count = 0
    for r in runs:
        if r.get("dataset_id") == dataset_id and r.get("status") in ["COMPLETED", "SELECTED", "SUPERSEDED", "ACTIVE"]:
            completed_version_count += 1

    return f"v{completed_version_count + 1}"


def generate_run_id(dataset_id: str) -> str:
    """Generate a collision-safe run identifier (e.g. run_20260812_001)."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sanitized_id = "".join(c if c.isalnum() else "_" for c in str(dataset_id)).strip("_")
    return f"run_{date_str}_{sanitized_id}"


def register_model_run(
    dataset_id: str,
    dataset_hash: str,
    run_id: str,
    model_type: str,
    config_dict: Dict[str, Any],
    selection_policy: str,
    train_count: int,
    test_count: int,
    feature_count: int,
    random_seed: int,
    cv_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    fairness_metrics: Dict[str, Any],
    calibration_metrics: Dict[str, Any],
    artifact_path: str,
    is_selected: bool = True,
    status: str = "ACTIVE",
    base_dir: str = "."
) -> Dict[str, Any]:
    """
    Register a model training run into registry/model_registry.json.

    Parameters:
        dataset_id (str): Dataset identifier.
        dataset_hash (str): SHA-256 hash of dataset.
        run_id (str): Unique run identifier.
        model_type (str): Selected model key (e.g. 'logistic_regression', 'random_forest').
        config_dict (dict): Configuration options.
        selection_policy (str): Selection policy name.
        train_count (int): Training sample size.
        test_count (int): Test sample size.
        feature_count (int): Feature count.
        random_seed (int): Random seed.
        cv_metrics (dict): CV metrics.
        test_metrics (dict): Test metrics.
        fairness_metrics (dict): Fairness metrics.
        calibration_metrics (dict): Calibration metrics.
        artifact_path (str): Model file path.
        is_selected (bool): Whether model was selected.
        status (str): Lifecycle status ('ACTIVE', 'COMPLETED', 'SUPERSEDED', etc.).
        base_dir (str): Base project directory path.

    Returns:
        dict: Registered model record dictionary.
    """
    reg = load_model_registry(base_dir=base_dir)

    # Determine version number for completed runs
    if status != "FAILED":
        version = get_next_model_version(dataset_id, base_dir=base_dir)
    else:
        version = "NONE"

    # Transition previous ACTIVE models for dataset_id to SUPERSEDED
    if status == "ACTIVE":
        for r in reg.get("runs", []):
            if r.get("dataset_id") == dataset_id and r.get("is_active"):
                r["is_active"] = False
                r["status"] = "SUPERSEDED"

    target_col = config_dict.get("target", {}).get("column") if isinstance(config_dict.get("target"), dict) else config_dict.get("target_column")
    pos_class = config_dict.get("target", {}).get("positive_class") if isinstance(config_dict.get("target"), dict) else config_dict.get("positive_class")
    now_iso = datetime.now(timezone.utc).isoformat()

    record = {
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "run_id": run_id,
        "model_id": f"{dataset_id}_{version}",
        "model_version": version,
        "model_type": model_type,
        "selected_model": is_selected,
        "is_selected": is_selected,
        "is_active": (status == "ACTIVE"),
        "status": status,
        "timestamp": now_iso,
        "training_timestamp": now_iso,
        "training_config": config_dict,
        "selection_policy": selection_policy,
        "model_selection_policy": selection_policy,
        "protected_attributes": config_dict.get("protected_attributes", []),
        "target_column": target_col,
        "positive_class": pos_class,
        "train_sample_count": train_count,
        "test_sample_count": test_count,
        "feature_count": feature_count,
        "random_seed": random_seed,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "fairness_metrics": fairness_metrics,
        "calibration_metrics": calibration_metrics,
        "model_artifact_path": artifact_path,
        "preprocessing_artifact_path": config_dict.get("preprocessing_artifact_path", None)
    }

    reg["runs"].append(record)

    # Update dataset summary in registry
    if dataset_id not in reg["datasets"]:
        reg["datasets"][dataset_id] = {
            "dataset_id": dataset_id,
            "dataset_hash": dataset_hash,
            "latest_version": version if version != "NONE" else "v0",
            "total_runs": 1
        }
    else:
        reg["datasets"][dataset_id]["dataset_hash"] = dataset_hash
        if version != "NONE":
            reg["datasets"][dataset_id]["latest_version"] = version
        reg["datasets"][dataset_id]["total_runs"] += 1

    save_model_registry(reg, base_dir=base_dir)
    return record


def get_dataset_registered_versions(dataset_id: str, base_dir: str = ".") -> List[Dict[str, Any]]:
    """
    Retrieve all registered model versions for dataset_id sorted by timestamp descending.

    Parameters:
        dataset_id (str): Dataset identifier string.
        base_dir (str): Base project directory path.

    Returns:
        list: List of model record dictionaries.
    """
    reg = load_model_registry(base_dir=base_dir)
    runs = [r for r in reg.get("runs", []) if r.get("dataset_id") == dataset_id]
    runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return runs


def set_active_model_version(dataset_id: str, model_version: str, base_dir: str = ".") -> bool:
    """
    Activate a specific model version for dataset_id and supersede others.

    Parameters:
        dataset_id (str): Dataset identifier string.
        model_version (str): Model version string (e.g. 'v1', 'v2').
        base_dir (str): Base project directory path.

    Returns:
        bool: True if version activated successfully, False otherwise.
    """
    reg = load_model_registry(base_dir=base_dir)
    runs = reg.get("runs", [])

    found = False
    for r in runs:
        if r.get("dataset_id") == dataset_id:
            if r.get("model_version") == model_version:
                r["is_active"] = True
                r["status"] = "ACTIVE"
                found = True
            else:
                r["is_active"] = False
                if r.get("status") == "ACTIVE":
                    r["status"] = "SUPERSEDED"

    if found:
        save_model_registry(reg, base_dir=base_dir)

    return found
