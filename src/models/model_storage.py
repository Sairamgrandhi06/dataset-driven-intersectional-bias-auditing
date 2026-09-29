"""
Model Storage & Run Manifest Serialization Module.

Persists trained model artifacts using joblib under models/<dataset_id>/run_<id>/ and generates
dataset-isolated audit run manifests under evidence/runs/<dataset_id>/<run_id>/.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import joblib


def save_model_run_artifacts(
    dataset_id: str,
    run_id: str,
    candidate_models: Dict[str, Any],
    selected_model_key: Optional[str],
    selection_result: Dict[str, Any],
    dataset_hash: str,
    config_dict: Dict[str, Any],
    base_dir: str = "."
) -> Dict[str, str]:
    """
    Serialize candidate models and selection metadata for the current run.

    Parameters:
        dataset_id (str): Dataset identifier string.
        run_id (str): Unique run identifier string.
        candidate_models (dict): Map of model_key -> fitted scikit-learn model object.
        selected_model_key (str, optional): Key of selected model.
        selection_result (dict): Model selection output report.
        dataset_hash (str): SHA-256 hash of dataset.
        config_dict (dict): Pipeline configuration dictionary.
        base_dir (str): Project root directory path.

    Returns:
        dict: Absolute file paths to saved artifacts.
    """
    run_folder = run_id if run_id.startswith("run_") else f"run_{run_id}"
    model_dir = os.path.abspath(os.path.join(base_dir, "models", dataset_id, run_folder))
    evidence_dir = os.path.abspath(os.path.join(base_dir, "evidence", "runs", dataset_id, run_id))

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(evidence_dir, exist_ok=True)

    saved_paths = {}

    # Save each candidate model using joblib
    for m_key, model_obj in candidate_models.items():
        m_path = os.path.join(model_dir, f"{m_key}.joblib")
        joblib.dump(model_obj, m_path)
        saved_paths[f"model_{m_key}"] = m_path

    # Save selected model copy
    if selected_model_key and selected_model_key in candidate_models:
        sel_path = os.path.join(model_dir, "selected_model.joblib")
        joblib.dump(candidate_models[selected_model_key], sel_path)
        saved_paths["selected_model"] = sel_path

    # Save model_metadata.json
    meta_path = os.path.join(model_dir, "model_metadata.json")
    meta_content = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": dataset_hash,
        "selected_model": selected_model_key,
        "selection_status": selection_result.get("selection_status"),
        "selection_reason": selection_result.get("reason"),
        "comparison_table": selection_result.get("comparison_table", []),
        "config": config_dict
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_content, f, indent=2)
    saved_paths["model_metadata"] = meta_path

    # Save run_manifest.json under evidence/runs/<dataset_id>/<run_id>/
    manifest_path = os.path.join(evidence_dir, "run_manifest.json")
    manifest_content = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "dataset_hash": dataset_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected_model_key,
        "candidate_models": list(candidate_models.keys()),
        "model_artifacts_directory": model_dir,
        "selection_status": selection_result.get("selection_status"),
        "target_column": config_dict.get("target", {}).get("column") if isinstance(config_dict.get("target"), dict) else config_dict.get("target_column"),
        "positive_class": config_dict.get("target", {}).get("positive_class") if isinstance(config_dict.get("target"), dict) else config_dict.get("positive_class"),
        "protected_attributes": config_dict.get("protected_attributes", [])
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_content, f, indent=2)
    saved_paths["run_manifest"] = manifest_path

    return saved_paths

    return saved_paths


def get_dataset_model_runs(dataset_id: str, base_dir: str = ".") -> List[Dict[str, Any]]:
    """Scan models/<dataset_id>/ for previous model runs."""
    ds_model_dir = os.path.abspath(os.path.join(base_dir, "models", dataset_id))
    runs = []

    if not os.path.exists(ds_model_dir):
        return runs

    for folder in os.listdir(ds_model_dir):
        run_path = os.path.join(ds_model_dir, folder)
        if os.path.isdir(run_path):
            meta_file = os.path.join(run_path, "model_metadata.json")
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        runs.append(json.load(f))
                except Exception:
                    pass

    runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return runs


def load_saved_model_run(dataset_id: str, run_id: str, base_dir: str = ".") -> Dict[str, Any]:
    """
    Load saved model estimator, metadata, and manifest for a given dataset ID and run ID.

    Parameters:
        dataset_id (str): Dataset identifier string.
        run_id (str): Run identifier string.
        base_dir (str): Project root directory path.

    Returns:
        dict: Loaded run components containing fitted_model, metadata, manifest, and preprocessor.
    """
    run_folder = run_id if run_id.startswith("run_") else f"run_{run_id}"
    model_dir = os.path.abspath(os.path.join(base_dir, "models", dataset_id, run_folder))
    evidence_dir = os.path.abspath(os.path.join(base_dir, "evidence", "runs", dataset_id, run_id))

    res = {
        "fitted_model": None,
        "preprocessor": None,
        "metadata": {},
        "manifest": {}
    }

    # Load model_metadata.json
    meta_path = os.path.join(model_dir, "model_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                res["metadata"] = json.load(f)
        except Exception:
            pass

    # Load run_manifest.json
    manifest_path = os.path.join(evidence_dir, "run_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                res["manifest"] = json.load(f)
        except Exception:
            pass

    # Load selected model estimator joblib
    sel_path = os.path.join(model_dir, "selected_model.joblib")
    if not os.path.exists(sel_path):
        sel_key = res["metadata"].get("selected_model")
        if sel_key:
            sel_path = os.path.join(model_dir, f"{sel_key}.joblib")

    if os.path.exists(sel_path):
        try:
            res["fitted_model"] = joblib.load(sel_path)
        except Exception:
            pass

    # Load preprocessor joblib if saved
    prep_path = os.path.join(model_dir, "preprocessor.joblib")
    if os.path.exists(prep_path):
        try:
            res["preprocessor"] = joblib.load(prep_path)
        except Exception:
            pass

    return res

