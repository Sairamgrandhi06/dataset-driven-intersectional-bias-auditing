"""
Deterministic Pipeline Replay & Idempotency Testing Module.

Executes multiple pipeline runs (Run A, Run B, Run C) to prove deterministic execution
and output idempotency. Distinguishes exact byte equality vs numerical equality.
Outputs 'evidence/reliability/replay_report.json'.
"""

import json
import os
import time
from typing import Any, Dict, List

from src.data.quality_gates import compute_file_sha256
from src.data.loader import load_raw_data, load_config
from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data
from src.models.classifier import train_baseline_model, predict_model
from src.fairness.intersectional import audit_intersectional_attributes


def run_pipeline_replay_verification(n_runs: int = 3, tolerance: float = 1e-4) -> Dict[str, Any]:
    """
    Execute n_runs of the baseline model audit and verify output idempotency.

    Returns:
        dict: Replay verification report.
    """
    raw_path = "data/raw/adult.csv"
    dataset_hash = compute_file_sha256(raw_path)

    run_results = []

    for i in range(n_runs):
        config = load_config("config/default_config.json")
        raw_df = load_raw_data("config/default_config.json")
        X, y, A = prepare_pipeline_data(raw_df, config=config)
        splits = split_pipeline_data(X, y, A, config=config)

        model = train_baseline_model(splits["X_train"], splits["y_train"], random_state=42)
        y_pred, _ = predict_model(model, splits["X_test"])

        audit = audit_intersectional_attributes(splits["y_test"], y_pred, splits["A_test"], min_group_size=30)
        run_results.append({
            "run_index": i + 1,
            "disparities": audit["disparities"],
            "total_groups": audit["total_groups_discovered"]
        })

    # Compare Run 1 vs Run 2 vs Run 3
    base_disp = run_results[0]["disparities"]
    all_numerical_match = True
    all_exact_byte_match = True

    for res in run_results[1:]:
        disp = res["disparities"]
        for k, v in base_disp.items():
            if abs(v - disp[k]) > tolerance:
                all_numerical_match = False
            if v != disp[k]:
                all_exact_byte_match = False

    report = {
        "dataset_hash": dataset_hash,
        "n_runs_executed": n_runs,
        "tolerance": tolerance,
        "all_numerical_match": all_numerical_match,
        "all_exact_byte_match": all_exact_byte_match,
        "reproducibility_status": "PASS" if all_numerical_match else "FAIL",
        "runs": run_results
    }

    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evidence", "reliability", "replay_report.json"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass

    return report
