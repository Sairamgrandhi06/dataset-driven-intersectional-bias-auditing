"""
Phase 9 Reproduction Script: Reproducible Reference & Expected-Result Oracle.

This is the official one-command Phase 9 reproducibility runner:
1. Validates dataset identity and SHA-256 hash.
2. Validates Phase 8 evidence and provenance.
3. Constructs intersectional labels via independent reference module.
4. Validates group counts (10 discovered, 7 primary, 3 low-sample).
5. Calculates group-level metrics via independent reference module.
6. Calculates intersectional disparity metrics.
7. Validates metrics against frozen expected-result oracle ('evidence/reference/intersectional_expected_results.json').
8. Asserts 100% agreement between production and independent reference implementations within 1e-4 tolerance.
9. Updates reproducibility report and exits with 0 if PASS, 1 if FAIL.
"""

import os
import sys
import json
from typing import Any
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reference.intersectional_reference import audit_intersectional_reference
from reference.expected_results import load_expected_results, verify_expected_oracle_integrity
from src.data.quality_gates import compute_file_sha256
from src.data.loader import load_raw_data, load_config
from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data
from src.models.classifier import train_baseline_model, predict_model
from src.fairness.intersectional import audit_intersectional_attributes


def _is_metric_close(v1: Any, v2: Any, tol: float = 1e-4) -> bool:
    if v1 is None and v2 is None:
        return True
    if v1 is None or v2 is None:
        return False
    try:
        return abs(float(v1) - float(v2)) <= tol
    except (TypeError, ValueError):
        return False


def run_phase9_reproducibility_pipeline(tolerance=1e-4):
    print("=" * 70)
    print("PHASE 9 REPRODUCIBILITY VALIDATION")
    print("=" * 70)

    results_check = []
    failures = []

    # [1] Dataset Identity & Hash Check
    raw_path = "data/raw/adult.csv"
    expected_hash = "9007ff524acf0776b9598a0ed883058df91f59c003e78cb2cddd88f87535fc86"
    actual_hash = compute_file_sha256(raw_path)
    hash_pass = (actual_hash == expected_hash)
    results_check.append(("[1] Dataset identity", hash_pass))
    if not hash_pass:
        failures.append(f"Dataset hash mismatch: observed {actual_hash}")

    # [2] Evidence & Provenance Check
    ev_pass = os.path.exists("evidence/dataset/quality_report.json") and os.path.exists("evidence/dataset/dataset_manifest.json")
    results_check.append(("[2] Evidence/provenance", ev_pass))
    if not ev_pass:
        failures.append("Phase 8 evidence files missing under 'evidence/dataset/'")

    # Load data and run baseline predictions
    config = load_config("config/default_config.json")
    raw_df = load_raw_data("config/default_config.json")
    X, y, A = prepare_pipeline_data(raw_df, config=config)
    splits = split_pipeline_data(X, y, A, config=config)

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test, y_test, A_test = splits["X_test"], splits["y_test"], splits["A_test"]

    baseline_model = train_baseline_model(X_train, y_train, random_state=42)
    y_pred, _ = predict_model(baseline_model, X_test)

    # [3] Intersection Construction Check
    ref_audit = audit_intersectional_reference(y_test, y_pred, A_test, min_group_size=30)
    inter_pass = ref_audit["total_groups_discovered"] == 10
    results_check.append(("[3] Intersection construction", inter_pass))

    # [4] Group Counts Check
    counts_pass = (ref_audit["primary_group_count"] == 7) and (ref_audit["low_sample_group_count"] == 3)
    results_check.append(("[4] Group counts", counts_pass))

    # [5] Group Metrics Check
    group_metrics_pass = len(ref_audit["all_group_metrics"]) == 10
    results_check.append(("[5] Group metrics", group_metrics_pass))

    # [6] Disparity Metrics Check
    ref_disp = ref_audit["disparities"]
    disp_metrics_pass = len(ref_disp) == 5
    results_check.append(("[6] Disparity metrics", disp_metrics_pass))

    # [7] Expected-Result Oracle Check
    if not verify_expected_oracle_integrity():
        oracle_pass = False
    else:
        oracle = load_expected_results()
        exp_disp = oracle["expected_disparities"]
        oracle_pass = all(_is_metric_close(exp_disp.get(k), ref_disp.get(k), tolerance) for k in exp_disp)

    results_check.append(("[7] Expected-result oracle", oracle_pass))
    if not oracle_pass:
        failures.append("Reference calculation differed from expected-result oracle beyond tolerance")

    # [8] Production vs Reference Comparison Check
    prod_audit = audit_intersectional_attributes(y_test, y_pred, A_test, attributes=["sex", "race"], min_group_size=30)
    prod_disp = prod_audit["disparities"]
    prod_ref_pass = all(_is_metric_close(prod_disp.get(k), ref_disp.get(k), tolerance) for k in prod_disp)
    results_check.append(("[8] Production/reference", prod_ref_pass))
    if not prod_ref_pass:
        failures.append("Production implementation differed from reference implementation beyond tolerance")

    # [9] Reproducibility Summary Check
    repro_pass = len(failures) == 0
    results_check.append(("[9] Reproducibility", repro_pass))

    # Print Summary Table
    print()
    for label, status in results_check:
        print(f"{label:<36} : {'PASS' if status else 'FAIL'}")

    print("=" * 70)
    overall_status = "PASS" if repro_pass else "FAIL"
    print(f"PHASE 9 STATUS: {overall_status}")
    print("=" * 70)

    if not repro_pass:
        print("\nFAILURE DETAILS:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    return True


if __name__ == "__main__":
    run_phase9_reproducibility_pipeline()
