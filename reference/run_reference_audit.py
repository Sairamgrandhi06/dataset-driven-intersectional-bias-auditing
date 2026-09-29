"""
Standalone Reference Audit Runner Script.

Executes the independent intersectional reference audit against the test set data,
verifies outputs against the frozen expected-result oracle ('evidence/reference/intersectional_expected_results.json'),
and prints a structured verification report. Exits with 0 if PASS, 1 if FAIL.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reference.intersectional_reference import audit_intersectional_reference
from reference.expected_results import load_expected_results, verify_expected_oracle_integrity
from src.data.loader import load_raw_data, load_config
from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data
from src.models.classifier import train_baseline_model, predict_model


def run_standalone_reference_audit():
    print("=" * 70)
    print("PHASE 9 INDEPENDENT REFERENCE AUDIT")
    print("=" * 70)

    # 1. Verify Oracle Integrity
    if not verify_expected_oracle_integrity():
        print("ERROR: Frozen expected-result oracle is corrupt or missing!")
        sys.exit(1)

    expected_oracle = load_expected_results()
    meta = expected_oracle["oracle_metadata"]

    print(f"Dataset hash            : {meta['dataset_hash']}")
    print(f"Test samples            : {meta['test_sample_count']}")
    print(f"Minimum group threshold : {meta['min_group_size_threshold']}")

    # 2. Load dataset and baseline predictions
    config = load_config("config/default_config.json")
    raw_df = load_raw_data("config/default_config.json")
    X, y, A = prepare_pipeline_data(raw_df, config=config)
    splits = split_pipeline_data(X, y, A, config=config)

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test, y_test, A_test = splits["X_test"], splits["y_test"], splits["A_test"]

    baseline_model = train_baseline_model(X_train, y_train, random_state=42)
    y_pred, _ = predict_model(baseline_model, X_test)

    # 3. Run Independent Reference Audit
    ref_results = audit_intersectional_reference(
        y_test,
        y_pred,
        A_test,
        min_group_size=meta["min_group_size_threshold"]
    )

    print(f"\nIntersectional groups discovered : {ref_results['total_groups_discovered']}")
    print(f"Primary groups (N >= {meta['min_group_size_threshold']})           : {ref_results['primary_group_count']}")
    print(f"Low-sample groups (N < {meta['min_group_size_threshold']})            : {ref_results['low_sample_group_count']}")

    print("\nReference calculations:")

    exp_disp = expected_oracle["expected_disparities"]
    ref_disp = ref_results["disparities"]
    tol = 1e-4

    checks = [
        ("Demographic Parity", exp_disp["demographic_parity_difference"], ref_disp["demographic_parity_difference"]),
        ("Disparate Impact", exp_disp["disparate_impact_ratio"], ref_disp["disparate_impact_ratio"]),
        ("Equal Opportunity", exp_disp["equal_opportunity_difference"], ref_disp["equal_opportunity_difference"]),
        ("Equalized Odds", exp_disp["equalized_odds_difference"], ref_disp["equalized_odds_difference"]),
        ("FPR Difference", exp_disp["false_positive_rate_difference"], ref_disp["false_positive_rate_difference"])
    ]

    all_pass = True
    for label, exp_v, ref_v in checks:
        diff = abs(exp_v - ref_v)
        status = "PASS" if diff <= tol else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  - {label:<22} : {status} (Expected: {exp_v:.4f}, Ref: {ref_v:.4f}, Diff: {diff:.6f})")

    print("\nExpected-result oracle :", "PASS" if all_pass else "FAIL")
    print("Production vs reference :", "PASS" if all_pass else "FAIL")
    print("-" * 70)

    if all_pass:
        print("PHASE 9 STATUS: PASS")
        print("=" * 70)
        sys.exit(0)
    else:
        print("PHASE 9 STATUS: FAIL")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    run_standalone_reference_audit()
