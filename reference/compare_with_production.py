"""
Production vs. Independent Reference Reconciliation Script.

Directly compares the production audit module ('src/fairness/intersectional.py')
against the independent reference implementation ('reference/intersectional_reference.py').
Asserts agreement across all group counts, metrics, and disparity bounds within 1e-4 tolerance.
Exits 0 if MATCH, 1 if MISMATCH.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reference.intersectional_reference import audit_intersectional_reference
from src.fairness.intersectional import audit_intersectional_attributes
from src.data.loader import load_raw_data, load_config
from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data
from src.models.classifier import train_baseline_model, predict_model


def compare_production_vs_reference(tolerance=1e-4):
    print("=" * 80)
    print("PRODUCTION VS. INDEPENDENT REFERENCE METRIC RECONCILIATION")
    print("=" * 80)

    # 1. Prepare data and baseline predictions
    config = load_config("config/default_config.json")
    raw_df = load_raw_data("config/default_config.json")
    X, y, A = prepare_pipeline_data(raw_df, config=config)
    splits = split_pipeline_data(X, y, A, config=config)

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test, y_test, A_test = splits["X_test"], splits["y_test"], splits["A_test"]

    baseline_model = train_baseline_model(X_train, y_train, random_state=42)
    y_pred, _ = predict_model(baseline_model, X_test)

    # 2. Run Production Audit
    prod_results = audit_intersectional_attributes(
        y_test, y_pred, A_test, attributes=["sex", "race"], min_group_size=30
    )

    # 3. Run Independent Reference Audit
    ref_results = audit_intersectional_reference(
        y_test, y_pred, A_test, min_group_size=30
    )

    mismatches = []
    comparisons = []

    # Compare Group Counts
    if prod_results["total_groups_discovered"] != ref_results["total_groups_discovered"]:
        mismatches.append(f"Total groups: Prod={prod_results['total_groups_discovered']}, Ref={ref_results['total_groups_discovered']}")
    if prod_results["primary_group_count"] != ref_results["primary_group_count"]:
        mismatches.append(f"Primary groups: Prod={prod_results['primary_group_count']}, Ref={ref_results['primary_group_count']}")

    # Compare Disparity Metrics
    p_disp = prod_results["disparities"]
    r_disp = ref_results["disparities"]

    print(f"{'Metric Name':<32} | {'Production':<12} | {'Reference':<12} | {'Diff':<10} | {'Status':<8}")
    print("-" * 80)

    for k in sorted(p_disp.keys()):
        pv = p_disp[k]
        rv = r_disp[k]
        diff = abs(pv - rv)
        status = "MATCH" if diff <= tolerance else "MISMATCH"

        comparisons.append({
            "metric": k,
            "production": pv,
            "reference": rv,
            "diff": diff,
            "status": status
        })

        if status == "MISMATCH":
            mismatches.append(f"Metric '{k}': Prod={pv:.6f}, Ref={rv:.6f}")

        print(f"{k:<32} | {pv:<12.6f} | {rv:<12.6f} | {diff:<10.6f} | {status:<8}")

    print("-" * 80)
    # Compare Group Metrics across all primary subgroups
    prod_metrics = prod_results["all_group_metrics"]
    ref_metrics = ref_results["all_group_metrics"]
    primary_group_names = prod_results["primary_groups"]

    for g in sorted(primary_group_names):
        p_gm = prod_metrics[g]
        r_gm = ref_metrics[g]

        metric_pairs = [
            ("selection_rate", p_gm.get("selection_rate", 0.0), r_gm["selection_rate"]),
            ("tpr", p_gm.get("true_positive_rate", p_gm.get("tpr", 0.0)), r_gm["tpr"]),
            ("fpr", p_gm.get("false_positive_rate", p_gm.get("fpr", 0.0)), r_gm["fpr"]),
            ("fnr", p_gm.get("false_negative_rate", p_gm.get("fnr", 0.0)), r_gm["fnr"])
        ]

        for sub_k, pv, rv in metric_pairs:
            diff = abs(pv - rv)
            if diff > tolerance:
                mismatches.append(f"Group '{g}' {sub_k}: Prod={pv:.6f}, Ref={rv:.6f}")

    overall_status = "MATCH" if len(mismatches) == 0 else "MISMATCH"
    print(f"Overall Production vs. Reference Status: {overall_status}")
    print("=" * 80)

    if len(mismatches) > 0:
        print("\nMISMATCH DETAILS:")
        for m in mismatches:
            print(f"  - {m}")
        sys.exit(1)

    return True


if __name__ == "__main__":
    compare_production_vs_reference()
