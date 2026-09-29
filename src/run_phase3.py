"""
Phase 3 Execution Script: Intersectional Fairness Audit (Sex x Race).

This script executes the complete Phase 3 workflow:
1. Loads raw dataset and preprocesses features (X), binary outcome (y), and protected attributes (A).
2. Uses reproducible train/test split (80% train, 20% test).
3. Trains baseline Logistic Regression classifier and generates test set predictions.
4. Conducts intersectional fairness audit across Sex x Race combinations.
5. Applies statistical safety thresholding (min_group_size=30) to partition primary vs low-sample groups.
6. Serializes audit results to 'data/processed/intersectional_results.json'.
7. Exports tabular group metrics to 'data/processed/intersectional_group_metrics.csv'.
8. Generates static visual charts under 'results/figures/'.
9. Prints comprehensive audit summary to stdout.
"""

import json
import os
import sys
import warnings
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Add project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import load_raw_data, load_config
from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data
from src.models.classifier import train_baseline_model, predict_model
from src.fairness.intersectional import audit_intersectional_attributes
from src.visualization.plots import plot_all_intersectional_figures


def run_phase3_pipeline(
    config_path="config/default_config.json",
    output_json_path="data/processed/intersectional_results.json",
    output_csv_path="data/processed/intersectional_group_metrics.csv",
    output_fig_dir="results/figures",
    min_group_size=30
):
    """
    Execute the Phase 3 Intersectional Fairness Audit pipeline.

    Parameters:
        config_path (str): Path to JSON configuration file.
        output_json_path (str): Destination path for intersectional audit JSON results.
        output_csv_path (str): Destination path for CSV table.
        output_fig_dir (str): Destination directory for chart figures.
        min_group_size (int): Minimum group size threshold.

    Returns:
        dict: Intersectional audit results dictionary.
    """
    print("=" * 80)
    print("PHASE 3: Intersectional Fairness Audit (Sex x Race)")
    print("=" * 80)

    # 1. Load configuration and raw dataset
    config = load_config(config_path)
    print(f"\n[1/6] Loading dataset from {config['dataset']['raw_path']}...")
    raw_df = load_raw_data(config_path)
    print(f"      Raw dataset loaded: {len(raw_df):,} rows.")

    # 2. Preprocess data into X, y, A
    print("\n[2/6] Cleaning missing values and preprocessing features...")
    X, y, A = prepare_pipeline_data(raw_df, config=config)
    print(f"      Feature matrix X shape: {X.shape}")
    print(f"      Target vector y shape: {y.shape}")
    print(f"      Protected attributes A shape: {A.shape}")

    # 3. Train-test split
    print("\n[3/6] Performing reproducible train/test split...")
    splits = split_pipeline_data(X, y, A, config=config)
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]
    A_train, A_test = splits["A_train"], splits["A_test"]

    print(f"      Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    # 4. Model Training & Prediction
    print("\n[4/6] Training baseline Logistic Regression classifier...")
    seed = config.get("random_seed", 42)
    model = train_baseline_model(X_train, y_train, random_state=seed, max_iter=1000)
    y_pred, y_prob = predict_model(model, X_test)

    # 5. Intersectional Fairness Auditing
    print(f"\n[5/6] Auditing intersectional combinations (Sex x Race) [Min Group Threshold N >= {min_group_size}]...")
    audit_results = audit_intersectional_attributes(
        y_true=y_test,
        y_pred=y_pred,
        A_df=A_test,
        attributes=["sex", "race"],
        min_group_size=min_group_size
    )

    # Display audit findings
    total_groups = audit_results["total_groups_discovered"]
    primary_count = audit_results["primary_group_count"]
    low_sample_count = audit_results["low_sample_group_count"]

    print(f"\n--- Intersectional Subgroups Summary ---")
    print(f"Total Groups Discovered: {total_groups}")
    print(f"Primary Groups (N >= {min_group_size}): {primary_count}")
    print(f"Low-Sample Groups (N < {min_group_size}): {low_sample_count}")

    print("\n--- Group Metrics Table ---")
    print(f"{'Group Name':<32} | {'N':<6} | {'Selection Rate':<14} | {'TPR (Recall)':<12} | {'FPR':<8} | {'Status'}")
    print("-" * 90)

    csv_records = []
    all_groups = audit_results["all_group_metrics"]
    for grp, m in all_groups.items():
        is_primary = m["sample_count"] >= min_group_size
        status = "Primary" if is_primary else "Low-Sample"
        print(f"{grp:<32} | {m['sample_count']:<6} | {m['selection_rate']:<14.4f} | {m['true_positive_rate']:<12.4f} | {m['false_positive_rate']:<8.4f} | {status}")

        csv_records.append({
            "intersectional_group": grp,
            "sample_count": m["sample_count"],
            "selection_rate": m["selection_rate"],
            "true_positive_rate": m["true_positive_rate"],
            "false_positive_rate": m["false_positive_rate"],
            "false_negative_rate": m["false_negative_rate"],
            "true_negative_rate": m["true_negative_rate"],
            "TP": m["confusion_matrix"]["TP"],
            "FP": m["confusion_matrix"]["FP"],
            "TN": m["confusion_matrix"]["TN"],
            "FN": m["confusion_matrix"]["FN"],
            "analysis_status": status
        })

    print("\n--- Intersectional Disparities (Primary Groups N >= {}) ---".format(min_group_size))
    disparities = audit_results["disparities"]
    for disp_k, disp_v in disparities.items():
        if isinstance(disp_v, (int, float)):
            print(f"  - {disp_k:<32}: {disp_v:.4f}")
        else:
            print(f"  - {disp_k:<32}: {disp_v}")

    print("\n--- Group Rankings & Extremes ---")
    rankings = audit_results["rankings"]
    if isinstance(rankings, dict) and rankings.get("status") == "AVAILABLE":
        def _fmt(key: str) -> str:
            item = rankings.get(key)
            if isinstance(item, dict):
                group = item.get("group", "N/A")
                val = item.get("value")
                val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                return f"{group} ({val_str})"
            return "N/A"

        def _fmt_custom(key: str, label: str) -> str:
            item = rankings.get(key)
            if isinstance(item, dict):
                group = item.get("group", "N/A")
                val = item.get("value")
                val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                return f"{group} ({label} = {val_str})"
            return "N/A"

        print(f"  - Highest Selection Rate : {_fmt('highest_selection_rate')}")
        print(f"  - Lowest Selection Rate  : {_fmt('lowest_selection_rate')}")
        print(f"  - Highest TPR            : {_fmt('highest_tpr')}")
        print(f"  - Lowest TPR             : {_fmt('lowest_tpr')}")
        print(f"  - Highest FPR            : {_fmt('highest_fpr')}")
        print(f"  - Lowest FPR             : {_fmt('lowest_fpr')}")
        print(f"  - Most Disadvantaged Subgroup : {_fmt_custom('most_disadvantaged_group', 'Selection Rate')}")
        print(f"  - Highest Performing Subgroup : {_fmt_custom('highest_performing_group', 'Selection Rate')}")
    else:
        reason = rankings.get("reason", "N/A") if isinstance(rankings, dict) else "N/A"
        print(f"  - Rankings unavailable: {reason}")

    if audit_results["low_sample_groups"]:
        print(f"\n--- Low-Sample Groups (Excluded from primary disparity bounds) ---")
        for lsg in audit_results["low_sample_groups"]:
            m = all_groups[lsg]
            print(f"  - {lsg} (N={m['sample_count']}): Selection Rate={m['selection_rate']:.4f}, TPR={m['true_positive_rate']:.4f}")

    # 6. Generate Figures
    print("\n[6/6] Generating visualization figures under '{}'...".format(output_fig_dir))
    plot_all_intersectional_figures(audit_results, output_dir=output_fig_dir)

    # Export JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(audit_results, f, indent=2)

    # Export CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    csv_df = pd.DataFrame(csv_records)
    csv_df.to_csv(output_csv_path, index=False)

    print(f"\nJSON results saved: '{output_json_path}'")
    print(f"CSV table saved   : '{output_csv_path}'")
    print("=" * 80)

    return audit_results


if __name__ == "__main__":
    run_phase3_pipeline()
