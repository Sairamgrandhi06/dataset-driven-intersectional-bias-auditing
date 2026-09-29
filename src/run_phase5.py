"""
Phase 5 Execution Script: Explicit Accuracy, Calibration, and Fairness Trade-off Reporting.

This script executes the complete Phase 5 workflow:
1. Loads dataset and performs reproducible train/test split (80% train, 20% test).
2. Trains baseline Logistic Regression model and Fairlearn mitigated model.
3. Generates test set predictions and prediction probabilities for both models.
4. Evaluates ML performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
5. Evaluates probability calibration metrics (Brier Score, Expected Calibration Error, Reliability Curves).
6. Conducts intersectional fairness audits (Sex x Race, N >= 30) for baseline and mitigated predictions.
7. Computes three-way trade-off report (Baseline vs. Mitigated metrics, absolute changes, % changes, impact status).
8. Exports:
   - 'data/processed/tradeoff_results.json'
   - 'data/processed/tradeoff_summary.csv'
9. Renders visual charts under 'results/figures/':
   - 'baseline_vs_mitigated_calibration.png'
   - 'probability_distributions.png'
   - 'tradeoff_summary_chart.png'
   - 'baseline_vs_mitigated_performance.png'
   - 'baseline_vs_mitigated_fairness.png'
10. Prints comprehensive report and automated objective interpretation to stdout.
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
from src.models.classifier import train_baseline_model, predict_model, evaluate_performance_metrics
from src.fairness.intersectional import create_intersectional_attribute, audit_intersectional_attributes
from src.fairness.calibration import evaluate_calibration, compare_calibration
from src.fairness.trade_off import (
    calculate_comprehensive_tradeoff,
    generate_tradeoff_summary_df,
    generate_tradeoff_interpretation
)
from src.mitigation.mitigator import train_mitigated_model, predict_mitigated_model
from src.visualization.plots import (
    plot_calibration_curves,
    plot_probability_distributions,
    plot_tradeoff_summary_chart,
    plot_baseline_vs_mitigated_performance,
    plot_baseline_vs_mitigated_fairness
)


def run_phase5_pipeline(
    config_path="config/default_config.json",
    output_json_path="data/processed/tradeoff_results.json",
    output_csv_path="data/processed/tradeoff_summary.csv",
    output_fig_dir="results/figures",
    constraint_type="EqualizedOdds",
    eps=0.01,
    min_group_size=30
):
    """
    Execute the Phase 5 Trade-off Evaluation & Reporting pipeline.
    """
    print("=" * 85)
    print("PHASE 5: Explicit Accuracy, Calibration, and Fairness Trade-off Reporting")
    print("=" * 85)

    # 1. Load configuration and dataset
    config = load_config(config_path)
    print(f"\n[1/7] Loading dataset from {config['dataset']['raw_path']}...")
    raw_df = load_raw_data(config_path)
    X, y, A = prepare_pipeline_data(raw_df, config=config)

    splits = split_pipeline_data(X, y, A, config=config)
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]
    A_train, A_test = splits["A_train"], splits["A_test"]

    print(f"      Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    # 2. Train and evaluate Baseline Model
    seed = config.get("random_seed", 42)
    print("\n[2/7] Training Baseline Classifier & Generating Predictions...")
    base_model = train_baseline_model(X_train, y_train, random_state=seed, max_iter=1000)
    y_base_pred, y_base_prob = predict_model(base_model, X_test)

    base_perf = evaluate_performance_metrics(y_test, y_base_pred, y_base_prob)
    base_calib = evaluate_calibration(y_test, y_base_prob, n_bins=10)
    base_intersectional_audit = audit_intersectional_attributes(y_test, y_base_pred, A_test, min_group_size=min_group_size)

    # 3. Train and evaluate Mitigated Model (Training data ONLY)
    print(f"\n[3/7] Training Fairlearn Mitigated Classifier ({constraint_type}, eps={eps})...")
    sens_train = create_intersectional_attribute(A_train, attributes=["sex", "race"])

    mit_model = train_mitigated_model(
        X_train,
        y_train,
        sensitive_features=sens_train,
        constraint_type=constraint_type,
        eps=eps,
        random_state=seed
    )

    y_mit_pred, y_mit_prob = predict_mitigated_model(mit_model, X_test, random_state=0)
    mit_perf = evaluate_performance_metrics(y_test, y_mit_pred, y_mit_prob)
    mit_calib = evaluate_calibration(y_test, y_mit_prob, n_bins=10)
    mit_intersectional_audit = audit_intersectional_attributes(y_test, y_mit_pred, A_test, min_group_size=min_group_size)

    # 4. Compute Three-Way Comprehensive Trade-off
    print("\n[4/7] Computing Three-Way Trade-off (Performance vs. Calibration vs. Fairness)...")
    tradeoff_report = calculate_comprehensive_tradeoff(
        base_perf,
        mit_perf,
        base_intersectional_audit,
        mit_intersectional_audit,
        base_calib,
        mit_calib
    )

    # 5. Render Calibration & Trade-off Figures
    print(f"\n[5/7] Rendering calibration & trade-off figures under '{output_fig_dir}'...")
    fig_calib = plot_calibration_curves(base_calib, mit_calib, output_path=os.path.join(output_fig_dir, "baseline_vs_mitigated_calibration.png"))
    fig_dist = plot_probability_distributions(y_base_prob, y_mit_prob, output_path=os.path.join(output_fig_dir, "probability_distributions.png"))
    fig_trade = plot_tradeoff_summary_chart(tradeoff_report, output_path=os.path.join(output_fig_dir, "tradeoff_summary_chart.png"))
    plot_baseline_vs_mitigated_performance(base_perf, mit_perf, output_path=os.path.join(output_fig_dir, "baseline_vs_mitigated_performance.png"))
    plot_baseline_vs_mitigated_fairness(base_intersectional_audit, mit_intersectional_audit, output_path=os.path.join(output_fig_dir, "baseline_vs_mitigated_fairness.png"))

    # 6. Console Report Output
    print("\n" + "=" * 90)
    print("PHASE 5: COMPREHENSIVE THREE-WAY TRADE-OFF REPORT TABLE")
    print("=" * 90)
    print(f"{'Category':<15} | {'Metric Name':<32} | {'Baseline':<10} | {'Mitigated':<10} | {'Abs Delta':<10} | {'% Delta':<9} | {'Status'}")
    print("-" * 90)

    summary_df = generate_tradeoff_summary_df(tradeoff_report)
    for _, row in summary_df.iterrows():
        print(f"{row['category']:<15} | {row['metric_name']:<32} | {row['baseline_value']:<10.4f} | {row['mitigated_value']:<10.4f} | {row['absolute_change']:<+10.4f} | {row['percentage_change']:<+8.1f}% | {row['impact_status']}")

    print("\n--- Objective Automated Interpretation ---")
    print(tradeoff_report["interpretation"])

    # 7. Serialize Output Files
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    full_json_output = {
        "phase": 5,
        "mitigation_configuration": {
            "algorithm": "Fairlearn ExponentiatedGradient",
            "constraint": constraint_type,
            "eps": eps
        },
        "performance_metrics": {
            "baseline": base_perf,
            "mitigated": mit_perf,
            "comparison": tradeoff_report["performance"]
        },
        "calibration_metrics": {
            "baseline": base_calib,
            "mitigated": mit_calib,
            "comparison": tradeoff_report["calibration"]
        },
        "fairness_metrics": {
            "baseline": base_intersectional_audit["disparities"],
            "mitigated": mit_intersectional_audit["disparities"],
            "comparison": tradeoff_report["fairness"]
        },
        "tradeoff_interpretation": tradeoff_report["interpretation"],
        "intersectional_threshold_policy": tradeoff_report["intersectional_threshold_policy"],
        "limitations": [
            "In-processing mitigation imposes a trade-off between fairness constraints and raw predictive performance.",
            "Probability calibration (Brier Score) can shift when training sample weights are optimized for equalized odds.",
            "Subgroup metric calculations apply a minimum group size threshold (N >= 30) to prevent statistical volatility."
        ]
    }

    with open(output_json_path, "w") as f:
        json.dump(full_json_output, f, indent=2)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    summary_df.to_csv(output_csv_path, index=False)

    print(f"\nJSON results saved: '{output_json_path}'")
    print(f"CSV summary saved : '{output_csv_path}'")
    print("=" * 90)

    return full_json_output


if __name__ == "__main__":
    run_phase5_pipeline()
