"""
Phase 4 Execution Script: Bias Mitigation & Comprehensive Baseline vs. Mitigated Reporting.

This script executes the complete Phase 4 workflow:
1. Loads raw dataset and performs reproducible train/test split.
2. Fits Fairlearn ExponentiatedGradient in-processing reduction on training set only (Sex x Race).
3. Evaluates mitigated model predictive performance on untouched test set.
4. Re-runs single-attribute and intersectional fairness audits on mitigated model predictions.
5. Generates explicit baseline-vs-mitigated performance and fairness comparisons.
6. Evaluates Pareto front grid points across constraint strictness parameters.
7. Exports:
   - 'data/processed/mitigated_results.json'
   - 'data/processed/mitigated_intersectional_results.json'
   - 'data/processed/baseline_vs_mitigated.csv'
8. Renders comparison figures under 'results/figures/':
   - 'baseline_vs_mitigated_performance.png'
   - 'baseline_vs_mitigated_fairness.png'
   - 'baseline_vs_mitigated_selection_rates.png'
   - 'baseline_vs_mitigated_tpr.png'
   - 'baseline_vs_mitigated_fpr.png'
   - 'mitigation_pareto_front.png'
9. Prints comprehensive summary table to stdout.
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
from src.models.classifier import evaluate_performance_metrics, train_baseline_model, predict_model
from src.fairness.single_attribute import run_single_attribute_audits
from src.fairness.intersectional import create_intersectional_attribute, audit_intersectional_attributes
from src.mitigation.mitigator import (
    train_mitigated_model,
    predict_mitigated_model,
    calculate_mitigation_tradeoff,
    generate_pareto_front_grid
)
from src.visualization.plots import (
    plot_tradeoff_pareto_front,
    plot_baseline_vs_mitigated_performance,
    plot_baseline_vs_mitigated_fairness,
    plot_baseline_vs_mitigated_group_metric
)


def run_phase4_pipeline(
    config_path="config/default_config.json",
    output_mitigated_json="data/processed/mitigated_results.json",
    output_intersectional_json="data/processed/mitigated_intersectional_results.json",
    output_csv_comparison="data/processed/baseline_vs_mitigated.csv",
    output_fig_dir="results/figures",
    constraint_type="EqualizedOdds",
    eps=0.01
):
    """
    Execute the Phase 4 Bias Mitigation & Comparison pipeline.
    """
    print("=" * 80)
    print("PHASE 4: Bias Mitigation & Comprehensive Baseline vs. Mitigated Reporting")
    print("=" * 80)

    # 1. Load data and splits
    config = load_config(config_path)
    print(f"\n[1/7] Loading dataset from {config['dataset']['raw_path']}...")
    raw_df = load_raw_data(config_path)
    X, y, A = prepare_pipeline_data(raw_df, config=config)

    splits = split_pipeline_data(X, y, A, config=config)
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]
    A_train, A_test = splits["A_train"], splits["A_test"]

    print(f"      Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    # 2. Baseline Model Evaluation (Reference)
    seed = config.get("random_seed", 42)
    base_model = train_baseline_model(X_train, y_train, random_state=seed, max_iter=1000)
    y_base_pred, y_base_prob = predict_model(base_model, X_test)
    base_perf = evaluate_performance_metrics(y_test, y_base_pred, y_base_prob)
    base_intersectional_audit = audit_intersectional_attributes(y_test, y_base_pred, A_test, min_group_size=30)

    # 3. Train Mitigated Model on Training Data ONLY
    print(f"\n[2/7] Training Fairlearn ExponentiatedGradient Reduction on Training Data ONLY...")
    print(f"      Constraint: {constraint_type} | eps parameter: {eps}")
    sens_train = create_intersectional_attribute(A_train, attributes=["sex", "race"])

    mit_model = train_mitigated_model(
        X_train,
        y_train,
        sensitive_features=sens_train,
        constraint_type=constraint_type,
        eps=eps,
        random_state=seed
    )

    # 4. Predict & Evaluate Mitigated Model on untouched Test Data
    print("\n[3/7] Generating predictions on untouched test set...")
    y_mit_pred, y_mit_prob = predict_mitigated_model(mit_model, X_test)
    mit_perf = evaluate_performance_metrics(y_test, y_mit_pred, y_mit_prob)

    # Re-run audits on mitigated predictions
    print("\n[4/7] Re-running Single and Intersectional Fairness Audits...")
    mit_single_audits = run_single_attribute_audits(y_test, y_mit_pred, A_test)
    mit_intersectional_audit = audit_intersectional_attributes(y_test, y_mit_pred, A_test, min_group_size=30)

    # 5. Compute Explicit Trade-off Metrics & Comparison Table
    print("\n[5/7] Computing explicit baseline-vs-mitigated comparisons and absolute changes...")
    tradeoff = calculate_mitigation_tradeoff(
        base_perf,
        mit_perf,
        base_intersectional_audit,
        mit_intersectional_audit
    )

    # 6. Evaluate Pareto Front Grid
    print("\n[6/7] Evaluating Pareto front grid across constraint strictness levels...")
    pareto_grid = generate_pareto_front_grid(
        X_train,
        y_train,
        sens_train,
        X_test,
        y_test,
        A_test,
        constraint_type=constraint_type,
        eps_grid=[0.005, 0.01, 0.05, 0.1],
        random_state=seed
    )

    # 7. Render Charts
    print(f"\n[7/7] Rendering 6 comparison figures under '{output_fig_dir}'...")
    fig_perf = plot_baseline_vs_mitigated_performance(base_perf, mit_perf, output_path=os.path.join(output_fig_dir, "baseline_vs_mitigated_performance.png"))
    fig_fair = plot_baseline_vs_mitigated_fairness(base_intersectional_audit, mit_intersectional_audit, output_path=os.path.join(output_fig_dir, "baseline_vs_mitigated_fairness.png"))
    fig_sr = plot_baseline_vs_mitigated_group_metric(base_intersectional_audit, mit_intersectional_audit, metric_key="selection_rate", metric_label="Positive Selection Rate P(y_hat=1)", title="Intersectional Selection Rates: Baseline vs. Mitigated", output_path=os.path.join(output_fig_dir, "baseline_vs_mitigated_selection_rates.png"))
    fig_tpr = plot_baseline_vs_mitigated_group_metric(base_intersectional_audit, mit_intersectional_audit, metric_key="true_positive_rate", metric_label="True Positive Rate (TPR)", title="Intersectional TPR: Baseline vs. Mitigated", output_path=os.path.join(output_fig_dir, "baseline_vs_mitigated_tpr.png"))
    fig_fpr = plot_baseline_vs_mitigated_group_metric(base_intersectional_audit, mit_intersectional_audit, metric_key="false_positive_rate", metric_label="False Positive Rate (FPR)", title="Intersectional FPR: Baseline vs. Mitigated", output_path=os.path.join(output_fig_dir, "baseline_vs_mitigated_fpr.png"))
    fig_pareto = plot_tradeoff_pareto_front(pareto_grid, output_path=os.path.join(output_fig_dir, "mitigation_pareto_front.png"))

    # Display stdout summary
    print("\n" + "=" * 85)
    print("BASELINE vs. MITIGATED MODEL METRIC COMPARISON TABLE")
    print("=" * 85)
    print(f"{'Metric Category':<15} | {'Metric Name':<32} | {'Baseline':<10} | {'Mitigated':<10} | {'Abs Delta':<10} | {'Status'}")
    print("-" * 85)

    comparison_records = []

    # Performance rows
    perf_items = [
        ("Accuracy", base_perf["accuracy"], mit_perf["accuracy"], True),
        ("Precision", base_perf["precision"], mit_perf["precision"], True),
        ("Recall", base_perf["recall"], mit_perf["recall"], True),
        ("F1-Score", base_perf["f1_score"], mit_perf["f1_score"], True),
        ("ROC-AUC", base_perf.get("roc_auc", 0.0) or 0.0, mit_perf.get("roc_auc", 0.0) or 0.0, True)
    ]
    for name, b_v, m_v, higher_is_better in perf_items:
        delta = m_v - b_v
        if abs(delta) < 1e-4:
            status = "Neutral"
        elif (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better):
            status = "Improved"
        else:
            status = "Worsened"

        print(f"{'Performance':<15} | {name:<32} | {b_v:<10.4f} | {m_v:<10.4f} | {delta:<+10.4f} | {status}")
        comparison_records.append({
            "category": "Performance",
            "metric_name": name,
            "baseline_value": b_v,
            "mitigated_value": m_v,
            "absolute_change": delta,
            "impact_status": status
        })

    # Fairness rows
    b_disp = base_intersectional_audit["disparities"]
    m_disp = mit_intersectional_audit["disparities"]

    fair_items = [
        ("Demographic Parity Diff", b_disp["demographic_parity_difference"], m_disp["demographic_parity_difference"], False),
        ("Disparate Impact Ratio", b_disp["disparate_impact_ratio"], m_disp["disparate_impact_ratio"], True),
        ("Equalized Odds Diff", b_disp["equalized_odds_difference"], m_disp["equalized_odds_difference"], False),
        ("Equal Opportunity Diff (TPR)", b_disp["equal_opportunity_difference"], m_disp["equal_opportunity_difference"], False),
        ("FPR Difference", b_disp["false_positive_rate_difference"], m_disp["false_positive_rate_difference"], False)
    ]
    for name, b_v, m_v, higher_is_better in fair_items:
        delta = m_v - b_v
        if abs(delta) < 1e-4:
            status = "Neutral"
        elif (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better):
            status = "Improved"
        else:
            status = "Worsened"

        print(f"{'Fairness':<15} | {name:<32} | {b_v:<10.4f} | {m_v:<10.4f} | {delta:<+10.4f} | {status}")
        comparison_records.append({
            "category": "Fairness",
            "metric_name": name,
            "baseline_value": b_v,
            "mitigated_value": m_v,
            "absolute_change": delta,
            "impact_status": status
        })

    # Serialize JSON & CSV outputs
    os.makedirs(os.path.dirname(output_mitigated_json), exist_ok=True)
    mit_results_data = {
        "phase": 4,
        "mitigation_algorithm": "Fairlearn ExponentiatedGradient",
        "constraint_type": constraint_type,
        "eps_parameter": eps,
        "performance_metrics": mit_perf,
        "single_attribute_audits": mit_single_audits
    }
    with open(output_mitigated_json, "w") as f:
        json.dump(mit_results_data, f, indent=2)

    os.makedirs(os.path.dirname(output_intersectional_json), exist_ok=True)
    with open(output_intersectional_json, "w") as f:
        json.dump(mit_intersectional_audit, f, indent=2)

    os.makedirs(os.path.dirname(output_csv_comparison), exist_ok=True)
    pd.DataFrame(comparison_records).to_csv(output_csv_comparison, index=False)

    print("\nFile outputs successfully saved:")
    print(f"  - '{output_mitigated_json}'")
    print(f"  - '{output_intersectional_json}'")
    print(f"  - '{output_csv_comparison}'")
    print("=" * 85)

    return {
        "mitigated_results": mit_results_data,
        "mitigated_intersectional_audit": mit_intersectional_audit,
        "comparison_table": comparison_records
    }


if __name__ == "__main__":
    run_phase4_pipeline()
