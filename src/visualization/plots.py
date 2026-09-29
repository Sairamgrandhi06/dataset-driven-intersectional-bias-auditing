"""
Visualization Module for Intersectional Bias Auditing and Mitigation.

This module generates publication-quality static charts comparing group-level
fairness metrics (Selection Rates, True Positive Rates, False Positive Rates)
and model performance before vs. after bias mitigation.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def setup_style():
    """Apply clean custom plot styling."""
    plt.style.use("ggplot")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 13,
        "figure.autolayout": True
    })


def plot_intersectional_metric_bars(
    audit_results,
    metric_key="selection_rate",
    metric_label="Selection Rate P(y_hat=1)",
    title="Intersectional Selection Rates (Sex x Race)",
    output_path="results/figures/intersectional_selection_rates.png"
):
    """
    Generate a formatted horizontal bar chart comparing an intersectional group metric.
    """
    setup_style()

    all_groups = audit_results.get("all_group_metrics", {})
    min_size = audit_results.get("min_group_size_threshold", 30)

    if not all_groups:
        print("Warning: No group metrics found to plot.")
        return None

    records = []
    for grp, m in all_groups.items():
        records.append({
            "group": grp,
            "val": m[metric_key],
            "count": m["sample_count"],
            "is_primary": m["sample_count"] >= min_size
        })

    df = pd.DataFrame(records)
    df.sort_values(by="val", ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2b5c8f" if is_p else "#a0aab2" for is_p in df["is_primary"]]
    bars = ax.barh(df["group"].tolist(), df["val"].tolist(), color=colors, edgecolor="black", alpha=0.85)

    for bar, val, count, is_p in zip(bars, df["val"], df["count"], df["is_primary"]):
        label_text = f"{val:.1%}  (N={count:,})"
        if not is_p:
            label_text += " [Low Sample]"
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            label_text,
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
            fontstyle="normal" if is_p else "italic"
        )

    ax.set_xlabel(metric_label, weight="bold")
    ax.set_title(title, weight="bold", pad=15)
    ax.set_xlim(0, max(df["val"].max() * 1.35, 0.1))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2b5c8f", edgecolor="black", label=f"Primary Group (N >= {min_size})"),
        Patch(facecolor="#a0aab2", edgecolor="black", label=f"Low-Sample Group (N < {min_size})")
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved: {output_path}")
    return output_path


def plot_all_intersectional_figures(audit_results, output_dir="results/figures"):
    """
    Generate and save all intersectional audit charts (Selection Rate, TPR, FPR).
    """
    os.makedirs(output_dir, exist_ok=True)

    p1 = plot_intersectional_metric_bars(
        audit_results,
        metric_key="selection_rate",
        metric_label="Positive Selection Rate P(y_hat=1)",
        title="Intersectional Selection Rate by Subgroup (Sex x Race)",
        output_path=os.path.join(output_dir, "intersectional_selection_rates.png")
    )

    p2 = plot_intersectional_metric_bars(
        audit_results,
        metric_key="true_positive_rate",
        metric_label="True Positive Rate (TPR / Recall)",
        title="Intersectional True Positive Rate by Subgroup (Sex x Race)",
        output_path=os.path.join(output_dir, "intersectional_tpr.png")
    )

    p3 = plot_intersectional_metric_bars(
        audit_results,
        metric_key="false_positive_rate",
        metric_label="False Positive Rate (FPR)",
        title="Intersectional False Positive Rate by Subgroup (Sex x Race)",
        output_path=os.path.join(output_dir, "intersectional_fpr.png")
    )

    return {"selection_rates": p1, "tpr": p2, "fpr": p3}


def plot_tradeoff_pareto_front(pareto_grid, output_path="results/figures/mitigation_pareto_front.png"):
    """
    Plot Pareto front curve comparing Model Accuracy vs. Equalized Odds Disparity.
    """
    setup_style()
    df = pd.DataFrame(pareto_grid)

    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["equalized_odds_difference"].tolist(), df["accuracy"].tolist(), marker="o", color="#1f77b4", linewidth=2.5, markersize=8)

    for _, row in df.iterrows():
        ax.annotate(
            f"eps={row['eps']}",
            (row["equalized_odds_difference"], row["accuracy"]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            fontsize=8,
            color="#333333"
        )

    ax.set_xlabel("Equalized Odds Disparity (Lower = Fairer)", weight="bold")
    ax.set_ylabel("Model Accuracy", weight="bold")
    ax.set_title("Accuracy vs. Fairness Pareto Trade-off Front", weight="bold", pad=15)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved: {output_path}")
    return output_path


def plot_before_after_comparison(baseline_audit, mitigated_audit, output_path="results/figures/before_after_intersectional_comparison.png"):
    """
    Generate side-by-side grouped bar chart comparing group selection rates before vs after mitigation.
    """
    setup_style()
    b_groups = baseline_audit.get("all_group_metrics", {})
    m_groups = mitigated_audit.get("all_group_metrics", {})

    groups = sorted(list(set(b_groups.keys()).union(set(m_groups.keys()))))
    b_rates = [b_groups[g]["selection_rate"] for g in groups]
    m_rates = [m_groups[g]["selection_rate"] for g in groups]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.barh(x - width/2, b_rates, width, label="Baseline (Before)", color="#d95f02", edgecolor="black", alpha=0.85)
    ax.barh(x + width/2, m_rates, width, label="Mitigated (After)", color="#7570b3", edgecolor="black", alpha=0.85)

    ax.set_ylabel("Intersectional Subgroup", weight="bold")
    ax.set_xlabel("Positive Selection Rate P(y_hat=1)", weight="bold")
    ax.set_title("Intersectional Selection Rates: Baseline vs. Mitigated Model", weight="bold", pad=15)
    ax.set_yticks(x)
    ax.set_yticklabels(groups)
    ax.legend(loc="lower right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved: {output_path}")
    return output_path


def plot_baseline_vs_mitigated_performance(baseline_metrics, mitigated_metrics, output_path="results/figures/baseline_vs_mitigated_performance.png"):
    """
    Plot baseline vs. mitigated predictive performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
    """
    setup_style()
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    b_vals = [
        baseline_metrics["accuracy"],
        baseline_metrics["precision"],
        baseline_metrics["recall"],
        baseline_metrics["f1_score"],
        baseline_metrics.get("roc_auc", 0.0) or 0.0
    ]
    m_vals = [
        mitigated_metrics["accuracy"],
        mitigated_metrics["precision"],
        mitigated_metrics["recall"],
        mitigated_metrics["f1_score"],
        mitigated_metrics.get("roc_auc", 0.0) or 0.0
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, b_vals, width, label="Baseline", color="#1f77b4", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x + width/2, m_vals, width, label="Mitigated", color="#2ca02c", edgecolor="black", alpha=0.85)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}", (rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Metric Score", weight="bold")
    ax.set_title("Overall ML Performance: Baseline vs. Mitigated Model", weight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved: {output_path}")
    return output_path


def plot_baseline_vs_mitigated_fairness(baseline_audit, mitigated_audit, output_path="results/figures/baseline_vs_mitigated_fairness.png"):
    """
    Plot baseline vs. mitigated fairness disparity metrics.
    """
    setup_style()
    b_disp = baseline_audit["disparities"]
    m_disp = mitigated_audit["disparities"]

    labels = ["Demographic Parity Diff", "Disparate Impact Ratio", "Equalized Odds Diff", "Equal Opp Diff (TPR)", "FPR Diff"]
    b_vals = [
        b_disp["demographic_parity_difference"],
        b_disp["disparate_impact_ratio"],
        b_disp["equalized_odds_difference"],
        b_disp["equal_opportunity_difference"],
        b_disp["false_positive_rate_difference"]
    ]
    m_vals = [
        m_disp["demographic_parity_difference"],
        m_disp["disparate_impact_ratio"],
        m_disp["equalized_odds_difference"],
        m_disp["equal_opportunity_difference"],
        m_disp["false_positive_rate_difference"]
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, b_vals, width, label="Baseline", color="#d95f02", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x + width/2, m_vals, width, label="Mitigated", color="#7570b3", edgecolor="black", alpha=0.85)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}", (rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Disparity Metric Value", weight="bold")
    ax.set_title("Fairness Disparities: Baseline vs. Mitigated Model", weight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, max(max(b_vals), max(m_vals)) * 1.25)
    ax.legend(loc="upper right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved: {output_path}")
    return output_path


def plot_baseline_vs_mitigated_group_metric(
    baseline_audit,
    mitigated_audit,
    metric_key="true_positive_rate",
    metric_label="True Positive Rate (TPR)",
    title="Intersectional TPR: Baseline vs. Mitigated Model",
    output_path="results/figures/baseline_vs_mitigated_tpr.png"
):
    """
    Plot baseline vs mitigated group metric for each intersectional group.
    """
    setup_style()
    b_groups = baseline_audit.get("all_group_metrics", {})
    m_groups = mitigated_audit.get("all_group_metrics", {})

    groups = sorted(list(set(b_groups.keys()).union(set(m_groups.keys()))))
    b_vals = [b_groups[g][metric_key] for g in groups]
    m_vals = [m_groups[g][metric_key] for g in groups]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.barh(x - width/2, b_vals, width, label="Baseline", color="#e7298a", edgecolor="black", alpha=0.85)
    ax.barh(x + width/2, m_vals, width, label="Mitigated", color="#66a61e", edgecolor="black", alpha=0.85)

    ax.set_ylabel("Intersectional Subgroup", weight="bold")
    ax.set_xlabel(metric_label, weight="bold")
    ax.set_title(title, weight="bold", pad=15)
    ax.set_yticks(x)
    ax.set_yticklabels(groups)
    ax.legend(loc="lower right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved: {output_path}")
    return output_path


def plot_calibration_curves(baseline_calib, mitigated_calib=None, output_path="results/figures/baseline_vs_mitigated_calibration.png"):
    """
    Plot reliability curves comparing Baseline vs. Mitigated model probability calibration.

    Parameters:
        baseline_calib (dict): Calibration dict from evaluate_calibration for baseline model.
        mitigated_calib (dict, optional): Calibration dict from evaluate_calibration for mitigated model.
        output_path (str): Output PNG file path.

    Returns:
        str: Output PNG path.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", linewidth=1.5)

    if baseline_calib and isinstance(baseline_calib, dict) and "reliability_curve" in baseline_calib:
        b_rc = baseline_calib["reliability_curve"]
        if b_rc and "prob_pred" in b_rc and "prob_true" in b_rc and len(b_rc["prob_pred"]) > 0:
            b_bs = baseline_calib.get("brier_score")
            b_label = f"Baseline (Brier={b_bs:.4f})" if b_bs is not None else "Baseline Model"
            ax.plot(b_rc["prob_pred"], b_rc["prob_true"], "s-", color="#d95f02", label=b_label, linewidth=2)

    if mitigated_calib and isinstance(mitigated_calib, dict) and "reliability_curve" in mitigated_calib:
        m_rc = mitigated_calib["reliability_curve"]
        if m_rc and "prob_pred" in m_rc and "prob_true" in m_rc and len(m_rc["prob_pred"]) > 0:
            m_bs = mitigated_calib.get("brier_score")
            m_label = f"Mitigated (Brier={m_bs:.4f})" if m_bs is not None else "Mitigated Model"
            ax.plot(m_rc["prob_pred"], m_rc["prob_true"], "o-", color="#7570b3", label=m_label, linewidth=2)

    ax.set_xlabel("Mean Predicted Probability", weight="bold")
    ax.set_ylabel("Fraction of Positives", weight="bold")
    ax.set_title("Probability Calibration (Reliability Curve)", weight="bold", pad=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved: {output_path}")
    return output_path


def plot_probability_distributions(baseline_y_prob, mitigated_y_prob, output_path="results/figures/probability_distributions.png"):
    """
    Plot predicted positive class probability distributions for baseline vs. mitigated models.

    Parameters:
        baseline_y_prob (np.ndarray): Baseline predicted probabilities.
        mitigated_y_prob (np.ndarray): Mitigated predicted probabilities.
        output_path (str): Output PNG path.

    Returns:
        str: Output PNG path.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(baseline_y_prob, bins=25, alpha=0.5, color="#1f77b4", label="Baseline Probabilities", edgecolor="black")
    ax.hist(mitigated_y_prob, bins=25, alpha=0.5, color="#2ca02c", label="Mitigated Probabilities", edgecolor="black")

    ax.set_xlabel("Predicted Probability P(y_hat=1)", weight="bold")
    ax.set_ylabel("Sample Frequency Count", weight="bold")
    ax.set_title("Prediction Probability Distributions: Baseline vs. Mitigated", weight="bold", pad=15)
    ax.legend(loc="upper right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved: {output_path}")
    return output_path


def plot_tradeoff_summary_chart(tradeoff_report, output_path="results/figures/tradeoff_summary_chart.png"):
    """
    Plot summary bar chart comparing percentage changes across Performance, Calibration, and Fairness.

    Parameters:
        tradeoff_report (dict): Output from calculate_comprehensive_tradeoff or dashboard trade-off dictionary.
        output_path (str): Output PNG path.

    Returns:
        str: Output PNG path.
    """
    setup_style()
    if not isinstance(tradeoff_report, dict):
        return None

    metrics = []
    pct_changes = []
    colors = []

    # Performance items
    perf = tradeoff_report.get("performance", {})
    if isinstance(perf, dict):
        for name, m in perf.items():
            if isinstance(m, dict):
                pct = m.get("percentage_change")
                if pct is not None:
                    try:
                        pct_val = float(pct)
                        metrics.append(f"Perf: {name}")
                        pct_changes.append(pct_val)
                        colors.append("#1f77b4" if m.get("impact_status") == "Improved" else ("#d62728" if m.get("impact_status") == "Worsened" else "#7f7f7f"))
                    except (ValueError, TypeError):
                        pass

    # Calibration items
    calib = tradeoff_report.get("calibration", {})
    if isinstance(calib, dict):
        for name, m in calib.items():
            if name == "interpretation":
                continue
            if isinstance(m, dict):
                pct = m.get("percentage_change")
                if pct is not None:
                    try:
                        pct_val = float(pct)
                        metrics.append(f"Calib: {name.replace('_', ' ').title()}")
                        pct_changes.append(pct_val)
                        colors.append("#2ca02c" if m.get("impact_status") == "Improved" else ("#d62728" if m.get("impact_status") == "Worsened" else "#7f7f7f"))
                    except (ValueError, TypeError):
                        pass

    # Fairness items
    fair = tradeoff_report.get("fairness", {})
    if isinstance(fair, dict):
        for name, m in fair.items():
            if name in ["intersectional_threshold_policy", "interpretation"]:
                continue
            if isinstance(m, dict):
                pct = m.get("percentage_change")
                if pct is not None:
                    try:
                        pct_val = float(pct)
                        metrics.append(f"Fair: {name}")
                        pct_changes.append(pct_val)
                        colors.append("#2ca02c" if m.get("impact_status") == "Improved" else ("#d62728" if m.get("impact_status") == "Worsened" else "#7f7f7f"))
                    except (ValueError, TypeError):
                        pass

    if not metrics:
        return None

    fig, ax = plt.subplots(figsize=(10, max(5, len(metrics) * 0.6)))
    bars = ax.barh(metrics, pct_changes, color=colors, edgecolor="black", alpha=0.85)

    for bar, val in zip(bars, pct_changes):
        ax.text(
            val + (1.5 if val >= 0 else -1.5),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}%",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
            weight="bold"
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Percentage Change (% Delta relative to Baseline)", weight="bold")
    ax.set_title("Three-Way Trade-off Metric Percentage Changes (% Delta)", weight="bold", pad=15)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", edgecolor="black", label="Improved"),
        Patch(facecolor="#d62728", edgecolor="black", label="Worsened / Degraded"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Performance Metric")
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved: {output_path}")
    return output_path



def plot_pre_post_group_comparison(
    base_audit,
    mit_audit,
    metric_key="true_positive_rate",
    metric_name="True Positive Rate (TPR)",
    output_path="results/figures/pre_post_group_tpr.png"
):
    """
    Plot grouped bar chart comparing group-level fairness metrics before vs. after mitigation.
    """
    setup_style()
    b_groups = base_audit.get("all_group_metrics", {})
    m_groups = mit_audit.get("all_group_metrics", {})

    common_groups = [g for g in b_groups.keys() if g in m_groups]
    if not common_groups:
        return None

    # Filter or sort by sample size / name
    common_groups.sort(key=lambda g: b_groups[g].get("sample_count", 0), reverse=True)
    plot_groups = common_groups[:10]  # top 10 groups

    base_vals = [b_groups[g].get(metric_key, 0.0) or 0.0 for g in plot_groups]
    mit_vals = [m_groups[g].get(metric_key, 0.0) or 0.0 for g in plot_groups]

    y = np.arange(len(plot_groups))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_groups) * 0.55)))
    rects1 = ax.barh(y - height/2, base_vals, height, label="Baseline (Normal Model)", color="#94A3B8", edgecolor="black", alpha=0.85)
    rects2 = ax.barh(y + height/2, mit_vals, height, label="Mitigated (Fair Model)", color="#2563EB", edgecolor="black", alpha=0.9)

    ax.set_xlabel(metric_name, weight="bold")
    ax.set_title(f"Group Opportunity Rebalancing: {metric_name} (Before vs. After Mitigation)", weight="bold", pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_groups)
    ax.set_xlim(0, max(max(base_vals + mit_vals) * 1.3, 0.1))
    ax.legend(loc="lower right", frameon=True)

    for r in rects1:
        w = r.get_width()
        ax.text(w + 0.01, r.get_y() + r.get_height() / 2, f"{w:.1%}", va="center", ha="left", fontsize=8, color="#475569")

    for r in rects2:
        w = r.get_width()
        ax.text(w + 0.01, r.get_y() + r.get_height() / 2, f"{w:.1%}", va="center", ha="left", fontsize=8, color="#1D4ED8", weight="bold")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path

