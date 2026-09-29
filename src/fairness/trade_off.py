"""
Comprehensive Trade-off Analysis Module.

This module evaluates the three-way trade-off between:
1. Predictive Machine Learning Performance (Accuracy, Precision, Recall, F1, ROC-AUC)
2. Model Probability Calibration (Brier Score, Expected Calibration Error)
3. Intersectional Fairness Disparities (Equalized Odds, Equal Opportunity, Demographic Parity)

Statistical & Ethical Reporting Rules:
--------------------------------------
1. Objective Trade-off Assessment: Reports metric improvements and regressions without
   claiming the mitigated model is unconditionally 'better'.
2. Non-Causal Framing: Disparity reductions are described as observed metric changes,
   not proof of eliminating underlying societal bias or intentional discrimination.
3. No Cherry-Picking: All predictive, calibration, and fairness metrics are reported
   side-by-side with exact baseline, mitigated, absolute change, and percentage change.
"""

import numpy as np
import pandas as pd
from src.fairness.calibration import compare_calibration


def compute_metric_change(baseline_val, mitigated_val, higher_is_better=True):
    """
    Calculate absolute change, percentage change, and directional impact status.

    Parameters:
        baseline_val (float or None): Baseline metric value.
        mitigated_val (float or None): Mitigated metric value.
        higher_is_better (bool): True if higher values indicate improvement (e.g., Accuracy),
                                False if lower values indicate improvement (e.g., Brier score, Disparity).

    Returns:
        dict: Dict containing baseline, mitigated, absolute_change, percentage_change, impact_status.
    """
    if baseline_val is None or mitigated_val is None:
        return {
            "baseline": baseline_val,
            "mitigated": mitigated_val,
            "absolute_change": None,
            "percentage_change": None,
            "impact_status": "Not estimable"
        }

    b_v = float(baseline_val)
    m_v = float(mitigated_val)

    delta = m_v - b_v
    pct = (delta / abs(b_v) * 100.0) if abs(b_v) > 1e-6 else 0.0

    if abs(delta) < 1e-5:
        status = "Neutral"
    elif (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better):
        status = "Improved"
    else:
        status = "Worsened"

    return {
        "baseline": b_v,
        "mitigated": m_v,
        "absolute_change": delta,
        "percentage_change": pct,
        "impact_status": status
    }


def calculate_comprehensive_tradeoff(
    baseline_metrics,
    mitigated_metrics,
    baseline_intersectional_audit,
    mitigated_intersectional_audit,
    baseline_calib,
    mitigated_calib
):
    """
    Calculate three-way trade-off across Performance, Calibration, and Fairness.

    Parameters:
        baseline_metrics (dict): Baseline ML performance metrics.
        mitigated_metrics (dict): Mitigated ML performance metrics.
        baseline_intersectional_audit (dict): Baseline intersectional audit report.
        mitigated_intersectional_audit (dict): Mitigated intersectional audit report.
        baseline_calib (dict): Baseline calibration report.
        mitigated_calib (dict): Mitigated calibration report.

    Returns:
        dict: Comprehensive trade-off report dictionary.
    """
    # 1. Performance Metrics
    perf_comp = {
        "Accuracy": compute_metric_change(baseline_metrics["accuracy"], mitigated_metrics["accuracy"], higher_is_better=True),
        "Precision": compute_metric_change(baseline_metrics["precision"], mitigated_metrics["precision"], higher_is_better=True),
        "Recall": compute_metric_change(baseline_metrics["recall"], mitigated_metrics["recall"], higher_is_better=True),
        "F1-Score": compute_metric_change(baseline_metrics["f1_score"], mitigated_metrics["f1_score"], higher_is_better=True),
        "ROC-AUC": compute_metric_change(
            baseline_metrics.get("roc_auc", 0.0) or 0.0,
            mitigated_metrics.get("roc_auc", 0.0) or 0.0,
            higher_is_better=True
        )
    }

    # 2. Calibration Metrics
    calib_comp = compare_calibration(baseline_calib, mitigated_calib)

    # 3. Fairness Metrics (Primary Intersectional Groups N >= 30)
    b_disp = baseline_intersectional_audit["disparities"]
    m_disp = mitigated_intersectional_audit["disparities"]

    fair_comp = {
        "Demographic Parity Difference": compute_metric_change(
            b_disp["demographic_parity_difference"], m_disp["demographic_parity_difference"], higher_is_better=False
        ),
        "Disparate Impact Ratio": compute_metric_change(
            b_disp["disparate_impact_ratio"], m_disp["disparate_impact_ratio"], higher_is_better=True
        ),
        "Equalized Odds Difference": compute_metric_change(
            b_disp["equalized_odds_difference"], m_disp["equalized_odds_difference"], higher_is_better=False
        ),
        "Equal Opportunity Difference (TPR)": compute_metric_change(
            b_disp["equal_opportunity_difference"], m_disp["equal_opportunity_difference"], higher_is_better=False
        ),
        "False Positive Rate Difference": compute_metric_change(
            b_disp["false_positive_rate_difference"], m_disp["false_positive_rate_difference"], higher_is_better=False
        )
    }

    tradeoff_report = {
        "performance": perf_comp,
        "calibration": calib_comp,
        "fairness": fair_comp,
        "intersectional_threshold_policy": "Minimum group size threshold N >= 30 enforced for primary disparity bounds."
    }

    tradeoff_report["interpretation"] = generate_tradeoff_interpretation(tradeoff_report)
    return tradeoff_report


def generate_tradeoff_summary_df(tradeoff_report):
    """
    Generate a clean pandas DataFrame summarizing Baseline vs. Mitigated metrics for CSV export.

    Parameters:
        tradeoff_report (dict): Output of calculate_comprehensive_tradeoff.

    Returns:
        pd.DataFrame: Tabular comparison DataFrame.
    """
    records = []

    # Add Performance
    for name, m in tradeoff_report["performance"].items():
        records.append({
            "category": "Performance",
            "metric_name": name,
            "baseline_value": m["baseline"],
            "mitigated_value": m["mitigated"],
            "absolute_change": m["absolute_change"],
            "percentage_change": m["percentage_change"],
            "impact_status": m["impact_status"]
        })

    # Add Calibration
    calib = tradeoff_report["calibration"]
    records.append({
        "category": "Calibration",
        "metric_name": "Brier Score",
        "baseline_value": calib["brier_score"]["baseline"],
        "mitigated_value": calib["brier_score"]["mitigated"],
        "absolute_change": calib["brier_score"]["absolute_change"],
        "percentage_change": calib["brier_score"]["percentage_change"],
        "impact_status": calib["brier_score"]["impact_status"]
    })
    records.append({
        "category": "Calibration",
        "metric_name": "Expected Calibration Error (ECE)",
        "baseline_value": calib["expected_calibration_error"]["baseline"],
        "mitigated_value": calib["expected_calibration_error"]["mitigated"],
        "absolute_change": calib["expected_calibration_error"]["absolute_change"],
        "percentage_change": calib["expected_calibration_error"]["percentage_change"],
        "impact_status": calib["expected_calibration_error"]["impact_status"]
    })

    # Add Fairness
    for name, m in tradeoff_report["fairness"].items():
        records.append({
            "category": "Fairness",
            "metric_name": name,
            "baseline_value": m["baseline"],
            "mitigated_value": m["mitigated"],
            "absolute_change": m["absolute_change"],
            "percentage_change": m["percentage_change"],
            "impact_status": m["impact_status"]
        })

    return pd.DataFrame(records)


def generate_tradeoff_interpretation(tradeoff_report):
    """
    Generate objective text interpretation supported strictly by empirical metric changes.

    Parameters:
        tradeoff_report (dict): Trade-off metrics report dictionary.

    Returns:
        str: Concise interpretation narrative.
    """
    perf = tradeoff_report["performance"]
    calib = tradeoff_report["calibration"]
    fair = tradeoff_report["fairness"]

    acc_info = perf["Accuracy"]
    acc_delta = acc_info["absolute_change"]
    acc_pct = acc_info["percentage_change"]

    acc_delta_str = f"{acc_delta:+.4f}" if acc_delta is not None else "N/A"
    acc_pct_str = f"{acc_pct:+.2f}%" if acc_pct is not None else "N/A"

    eod_info = fair.get("EqualizedOddsDifference") or fair.get("Equalized Odds Difference", {})
    eod_b = eod_info.get("baseline")
    eod_m = eod_info.get("mitigated")
    eod_delta = eod_info.get("absolute_change")

    if eod_b is None or eod_m is None or eod_delta is None:
        fair_narrative = "Intersectional fairness disparity metrics were not estimable due to insufficient group coverage (N < min_group_size)."
    else:
        fair_narrative = (
            f"Intersectional fairness disparity (Equalized Odds Difference) changed from {eod_b:.4f} to {eod_m:.4f} "
            f"(absolute change of {eod_delta:+.4f})."
        )

    bs_b = calib["brier_score"]["baseline"]
    bs_m = calib["brier_score"]["mitigated"]
    bs_status = calib["brier_score"]["impact_status"]

    bs_b_str = f"{bs_b:.4f}" if bs_b is not None else "N/A"
    bs_m_str = f"{bs_m:.4f}" if bs_m is not None else "N/A"

    interpretation = (
        f"{fair_narrative} Simultaneously, model predictive accuracy changed from "
        f"{perf['Accuracy']['baseline']:.4f} to {perf['Accuracy']['mitigated']:.4f} ({acc_delta_str}, {acc_pct_str}), "
        f"and probability calibration Brier Score changed from {bs_b_str} to {bs_m_str} ({bs_status.lower()}). "
        f"Therefore, the bias mitigation algorithm produced quantifiable changes in performance, calibration, and fairness."
    )

    return interpretation


# Function aliases for API compatibility
compute_three_way_tradeoff = calculate_comprehensive_tradeoff
generate_tradeoff_summary_table = generate_tradeoff_summary_df

