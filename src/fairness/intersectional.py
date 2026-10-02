"""
Intersectional Fairness Auditing Module.

This module evaluates classification model predictions across multi-attribute
group combinations (e.g., Sex x Race) to calculate group-level performance metrics,
apply statistical safety thresholds to filter small-sample noise, and quantify
intersectional disparities.

Statistical Safety Policy:
--------------------------
Small group sample sizes (e.g., N < min_group_size) can produce volatile rate estimates.
This module partitions subgroups into:
1. Primary Groups (N >= min_group_size): Evaluated for primary disparity conclusions.
2. Low-Sample Groups (N < min_group_size): Reported in full transparency, but flagged as
   statistically unstable and excluded from primary disparity calculations.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from src.fairness.single_attribute import calculate_group_metrics


def create_intersectional_attribute(A_df, attributes=None, join_str=" + "):
    """
    Construct a combined pandas Series representing intersectional group identities.

    Parameters:
        A_df (pd.DataFrame): DataFrame containing protected attribute columns.
        attributes (list, optional): List of column names to combine. Defaults to all columns in A_df.
        join_str (str): Delimiter string used to join attribute values.

    Returns:
        pd.Series: Joined string Series representing intersectional group combinations.
    """
    if attributes is None:
        attributes = list(A_df.columns)

    missing_cols = [col for col in attributes if col not in A_df.columns]
    if missing_cols:
        raise ValueError(f"Missing attributes in DataFrame: {missing_cols}")

    # Combine string representations of selected attribute columns
    combined = A_df[attributes[0]].astype(str).str.strip()
    for col in attributes[1:]:
        combined = combined + join_str + A_df[col].astype(str).str.strip()

    combined.name = join_str.join(attributes)
    return combined


def audit_intersectional_attributes(
    y_true,
    y_pred,
    A_df,
    attributes=None,
    min_group_size=30,
    join_str=" + "
):
    """
    Perform an intersectional fairness audit across combinations of protected attributes.

    Parameters:
        y_true (pd.Series or np.ndarray): Ground truth binary labels.
        y_pred (pd.Series or np.ndarray): Predicted binary labels.
        A_df (pd.DataFrame): DataFrame containing protected attribute columns.
        attributes (list, optional): List of protected columns to combine (e.g., ['sex', 'race']).
        min_group_size (int): Minimum sample count threshold for primary disparity evaluation.
        join_str (str): Joining delimiter for group names.

    Returns:
        dict: Intersectional audit report dictionary containing group metrics,
              disparities, primary/low-sample group partitions, rankings, and definitions.
    """
    if attributes is None:
        attributes = list(A_df.columns)

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    intersectional_series = create_intersectional_attribute(A_df, attributes=attributes, join_str=join_str)
    intersectional_arr = np.asarray(intersectional_series)

    unique_groups = np.unique(intersectional_arr)
    all_group_metrics = {}

    for group in unique_groups:
        mask = (intersectional_arr == group)
        all_group_metrics[str(group)] = calculate_group_metrics(y_true_arr, y_pred_arr, mask)

    # Partition groups based on minimum sample size threshold
    primary_groups = {
        g: m for g, m in all_group_metrics.items() if m["sample_count"] >= min_group_size
    }
    low_sample_groups = {
        g: m for g, m in all_group_metrics.items() if m["sample_count"] < min_group_size
    }

    # Coverage status determination
    is_coverage_sufficient = len(primary_groups) >= 2
    if is_coverage_sufficient:
        coverage_code = "AVAILABLE"
        coverage_msg = f"Disparity metrics evaluated across {len(primary_groups)} primary groups (N >= {min_group_size})."
        reason = None
    else:
        coverage_code = "INSUFFICIENT_GROUP_COVERAGE"
        coverage_msg = (
            f"Only {len(primary_groups)} of {len(all_group_metrics)} intersectional groups satisfies N >= {min_group_size}. "
            "At least 2 eligible groups are required to calculate a meaningful disparity."
        )
        reason = f"Fewer than 2 primary groups met the minimum group size threshold (N >= {min_group_size})."

    coverage_status = {
        "code": coverage_code,
        "message": coverage_msg,
        "eligible_group_count": len(primary_groups),
        "total_group_count": len(all_group_metrics),
        "minimum_required_groups": 2,
        "min_group_size": min_group_size,
        "eligible_groups": list(primary_groups.keys()),
        "low_sample_groups": list(low_sample_groups.keys()),
        "reason": reason
    }

    if is_coverage_sufficient:
        selection_rates = {g: m["selection_rate"] for g, m in primary_groups.items()}
        tprs = {g: m["true_positive_rate"] for g, m in primary_groups.items()}
        fprs = {g: m["false_positive_rate"] for g, m in primary_groups.items()}

        max_sr_group = max(selection_rates, key=lambda g: selection_rates[g])
        min_sr_group = min(selection_rates, key=lambda g: selection_rates[g])

        max_tpr_group = max(tprs, key=lambda g: tprs[g])
        min_tpr_group = min(tprs, key=lambda g: tprs[g])

        max_fpr_group = max(fprs, key=lambda g: fprs[g])
        min_fpr_group = min(fprs, key=lambda g: fprs[g])

        max_sr = selection_rates[max_sr_group]
        min_sr = selection_rates[min_sr_group]

        max_tpr = tprs[max_tpr_group]
        min_tpr = tprs[min_tpr_group]

        max_fpr = fprs[max_fpr_group]
        min_fpr = fprs[min_fpr_group]

        dpd = float(max_sr - min_sr)
        dir_ratio = float(min_sr / max_sr) if max_sr > 0 else 1.0
        equal_opp_diff = float(max_tpr - min_tpr)
        fpr_diff = float(max_fpr - min_fpr)
        equalized_odds_diff = max(equal_opp_diff, fpr_diff)

        disparities: Dict[str, Optional[float]] = {
            "demographic_parity_difference": dpd,
            "disparate_impact_ratio": dir_ratio,
            "equalized_odds_difference": equalized_odds_diff,
            "equal_opportunity_difference": equal_opp_diff,
            "false_positive_rate_difference": fpr_diff
        }

        rankings = {
            "status": "AVAILABLE",
            "highest_selection_rate": {"group": max_sr_group, "value": max_sr},
            "lowest_selection_rate": {"group": min_sr_group, "value": min_sr},
            "highest_tpr": {"group": max_tpr_group, "value": max_tpr},
            "lowest_tpr": {"group": min_tpr_group, "value": min_tpr},
            "highest_fpr": {"group": max_fpr_group, "value": max_fpr},
            "lowest_fpr": {"group": min_fpr_group, "value": min_fpr},
            "most_disadvantaged_group": {"group": min_sr_group, "value": min_sr, "criterion": "lowest_selection_rate"},
            "highest_performing_group": {"group": max_sr_group, "value": max_sr, "criterion": "highest_selection_rate"}
        }
    else:
        disparities: Dict[str, Optional[float]] = {
            "demographic_parity_difference": None,
            "disparate_impact_ratio": None,
            "equalized_odds_difference": None,
            "equal_opportunity_difference": None,
            "false_positive_rate_difference": None
        }

        rankings = {
            "status": "UNAVAILABLE",
            "reason": coverage_msg,
            "highest_selection_rate": None,
            "lowest_selection_rate": None,
            "highest_tpr": None,
            "lowest_tpr": None,
            "highest_fpr": None,
            "lowest_fpr": None,
            "most_disadvantaged_group": None,
            "highest_performing_group": None
        }

    definitions = {
        "demographic_parity_difference": "max(Selection Rate) - min(Selection Rate) across primary intersectional groups.",
        "disparate_impact_ratio": "min(Selection Rate) / max(Selection Rate) across primary intersectional groups.",
        "equalized_odds_difference": "max(Equal Opportunity Difference, False Positive Rate Difference).",
        "equal_opportunity_difference": "max(True Positive Rate) - min(True Positive Rate) across primary intersectional groups.",
        "false_positive_rate_difference": "max(False Positive Rate) - min(False Positive Rate) across primary intersectional groups."
    }

    return {
        "intersection_definition": " x ".join(attributes),
        "attributes_combined": attributes,
        "min_group_size_threshold": min_group_size,
        "total_groups_discovered": len(unique_groups),
        "primary_group_count": len(primary_groups),
        "low_sample_group_count": len(low_sample_groups),
        "coverage_status": coverage_status,
        "all_group_metrics": all_group_metrics,
        "primary_groups": list(primary_groups.keys()),
        "low_sample_groups": list(low_sample_groups.keys()),
        "disparities": disparities,
        "rankings": rankings,
        "metric_definitions": definitions,
        "statistical_safety_policy": (
            f"Groups with sample size N < {min_group_size} are designated as low-sample groups. "
            "Their rate estimates are subject to statistical volatility and are excluded from primary "
            "disparity bounds to prevent small-sample noise from distorting conclusions."
        )
    }
