"""
Target / Label Drift Monitoring Module.

Monitors target label distribution shifts (class balance, positive class rate changes) between reference
and monitoring datasets when ground-truth labels are present. Gracefully reports TARGET_LABELS_UNAVAILABLE when missing.
"""

from typing import Dict, Any, Optional
import pandas as pd


def analyze_target_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: str,
    positive_class: Any,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze target label distribution changes between reference and current monitoring datasets.

    Parameters:
        reference_df (pd.DataFrame): Reference baseline dataset.
        current_df (pd.DataFrame): Current monitoring dataset.
        target_column (str): Target column name.
        positive_class (Any): Value representing positive target class.
        config (dict, optional): Monitoring config options.

    Returns:
        dict: Target drift report.
    """
    if target_column not in current_df.columns or current_df[target_column].dropna().empty:
        return {
            "status": "TARGET_LABELS_UNAVAILABLE",
            "labels_available": False,
            "message": f"Target column '{target_column}' is not present or contains no valid labels in current monitoring data."
        }

    ref_target = reference_df[target_column].dropna()
    cur_target = current_df[target_column].dropna()

    ref_total = len(ref_target)
    cur_total = len(cur_target)

    if ref_total == 0 or cur_total == 0:
        return {
            "status": "TARGET_LABELS_UNAVAILABLE",
            "labels_available": False,
            "message": "Target series contains zero valid non-null rows."
        }

    # Calculate positive class rates (case-insensitive for string classes)
    if isinstance(positive_class, str):
        pos_str = positive_class.strip().lower()
        ref_pos_rate = float((ref_target.astype(str).str.strip().str.lower() == pos_str).mean())
        cur_pos_rate = float((cur_target.astype(str).str.strip().str.lower() == pos_str).mean())
    else:
        ref_pos_rate = float((ref_target == positive_class).mean())
        cur_pos_rate = float((cur_target == positive_class).mean())
    pos_rate_delta = cur_pos_rate - ref_pos_rate
    abs_pos_rate_delta = abs(pos_rate_delta)

    ref_dist = ref_target.value_counts(normalize=True).to_dict()
    cur_dist = cur_target.value_counts(normalize=True).to_dict()

    # Convert keys to strings for JSON safety
    ref_dist_str = {str(k): float(v) for k, v in ref_dist.items()}
    cur_dist_str = {str(k): float(v) for k, v in cur_dist.items()}

    if abs_pos_rate_delta >= 0.15:
        status = "DRIFT_DETECTED"
    elif abs_pos_rate_delta >= 0.05:
        status = "WARNING"
    else:
        status = "TARGET_STABLE"

    return {
        "status": status,
        "labels_available": True,
        "reference_positive_rate": ref_pos_rate,
        "current_positive_rate": cur_pos_rate,
        "positive_rate_delta": pos_rate_delta,
        "absolute_positive_rate_delta": abs_pos_rate_delta,
        "reference_distribution": ref_dist_str,
        "current_distribution": cur_dist_str
    }
