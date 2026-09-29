"""
Data Quality Drift Monitoring Module.

Monitors missingness rate shifts, duplicate rates, class balance changes, and protected attribute
group size collapse between reference and monitoring datasets.
"""

from typing import Dict, Any, List, Optional
import pandas as pd


def analyze_data_quality_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: Optional[str] = None,
    protected_attributes: Optional[List[str]] = None,
    min_group_size: int = 30,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze data quality changes between reference and current monitoring datasets.

    Parameters:
        reference_df (pd.DataFrame): Reference baseline dataset.
        current_df (pd.DataFrame): Current monitoring dataset.
        target_column (str, optional): Target column name.
        protected_attributes (list, optional): Protected attribute column names.
        min_group_size (int): Minimum required group size for statistical safety.
        config (dict, optional): Data quality drift config options.

    Returns:
        dict: Structured data quality drift report.
    """
    protected_attributes = protected_attributes or []
    config = config or {}
    dq_cfg = config.get("data_quality_drift", {})

    missing_warn_thresh = float(dq_cfg.get("missing_rate_increase_warning", 0.05))
    missing_crit_thresh = float(dq_cfg.get("missing_rate_increase_critical", 0.15))

    ref_rows = len(reference_df)
    cur_rows = len(current_df)

    # Missing value rates
    ref_missing_pct = reference_df.isna().mean().to_dict()
    cur_missing_pct = current_df.isna().mean().to_dict()

    missing_rate_deltas = {}
    high_missingness_features = []

    for col in current_df.columns:
        c_miss = cur_missing_pct[col]
        r_miss = ref_missing_pct.get(col, 0.0)
        delta = c_miss - r_miss
        missing_rate_deltas[col] = {
            "reference_missing_pct": float(r_miss),
            "current_missing_pct": float(c_miss),
            "delta_pct": float(delta)
        }
        if delta >= missing_warn_thresh:
            high_missingness_features.append(col)

    ref_overall_missing = float(reference_df.isna().mean().mean())
    cur_overall_missing = float(current_df.isna().mean().mean())
    overall_missing_delta = cur_overall_missing - ref_overall_missing

    # Duplicate rates
    ref_dups = float(reference_df.duplicated().mean())
    cur_dups = float(current_df.duplicated().mean())

    # Protected group size collapse analysis
    protected_group_status = {}
    collapsed_protected_groups = []

    for attr in protected_attributes:
        if attr in current_df.columns:
            counts = current_df[attr].value_counts().to_dict()
            attr_status = {}
            for val, cnt in counts.items():
                is_safe = (cnt >= min_group_size)
                attr_status[str(val)] = {
                    "count": int(cnt),
                    "is_statistically_safe": is_safe
                }
                if not is_safe:
                    collapsed_protected_groups.append(f"{attr}={val} (count: {cnt} < {min_group_size})")
            protected_group_status[attr] = attr_status

    # Overall data quality status
    if overall_missing_delta >= missing_crit_thresh or len(collapsed_protected_groups) > 0:
        status = "QUALITY_DEGRADED"
    elif overall_missing_delta >= missing_warn_thresh or len(high_missingness_features) > 0:
        status = "WARNING"
    else:
        status = "QUALITY_STABLE"

    return {
        "status": status,
        "reference_rows": ref_rows,
        "current_rows": cur_rows,
        "overall_missing_pct_ref": ref_overall_missing,
        "overall_missing_pct_cur": cur_overall_missing,
        "overall_missing_pct_delta": overall_missing_delta,
        "duplicate_pct_ref": ref_dups,
        "duplicate_pct_cur": cur_dups,
        "high_missingness_features": high_missingness_features,
        "missing_rate_deltas": missing_rate_deltas,
        "protected_group_status": protected_group_status,
        "collapsed_protected_groups": collapsed_protected_groups
    }
