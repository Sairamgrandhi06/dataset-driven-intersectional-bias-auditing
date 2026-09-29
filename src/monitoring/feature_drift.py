"""
Feature-Level Drift Aggregator Module.

Loops across all feature columns in reference and monitoring DataFrames, applying numerical (PSI)
or categorical (TVD) drift detectors, and generates a consolidated feature drift report.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from src.monitoring.data_drift import detect_numerical_feature_drift, detect_categorical_feature_drift


def analyze_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: Optional[str] = None,
    protected_attributes: Optional[List[str]] = None,
    id_columns: Optional[List[str]] = None,
    ignore_columns: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze drift for all predictive feature columns between reference and current monitoring datasets.
    Excludes target_column, id_columns, and ignore_columns from predictive feature drift calculations.

    Parameters:
        reference_df (pd.DataFrame): Reference training dataset.
        current_df (pd.DataFrame): Current monitoring dataset.
        target_column (str, optional): Target column to exclude from feature drift.
        protected_attributes (list, optional): Protected attributes list.
        id_columns (list, optional): ID columns to exclude from feature drift.
        ignore_columns (list, optional): Ignored columns to exclude from feature drift.
        config (dict, optional): Monitoring configuration options.

    Returns:
        dict: Consolidated feature drift report.
    """
    config = config or {}
    dd_cfg = config.get("data_drift", {})

    psi_warn = float(dd_cfg.get("psi_warning_threshold", 0.10))
    psi_crit = float(dd_cfg.get("psi_critical_threshold", 0.25))
    tvd_warn = float(dd_cfg.get("tvd_warning_threshold", 0.10))
    tvd_crit = float(dd_cfg.get("tvd_critical_threshold", 0.25))

    exclude_cols = set()
    if target_column:
        exclude_cols.add(target_column)
    if protected_attributes:
        exclude_cols.update(protected_attributes)
    if id_columns:
        exclude_cols.update(id_columns)
    if ignore_columns:
        exclude_cols.update(ignore_columns)

    feature_cols = [c for c in reference_df.columns if c in current_df.columns and c not in exclude_cols]

    feature_drift_reports = {}
    drifted_features = []
    warning_features = []
    stable_features = []

    for col in feature_cols:
        is_ref_num = pd.api.types.is_numeric_dtype(reference_df[col])
        is_cur_num = pd.api.types.is_numeric_dtype(current_df[col])

        if is_ref_num and is_cur_num:
            res = detect_numerical_feature_drift(
                reference_df[col], current_df[col],
                psi_warning=psi_warn, psi_critical=psi_crit
            )
            res["feature_type"] = "numerical"
        elif (not is_ref_num) and (not is_cur_num):
            res = detect_categorical_feature_drift(
                reference_df[col], current_df[col],
                tvd_warning=tvd_warn, tvd_critical=tvd_crit
            )
            res["feature_type"] = "categorical"
        else:
            # Datatype mismatch on this specific column
            res = {
                "feature_name": col,
                "status": "TYPE_MISMATCH",
                "metric_name": "PSI/TVD",
                "drift_score": None,
                "p_value": None,
                "threshold_warning": psi_warn,
                "threshold_critical": psi_crit,
                "drift_detected": False,
                "warning_level": False,
                "feature_type": "incompatible",
                "details": f"Incompatible data types: reference is {reference_df[col].dtype}, current is {current_df[col].dtype}"
            }

        feature_drift_reports[col] = res

        if res["status"] == "DRIFT_DETECTED":
            drifted_features.append(col)
        elif res["status"] == "WARNING":
            warning_features.append(col)
        elif res["status"] in ["NO_DRIFT", "STABLE"]:
            stable_features.append(col)

    total_features = len(feature_cols)
    drift_ratio = len(drifted_features) / total_features if total_features > 0 else 0.0

    if len(drifted_features) > 0 or drift_ratio >= 0.20:
        overall_status = "DRIFT_DETECTED"
    elif len(warning_features) > 0:
        overall_status = "WARNING"
    else:
        overall_status = "NO_DRIFT"

    return {
        "status": overall_status,
        "total_features_analyzed": total_features,
        "drifted_features_count": len(drifted_features),
        "warning_features_count": len(warning_features),
        "stable_features_count": len(stable_features),
        "drifted_features": drifted_features,
        "warning_features": warning_features,
        "stable_features": stable_features,
        "feature_metrics": feature_drift_reports
    }
