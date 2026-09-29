"""
Formal Input Contract for Fairness Criteria Compatibility Analysis.

Validates input arrays, protected attributes, requested criteria names, configured thresholds,
minimum subgroup size, and dataset identity metadata prior to compatibility evaluation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.contracts.error_contract import create_structured_error

SUPPORTED_CRITERIA = {
    "demographic_parity",
    "disparate_impact",
    "equal_opportunity",
    "equalized_odds",
    "fpr_difference"
}


def validate_criteria_input_contract(
    y_true: Any,
    y_pred: Any,
    protected_attributes: Any,
    selected_criteria: List[str],
    configured_thresholds: Dict[str, float],
    minimum_group_size: int = 30,
    positive_label: int = 1,
    run_id: str = "RUN-001",
    dataset_hash: str = "9007ff524acf0776b9598a0ed883058df91f59c003e78cb2cddd88f87535fc86",
    model_id: str = "BASELINE_LOGISTIC_REGRESSION",
    configuration_version: str = "1.0.0"
) -> Dict[str, Any]:
    """
    Validate input contract parameters and return structured validation dictionary.

    Returns:
        dict: Validation output containing 'is_valid' (bool), 'errors' (list), 'warnings' (list).
    """
    errors = []
    warnings = []

    # 1. Missing Inputs Check
    if y_true is None:
        errors.append(create_structured_error("MISSING_INPUT", "y_true ground truth array is missing", "y_true", recovery_action="Provide valid y_true binary array."))
    if y_pred is None:
        errors.append(create_structured_error("MISSING_INPUT", "y_pred prediction array is missing", "y_pred", recovery_action="Provide valid y_pred binary array."))
    if protected_attributes is None:
        errors.append(create_structured_error("MISSING_PROTECTED_ATTRIBUTE", "protected_attributes DataFrame/Series is missing", "protected_attributes", recovery_action="Provide DataFrame containing protected attribute columns."))

    if errors:
        return {"is_valid": False, "errors": errors, "warnings": warnings}

    # Convert inputs to numpy arrays / pandas objects for length checking
    yt_arr = np.asarray(y_true)
    yp_arr = np.asarray(y_pred)

    # 2. Length Mismatch Check
    if len(yt_arr) != len(yp_arr):
        errors.append(create_structured_error("LENGTH_MISMATCH", f"Length mismatch between y_true ({len(yt_arr)}) and y_pred ({len(yp_arr)})", "y_true/y_pred", recovery_action="Ensure y_true and y_pred have identical sample length."))

    if isinstance(protected_attributes, (pd.DataFrame, pd.Series)):
        pa_len = len(protected_attributes)
    else:
        pa_len = len(np.asarray(protected_attributes))

    if len(yt_arr) != pa_len:
        errors.append(create_structured_error("LENGTH_MISMATCH", f"Length mismatch between y_true ({len(yt_arr)}) and protected_attributes ({pa_len})", "protected_attributes", recovery_action="Ensure protected_attributes length matches y_true."))

    # 3. Binary Target Check
    unique_yt = set(np.unique(yt_arr))
    if not (unique_yt.issubset({0, 1}) or len(unique_yt) <= 2):
        errors.append(create_structured_error("INVALID_TARGET", f"Ground truth y_true is non-binary: observed classes {unique_yt}", "y_true", recovery_action="Binary classification requires y_true to contain binary labels."))

    # 4. Selected Criteria Validation
    if not selected_criteria or not isinstance(selected_criteria, list):
        errors.append(create_structured_error("INVALID_CONFIGURATION", "selected_criteria must be a non-empty list of criteria names", "selected_criteria", recovery_action="Provide list of criteria names."))
    else:
        # Check for duplicates
        if len(selected_criteria) != len(set(selected_criteria)):
            errors.append(create_structured_error("INVALID_CRITERION", "selected_criteria contains duplicate criteria names", "selected_criteria", recovery_action="Remove duplicate entries from selected_criteria."))

        # Check supported names
        for crit in selected_criteria:
            if crit not in SUPPORTED_CRITERIA:
                errors.append(create_structured_error("INVALID_CRITERION", f"Unsupported criterion '{crit}'", "selected_criteria", recovery_action=f"Select from supported criteria: {sorted(list(SUPPORTED_CRITERIA))}"))

    # 5. Threshold Validation
    if not isinstance(configured_thresholds, dict):
        errors.append(create_structured_error("INVALID_THRESHOLD", "configured_thresholds must be a dictionary", "configured_thresholds", recovery_action="Provide dictionary of threshold values."))
    else:
        for t_name, t_val in configured_thresholds.items():
            if not isinstance(t_val, (int, float)) or t_val < 0.0 or np.isnan(t_val):
                errors.append(create_structured_error("INVALID_THRESHOLD", f"Invalid threshold value '{t_val}' for threshold '{t_name}'", "configured_thresholds", recovery_action="Thresholds must be positive numeric floats."))

    # 6. Minimum Group Size Validation
    if not isinstance(minimum_group_size, int) or minimum_group_size < 1:
        errors.append(create_structured_error("INSUFFICIENT_GROUP_SIZE", f"minimum_group_size ({minimum_group_size}) must be a positive integer >= 1", "minimum_group_size", recovery_action="Set minimum_group_size >= 1 (default 30)."))

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "validated_parameters": {
            "run_id": run_id,
            "dataset_hash": dataset_hash,
            "model_id": model_id,
            "configuration_version": configuration_version,
            "sample_count": len(yt_arr) if len(yt_arr) == len(yp_arr) else 0,
            "minimum_group_size": minimum_group_size
        }
    }
