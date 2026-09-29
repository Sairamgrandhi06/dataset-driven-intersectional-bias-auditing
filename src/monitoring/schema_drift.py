"""
Schema Drift Detection Module.

Verifies feature, target, and protected attribute schema compatibility between reference
and monitoring datasets. Detects added/removed columns, data type shifts, and category shifts.
"""

from typing import Dict, Any, List, Optional
import pandas as pd


def detect_schema_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: Optional[str] = None,
    protected_attributes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Detect schema changes and incompatibilities between reference and current monitoring DataFrames.

    Parameters:
        reference_df (pd.DataFrame): Reference training/baseline dataset.
        current_df (pd.DataFrame): Current monitoring dataset.
        target_column (str, optional): Target outcome column name.
        protected_attributes (list, optional): List of protected attribute column names.

    Returns:
        dict: Structured schema drift report.
    """
    protected_attributes = protected_attributes or []

    ref_cols = set(reference_df.columns)
    cur_cols = set(current_df.columns)

    added_columns = sorted(list(cur_cols - ref_cols))
    removed_columns = sorted(list(ref_cols - cur_cols))

    common_cols = sorted(list(ref_cols.intersection(cur_cols)))

    dtype_changes = []
    for col in common_cols:
        ref_dt = str(reference_df[col].dtype)
        cur_dt = str(current_df[col].dtype)
        if ref_dt != cur_dt:
            # Differentiate minor integer/float and string/object conversions from object/numeric shifts
            is_ref_num = pd.api.types.is_numeric_dtype(reference_df[col])
            is_cur_num = pd.api.types.is_numeric_dtype(current_df[col])
            is_ref_str = pd.api.types.is_string_dtype(reference_df[col]) or pd.api.types.is_object_dtype(reference_df[col])
            is_cur_str = pd.api.types.is_string_dtype(current_df[col]) or pd.api.types.is_object_dtype(current_df[col])
            if not ((is_ref_num and is_cur_num) or (is_ref_str and is_cur_str)):
                dtype_changes.append({
                    "column": col,
                    "reference_dtype": ref_dt,
                    "current_dtype": cur_dt
                })

    new_categories = {}
    missing_expected_categories = {}

    for col in common_cols:
        if pd.api.types.is_object_dtype(reference_df[col]) or pd.api.types.is_string_dtype(reference_df[col]) or isinstance(reference_df[col].dtype, pd.CategoricalDtype):
            ref_cats = set(reference_df[col].dropna().unique())
            cur_cats = set(current_df[col].dropna().unique())

            new_cats = sorted([str(c) for c in (cur_cats - ref_cats)])
            miss_cats = sorted([str(c) for c in (ref_cats - cur_cats)])

            if new_cats:
                new_categories[col] = new_cats
            if miss_cats:
                missing_expected_categories[col] = miss_cats

    target_present = target_column in current_df.columns if target_column else True
    protected_presence = {attr: (attr in current_df.columns) for attr in protected_attributes}

    # Critical schema drift triggers block
    critical_issues = []

    # Check if features present in reference (excluding target) are missing in current
    expected_features = [c for c in reference_df.columns if c != target_column]
    missing_features = [c for c in expected_features if c not in current_df.columns]

    if missing_features:
        critical_issues.append(f"Missing required feature columns: {missing_features}")

    if dtype_changes:
        critical_issues.append(f"Incompatible data type changes in columns: {[d['column'] for d in dtype_changes]}")

    has_critical_drift = len(critical_issues) > 0
    status = "SCHEMA_DRIFT_DETECTED" if has_critical_drift else "SCHEMA_OK"

    return {
        "status": status,
        "has_critical_drift": has_critical_drift,
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "missing_features": missing_features,
        "dtype_changes": dtype_changes,
        "new_categories": new_categories,
        "missing_expected_categories": missing_expected_categories,
        "target_column_present": target_present,
        "protected_attributes_presence": protected_presence,
        "critical_issues": critical_issues
    }


def compare_reference_and_monitoring_schemas(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: Optional[str] = None,
    protected_attributes: Optional[List[str]] = None,
    id_columns: Optional[List[str]] = None,
    ignore_columns: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Produce a detailed column-by-column pre-run schema comparison table between reference and monitoring datasets.

    Returns:
        list of dicts with column, role, reference_dtype, monitoring_dtype, status, details, is_compatible.
    """
    protected_attributes = protected_attributes or []
    id_columns = id_columns or []
    ignore_columns = ignore_columns or []

    ref_cols = list(reference_df.columns)
    cur_cols = list(current_df.columns)

    all_cols = []
    for c in ref_cols:
        if c not in all_cols:
            all_cols.append(c)
    for c in cur_cols:
        if c not in all_cols:
            all_cols.append(c)

    rows = []
    for col in all_cols:
        in_ref = col in reference_df.columns
        in_cur = col in current_df.columns

        # Determine Role
        if col == target_column:
            role = "Target"
        elif col in protected_attributes:
            role = "Protected Attribute"
        elif col in id_columns:
            role = "ID Column"
        elif col in ignore_columns:
            role = "Ignored Column"
        else:
            role = "Predictive Feature" if in_ref else "Unexpected Extra Column"

        if in_ref and in_cur:
            ref_dt = str(reference_df[col].dtype)
            cur_dt = str(current_df[col].dtype)

            is_ref_num = pd.api.types.is_numeric_dtype(reference_df[col])
            is_cur_num = pd.api.types.is_numeric_dtype(current_df[col])
            is_ref_str = pd.api.types.is_string_dtype(reference_df[col]) or pd.api.types.is_object_dtype(reference_df[col])
            is_cur_str = pd.api.types.is_string_dtype(current_df[col]) or pd.api.types.is_object_dtype(current_df[col])

            if ref_dt == cur_dt:
                status = "✅ MATCH"
                details = f"Exact type match ({ref_dt})"
                is_compat = True
            elif is_ref_num and is_cur_num:
                status = "✅ MATCH"
                details = f"Compatible numeric types ({ref_dt} -> {cur_dt})"
                is_compat = True
            elif is_ref_str and is_cur_str:
                status = "✅ MATCH"
                details = f"Compatible string/categorical types ({ref_dt} -> {cur_dt})"
                is_compat = True
            else:
                status = "❌ TYPE MISMATCH"
                details = f"Incompatible: Reference is {ref_dt}, Monitoring is {cur_dt}"
                is_compat = False

            rows.append({
                "column": col,
                "role": role,
                "reference_dtype": ref_dt,
                "monitoring_dtype": cur_dt,
                "status": status,
                "details": details,
                "is_compatible": is_compat
            })

        elif in_ref and not in_cur:
            ref_dt = str(reference_df[col].dtype)
            if col == target_column:
                status = "ℹ️ UNLABELED BATCH"
                details = f"Target '{col}' not present in batch (performance/fairness evaluation will be skipped)"
                is_compat = True
            else:
                status = "❌ MISSING COLUMN"
                details = f"Required {role.lower()} '{col}' is missing in monitoring batch"
                is_compat = False

            rows.append({
                "column": col,
                "role": role,
                "reference_dtype": ref_dt,
                "monitoring_dtype": "MISSING",
                "status": status,
                "details": details,
                "is_compatible": is_compat
            })

        elif not in_ref and in_cur:
            cur_dt = str(current_df[col].dtype)
            status = "⚠️ EXTRA COLUMN"
            details = f"Column not present in reference model schema (will be ignored during prediction)"
            is_compat = True

            rows.append({
                "column": col,
                "role": "Extra Column",
                "reference_dtype": "NOT IN REFERENCE",
                "monitoring_dtype": cur_dt,
                "status": status,
                "details": details,
                "is_compatible": is_compat
            })

    return rows

