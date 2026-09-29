"""
Data Quality Gates & Validation Module.

Provides independent, reusable quality checks for validating dataset existence, readability,
schema integrity, target binary compatibility, protected attribute distributions,
intersectional group sizes, missing values, duplicates, and SHA-256 file hashes.
"""

import hashlib
import json
import os
from typing import Any
import numpy as np
import pandas as pd


def compute_file_sha256(filepath):
    """
    Calculate the SHA-256 hash of a file on disk.

    Parameters:
        filepath (str): Absolute or relative path to file.

    Returns:
        str: Hexadecimal SHA-256 hash string, or 'UNKNOWN' if file cannot be read.
    """
    if not os.path.exists(filepath):
        return "UNKNOWN"

    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return "UNKNOWN"


def load_quality_config(config_path="config/data_quality_config.json"):
    """
    Load data quality configuration settings.

    Parameters:
        config_path (str): Path to data quality JSON configuration file.

    Returns:
        dict: Quality configuration parameters.
    """
    default_config = {
        "dataset_path": "data/raw/adult.csv",
        "minimum_rows": 30000,
        "maximum_missing_percent": 10.0,
        "allow_duplicates": True,
        "require_binary_target": True,
        "target_column": "income",
        "expected_target_classes": ["<=50K", ">50K"],
        "required_protected_attributes": ["sex", "race"],
        "minimum_intersectional_group_size": 30
    }

    if not os.path.exists(config_path):
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
            default_config.update(user_config)
            return default_config
    except Exception:
        return default_config


def run_data_quality_gates(
    df: pd.DataFrame | None = None,
    config: dict | None = None,
    dataset_path: str | None = None
) -> dict[str, Any]:
    """
    Execute 15 comprehensive data quality checks on the dataset.

    Parameters:
        df (pd.DataFrame or None): Raw dataset DataFrame. If None, loaded from dataset_path.
        config (dict or None): Quality configuration settings.
        dataset_path (str or None): Path to raw CSV dataset file.

    Returns:
        dict: Structured quality report containing overall status, check results, and metrics.
    """
    if config is None:
        config = load_quality_config()

    path = dataset_path or config.get("dataset_path", "data/raw/adult.csv")

    checks = []
    warnings = []
    failures = []

    # 1. Dataset Exists Check
    file_exists = os.path.exists(path)
    checks.append({
        "name": "dataset_exists",
        "status": "PASS" if file_exists else "FAIL",
        "observed": f"File exists at '{path}'" if file_exists else f"File missing at '{path}'",
        "expected": f"Dataset file must exist at '{path}'"
    })
    if not file_exists:
        failures.append(f"Dataset file missing at '{path}'")

    # 2. File Hash Calculation
    file_hash = compute_file_sha256(path)
    file_size = os.path.getsize(path) if file_exists else 0
    checks.append({
        "name": "dataset_hash_calculated",
        "status": "PASS" if file_hash != "UNKNOWN" else "FAIL",
        "observed": f"SHA256: {file_hash[:16]}... (Size: {file_size:,} bytes)",
        "expected": "SHA-256 file hash and size must be calculated"
    })

    # 3. Dataset Readability Check
    if df is None and file_exists:
        try:
            # Strip leading/trailing whitespace from string fields upon loading
            df = pd.read_csv(path, skipinitialspace=True)
        except Exception as e:
            checks.append({
                "name": "dataset_readable",
                "status": "FAIL",
                "observed": f"Read error: {str(e)}",
                "expected": "Dataset must be readable into pandas DataFrame"
            })
            failures.append(f"Read error: {str(e)}")
            return {
                "overall_status": "FAIL",
                "dataset_hash": file_hash,
                "checks": checks,
                "warnings": warnings,
                "failures": failures
            }

    readable = df is not isinstance(df, type(None)) and isinstance(df, pd.DataFrame)
    checks.append({
        "name": "dataset_readable",
        "status": "PASS" if readable else "FAIL",
        "observed": f"DataFrame loaded ({len(df):,} rows, {len(df.columns)} columns)" if readable else "DataFrame is None",
        "expected": "Dataset must be readable into pandas DataFrame"
    })

    if not readable:
        return {
            "overall_status": "FAIL",
            "dataset_hash": file_hash,
            "checks": checks,
            "warnings": warnings,
            "failures": failures
        }

    # Clean whitespace strings for consistent checking
    df_clean = df.copy()
    for col in df_clean.columns:
        if pd.api.types.is_string_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].astype(str).str.strip()

    # 4. Target Column Exists Check
    target_col = config.get("target_column", "income")
    target_exists = target_col in df_clean.columns
    checks.append({
        "name": "target_exists",
        "status": "PASS" if target_exists else "FAIL",
        "observed": f"Target column '{target_col}' present" if target_exists else f"Target column '{target_col}' missing",
        "expected": f"Target column '{target_col}' required"
    })
    if not target_exists:
        failures.append(f"Target column '{target_col}' missing")

    # 5. Protected Attributes Exist Check
    req_protected = config.get("required_protected_attributes", ["sex", "race"])
    missing_protected = [attr for attr in req_protected if attr not in df_clean.columns]
    protected_pass = len(missing_protected) == 0
    checks.append({
        "name": "protected_attributes_exist",
        "status": "PASS" if protected_pass else "FAIL",
        "observed": f"Protected attributes present: {req_protected}" if protected_pass else f"Missing protected attributes: {missing_protected}",
        "expected": f"Protected attributes required: {req_protected}"
    })
    if not protected_pass:
        failures.append(f"Missing protected attributes: {missing_protected}")

    # 6. Binary Target Compatibility Check
    if target_exists:
        unique_targets = sorted(df_clean[target_col].dropna().unique().tolist())
        expected_targets = config.get("expected_target_classes", ["<=50K", ">50K"])
        is_binary = len(unique_targets) == 2
        checks.append({
            "name": "target_binary_compatible",
            "status": "PASS" if is_binary else "FAIL",
            "observed": f"Target classes: {unique_targets} (Count: {len(unique_targets)})",
            "expected": f"Binary target required ({expected_targets})"
        })
        if not is_binary:
            failures.append(f"Target is not binary: observed classes {unique_targets}")

    # 7. No Completely Empty Columns Check
    empty_cols = [col for col in df_clean.columns if df_clean[col].isna().all()]
    no_empty_pass = len(empty_cols) == 0
    checks.append({
        "name": "no_empty_columns",
        "status": "PASS" if no_empty_pass else "FAIL",
        "observed": "No completely empty columns" if no_empty_pass else f"Completely empty columns found: {empty_cols}",
        "expected": "All columns must contain at least one valid non-null value"
    })
    if not no_empty_pass:
        failures.append(f"Completely empty columns: {empty_cols}")

    # 8. Missing Value Percentage Check
    total_cells = df_clean.size
    # Count missing values including '?' character used in raw adult dataset
    missing_cells = int(((df_clean.isna()) | (df_clean == "?")).values.sum())
    missing_pct = (missing_cells / total_cells) * 100.0 if total_cells > 0 else 0.0
    max_missing_limit = config.get("maximum_missing_percent", 10.0)
    missing_pass = missing_pct <= max_missing_limit
    checks.append({
        "name": "missing_value_percentage",
        "status": "PASS" if missing_pass else "WARN",
        "observed": f"{missing_pct:.2f}% missing/sentinel values",
        "expected": f"Missing value percentage <= {max_missing_limit}%"
    })
    if not missing_pass:
        warnings.append(f"Missing value percentage ({missing_pct:.2f}%) exceeds limit of {max_missing_limit}%")

    # 9. Duplicate Rows Measurement Check
    dup_count = int(df_clean.duplicated().sum())
    dup_pct = (dup_count / len(df_clean)) * 100.0 if len(df_clean) > 0 else 0.0
    checks.append({
        "name": "duplicate_rows_measured",
        "status": "PASS",
        "observed": f"{dup_count:,} duplicate rows ({dup_pct:.2f}%)",
        "expected": "Duplicate rows measured and recorded"
    })

    # 10. Minimum Row Count Threshold Check
    min_rows = config.get("minimum_rows", 30000)
    row_count_pass = len(df_clean) >= min_rows
    checks.append({
        "name": "row_count_above_minimum",
        "status": "PASS" if row_count_pass else "FAIL",
        "observed": f"{len(df_clean):,} raw rows",
        "expected": f"Row count >= {min_rows:,} rows"
    })
    if not row_count_pass:
        failures.append(f"Row count ({len(df_clean):,}) below minimum requirement ({min_rows:,})")

    # 11. Protected Attribute Group Counts Check
    group_distributions = {}
    for attr in req_protected:
        if attr in df_clean.columns:
            group_distributions[attr] = df_clean[attr].value_counts().to_dict()

    checks.append({
        "name": "protected_attribute_group_counts",
        "status": "PASS" if len(group_distributions) == len(req_protected) else "WARN",
        "observed": f"Group distributions measured for {list(group_distributions.keys())}",
        "expected": f"Distributions measured for all protected attributes: {req_protected}"
    })

    # 12. Intersectional Group Sizes Check
    inter_group_counts = {}
    if protected_pass:
        inter_series = df_clean[req_protected[0]].astype(str) + " + " + df_clean[req_protected[1]].astype(str)
        inter_counts = inter_series.value_counts().to_dict()
        inter_group_counts = {str(k): int(v) for k, v in inter_counts.items()}

    min_group_size = config.get("minimum_intersectional_group_size", 30)
    low_sample_groups = [g for g, c in inter_group_counts.items() if c < min_group_size]

    checks.append({
        "name": "intersectional_groups_measurable",
        "status": "PASS",
        "observed": f"{len(inter_group_counts)} subgroups ({len(low_sample_groups)} low-sample groups N < {min_group_size})",
        "expected": f"Intersectional subgroup distributions measured against safety threshold (N >= {min_group_size})"
    })

    # 13. Schema Compatibility Check
    schema_pass = target_exists and protected_pass and len(df_clean.columns) > len(req_protected) + 1
    checks.append({
        "name": "schema_compatibility",
        "status": "PASS" if schema_pass else "FAIL",
        "observed": f"Total columns: {len(df_clean.columns)}, Features: {len(df_clean.columns) - len(req_protected) - 1}",
        "expected": "Schema must contain target, protected attributes, and feature columns"
    })

    # 14. No Silent Schema Changes Check
    expected_col_count = 15
    schema_change_pass = len(df_clean.columns) == expected_col_count
    checks.append({
        "name": "no_silent_schema_changes",
        "status": "PASS" if schema_change_pass else "WARN",
        "observed": f"Observed column count: {len(df_clean.columns)}",
        "expected": f"Expected raw column count: {expected_col_count}"
    })

    # 15. Quality Gate Configuration Validity Check
    config_valid = isinstance(config, dict) and "dataset_path" in config
    checks.append({
        "name": "config_validity",
        "status": "PASS" if config_valid else "FAIL",
        "observed": "Quality configuration object valid" if config_valid else "Invalid quality configuration",
        "expected": "Quality gate configuration dictionary must be valid"
    })

    # Overall Status Calculation
    overall_status = "FAIL" if len(failures) > 0 else ("WARN" if len(warnings) > 0 else "PASS")
    passed_count = sum(1 for c in checks if c["status"] == "PASS")
    failed_count = sum(1 for c in checks if c["status"] == "FAIL")

    return {
        "overall_status": overall_status,
        "config_version": config.get("config_version", "1.0"),
        "dataset_hash": file_hash,
        "file_size_bytes": file_size,
        "raw_row_count": len(df_clean),
        "raw_column_count": len(df_clean.columns),
        "missing_value_percentage": missing_pct,
        "duplicate_row_count": dup_count,
        "quality_gates_passed": passed_count,
        "quality_gates_failed": failed_count,
        "group_distributions": group_distributions,
        "intersectional_group_counts": inter_group_counts,
        "checks": checks,
        "warnings": warnings,
        "failures": failures
    }
