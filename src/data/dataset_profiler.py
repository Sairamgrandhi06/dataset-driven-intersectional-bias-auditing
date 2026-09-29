"""
AI Dataset Profiler & Heuristic Recommendation Engine.

Analyzes raw tabular DataFrames to extract dataset metadata, column dtypes, missingness,
cardinality, near-constant fields, and potential ID columns.
Detects binary target candidates and protected attribute candidates using explainable
heuristic scoring, generates data quality warnings, and produces a 100% JSON-serializable
profiling contract.
"""

import re
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


def to_py_type(val: Any) -> Any:
    """Recursively convert NumPy/Pandas objects into native Python primitives for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, (list, tuple, np.ndarray, pd.Index)):
        return [to_py_type(item) for item in val]
    if isinstance(val, dict):
        return {str(k): to_py_type(v) for k, v in val.items()}
    
    try:
        if pd.isna(val):
            return None
    except (ValueError, TypeError):
        pass

    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(val)
    if isinstance(val, (str, np.str_)):
        return str(val)
    return str(val)


def detect_column_type(series: pd.Series) -> str:
    """Infer high-level column data type: numerical, categorical, boolean, or datetime."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # Try datetime parsing for string series if format matches
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = series.dropna().head(20)
        if not sample.empty:
            try:
                pd.to_datetime(sample, errors="raise")
                return "datetime"
            except (ValueError, TypeError):
                pass

    if pd.api.types.is_numeric_dtype(series):
        # Check if numeric series is actually boolean (0 and 1 only)
        unique_vals = set(series.dropna().unique())
        if unique_vals.issubset({0, 1}) and len(unique_vals) <= 2:
            return "boolean"
        return "numerical"

    return "categorical"


def is_potential_id(col_name: str, series: pd.Series) -> bool:
    """
    Identify likely identifier columns using name heuristics, dtype semantics, and uniqueness ratio.
    Continuous numerical predictive features (like income, amount, score, age) are NEVER classified as IDs.
    """
    name_clean = str(col_name).lower().strip()
    tokens = set(re.split(r'[^a-z0-9]+', name_clean))
    
    # Specific ID naming patterns
    has_id_keyword = bool(
        (tokens & {"id", "uuid", "guid", "ssn", "identifier"})
        or name_clean.endswith("_id")
        or name_clean.startswith("id_")
        or (name_clean.endswith("id") and len(name_clean) > 2 and name_clean not in ["paid", "unpaid", "grid", "solid", "liquid", "valid", "invalid", "hybrid", "pyramid", "acid", "lipid", "mid"])
        or name_clean in ["id", "uuid", "guid", "ssn", "identifier", "pk", "uid"]
    )

    n_samples = len(series)
    if n_samples == 0:
        return False
    
    clean_series = series.dropna()
    unique_count = clean_series.nunique()
    uniqueness_ratio = (unique_count / n_samples) if n_samples > 0 else 0.0

    is_numeric = pd.api.types.is_numeric_dtype(series)

    # For numeric columns: High uniqueness alone is normal for continuous features (income, amount, score, age).
    # A numeric column is only an ID if it explicitly matches ID keywords AND has high uniqueness.
    if is_numeric:
        if not has_id_keyword:
            return False
        return (uniqueness_ratio > 0.80) and (n_samples > 10)

    # For string/object columns:
    if has_id_keyword:
        return True
    
    # String column with extremely high uniqueness (>98%) and structured alphanumeric ID patterns
    if uniqueness_ratio > 0.98 and n_samples > 20:
        sample = clean_series.head(10).astype(str)
        has_alphanumeric_code = any(
            bool(re.search(r'^[a-zA-Z0-9_\-]+$', val)) and bool(re.search(r'\d', val)) and bool(re.search(r'[a-zA-Z]', val))
            for val in sample
        )
        if has_alphanumeric_code:
            return True

    return False


def detect_target_candidates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect potential target columns prioritized by confidence score.
    Returns list of candidate dictionaries.
    """
    candidates = []
    target_keywords = [
        "target", "outcome", "label", "status", "class", "result", "approved",
        "default", "decision", "income", "churn", "response", "credit_risk",
        "loan_status", "grant_status", "pass", "survived", "readmitted"
    ]

    for col in df.columns:
        series = df[col]
        n_samples = len(series)
        if n_samples == 0:
            continue

        clean_series = series.dropna()
        unique_vals = clean_series.unique()
        unique_count = len(unique_vals)

        if unique_count < 2:
            continue

        name_lower = str(col).lower().strip()
        matches_kw = any(kw in name_lower for kw in target_keywords)

        # Candidate assessment
        if unique_count == 2:
            task_type = "binary_classification"
            # Base confidence for binary columns
            confidence_score = 0.85 if matches_kw else 0.60
            reason = f"Binary column with 2 distinct classes ({list(unique_vals)[:2]})."
            if matches_kw:
                reason += f" Column name '{col}' matches common target keywords."
        elif 3 <= unique_count <= 20 and not pd.api.types.is_float_dtype(series):
            task_type = "multiclass_classification"
            confidence_score = 0.50 if matches_kw else 0.30
            reason = f"Categorical column with {unique_count} distinct classes."
        else:
            task_type = "regression"
            confidence_score = 0.40 if matches_kw else 0.10
            reason = f"Continuous numerical column with {unique_count} unique values."

        # Class distribution & recommended positive class for binary targets
        class_dist = {}
        counts = clean_series.value_counts()
        for cls_val, cnt in counts.items():
            class_dist[str(cls_val)] = {
                "count": int(cnt),
                "percentage": float((cnt / len(clean_series)) * 100.0)
            }

        rec_pos_class = None
        imbalance_ratio = 1.0
        if task_type == "binary_classification":
            c_list = list(counts.index)
            # Recommend positive class based on affirmative keywords
            pos_keywords = ["approved", ">50k", "1", "yes", "true", "pass", "good", "granted", "paid", "survived", "positive", "accepted"]
            rec_pos_class = str(c_list[0])
            for c in c_list:
                if str(c).lower().strip().rstrip(".") in pos_keywords:
                    rec_pos_class = str(c)
                    break
            
            maj_cnt = counts.iloc[0]
            min_cnt = counts.iloc[-1] if len(counts) > 1 else maj_cnt
            imbalance_ratio = float(maj_cnt / min_cnt) if min_cnt > 0 else 1.0

        candidates.append({
            "column": str(col),
            "task_type": task_type,
            "confidence_score": float(confidence_score),
            "confidence": "High" if confidence_score >= 0.75 else ("Medium" if confidence_score >= 0.40 else "Low"),
            "reason": reason,
            "classes": [str(v) for v in unique_vals[:10]],
            "class_distribution": class_dist,
            "imbalance_ratio": imbalance_ratio,
            "recommended_positive_class": rec_pos_class
        })

    # Sort candidates by confidence score descending
    candidates.sort(key=lambda x: x["confidence_score"], reverse=True)
    return candidates


def detect_protected_attribute_candidates(
    df: pd.DataFrame,
    target_col: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Suggest potential protected attributes using generic demographic heuristics.
    Returns list of protected candidate dictionaries.
    """
    candidates = []
    demo_keywords = [
        "gender", "sex", "race", "ethnicity", "age", "disability", "religion",
        "nationality", "marital", "family", "skin", "caste", "veteran",
        "income_group", "education", "location", "zip", "region", "native_country"
    ]

    for col in df.columns:
        if target_col and col == target_col:
            continue

        series = df[col]
        clean_series = series.dropna()
        unique_count = clean_series.nunique()

        if unique_count < 2 or is_potential_id(col, series):
            continue

        name_lower = str(col).lower().strip()
        matches_kw = any(kw in name_lower for kw in demo_keywords)
        col_type = detect_column_type(series)

        if matches_kw:
            confidence_score = 0.90 if unique_count <= 20 else 0.65
            reason = f"Column name '{col}' matches known sensitive demographic attributes ({unique_count} categories)."
        elif col_type in ["categorical", "boolean"] and 2 <= unique_count <= 10:
            confidence_score = 0.50
            reason = f"Discrete categorical column with {unique_count} distinct groups."
        else:
            continue

        unique_vals = clean_series.unique()
        candidates.append({
            "column": str(col),
            "confidence_score": float(confidence_score),
            "confidence": "High" if confidence_score >= 0.75 else ("Medium" if confidence_score >= 0.40 else "Low"),
            "reason": reason,
            "unique_count": int(unique_count),
            "unique_values": [str(v) for v in unique_vals[:10]]
        })

    candidates.sort(key=lambda x: x["confidence_score"], reverse=True)
    return candidates


def generate_quality_warnings(
    df: pd.DataFrame,
    columns_meta: Dict[str, Any],
    target_cand: List[Dict[str, Any]],
    protected_cand: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Generate informational data quality warnings without silently altering dataset content."""
    warnings = []
    n_rows = len(df)
    n_cols = len(df.columns)

    if n_rows == 0:
        warnings.append({
            "code": "EMPTY_DATASET",
            "severity": "CRITICAL",
            "message": "Dataset contains 0 rows."
        })
        return warnings

    if n_rows < 500:
        warnings.append({
            "code": "SMALL_SAMPLE_SIZE",
            "severity": "WARNING",
            "message": f"Dataset sample size is small (N = {n_rows:,} rows). Intersectional subgroup sample counts may fall below safety thresholds."
        })

    # Duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = (dup_count / n_rows) * 100.0
        warnings.append({
            "code": "DUPLICATE_ROWS",
            "severity": "INFO",
            "message": f"Dataset contains {dup_count:,} duplicate rows ({dup_pct:.2f}% of total)."
        })

    # Missing values
    missing_cols = [col for col, meta in columns_meta.items() if meta["missing_count"] > 0]
    if missing_cols:
        warnings.append({
            "code": "MISSING_VALUES",
            "severity": "WARNING",
            "message": f"{len(missing_cols)} columns contain missing values: {missing_cols[:5]}."
        })

    # Constant & Near-constant columns
    constant_cols = [col for col, meta in columns_meta.items() if meta["is_constant"]]
    if constant_cols:
        warnings.append({
            "code": "CONSTANT_COLUMNS",
            "severity": "WARNING",
            "message": f"Columns with zero variance (single value): {constant_cols}."
        })

    # Potential ID columns
    id_cols = [col for col, meta in columns_meta.items() if meta["potential_id"]]
    if id_cols:
        warnings.append({
            "code": "POTENTIAL_ID_COLUMNS",
            "severity": "INFO",
            "message": f"Detected potential identifier columns: {id_cols}. Recommend excluding them from feature encoding."
        })

    # Target class imbalance check
    if target_cand:
        top_target = target_cand[0]
        if top_target["task_type"] == "binary_classification":
            imb = top_target.get("imbalance_ratio", 1.0)
            if imb > 4.0:
                warnings.append({
                    "code": "SEVERE_CLASS_IMBALANCE",
                    "severity": "WARNING",
                    "message": f"Target '{top_target['column']}' exhibits severe class imbalance (Ratio: {imb:.2f}:1)."
                })

    return warnings


def profile_dataset(df: pd.DataFrame, dataset_id: str = "dataset") -> Dict[str, Any]:
    """
    Perform complete dataset profiling and generate standardized JSON-serializable report contract.

    Parameters:
        df (pd.DataFrame): Raw input pandas DataFrame.
        dataset_id (str): Unique dataset identifier string.

    Returns:
        dict: Standardized, JSON-serializable dataset profile report.
    """
    n_rows = len(df)
    n_cols = len(df.columns)
    dup_rows = int(df.duplicated().sum()) if n_rows > 0 else 0
    mem_usage = float(df.memory_usage(deep=True).sum() / 1024.0) if n_rows > 0 else 0.0

    columns_meta = {}
    for col in df.columns:
        series = df[col]
        c_type = detect_column_type(series)
        m_cnt = int(series.isna().sum())
        m_pct = float((m_cnt / n_rows) * 100.0) if n_rows > 0 else 0.0
        u_cnt = int(series.nunique())
        pot_id = is_potential_id(str(col), series)
        is_const = (u_cnt <= 1)
        
        # Check near-constant (top value > 99%)
        is_near_const = False
        if n_rows > 0 and u_cnt > 1:
            top_freq = series.value_counts().iloc[0]
            if (top_freq / n_rows) > 0.99:
                is_near_const = True

        sample_vals = [to_py_type(v) for v in series.dropna().unique()[:5]]

        columns_meta[str(col)] = {
            "name": str(col),
            "type": c_type,
            "dtype": str(series.dtype),
            "missing_count": m_cnt,
            "missing_percentage": m_pct,
            "unique_count": u_cnt,
            "sample_values": sample_vals,
            "potential_id": pot_id,
            "is_constant": is_const,
            "is_near_constant": is_near_const
        }

    target_candidates = detect_target_candidates(df)
    top_target_col = target_candidates[0]["column"] if target_candidates else None

    protected_candidates = detect_protected_attribute_candidates(df, target_col=top_target_col)
    quality_warnings = generate_quality_warnings(df, columns_meta, target_candidates, protected_candidates)

    # Determine overall profile status
    if n_rows == 0:
        profile_status = "FAIL"
    elif target_candidates and target_candidates[0]["task_type"] == "multiclass_classification":
        profile_status = "UNSUPPORTED_MULTICLASS"
    elif quality_warnings:
        profile_status = "WARN"
    else:
        profile_status = "PASS"

    profile_report = {
        "dataset_id": dataset_id,
        "dataset_profile": {
            "row_count": n_rows,
            "column_count": n_cols,
            "duplicate_rows": dup_rows,
            "memory_usage_kb": round(mem_usage, 2)
        },
        "columns": columns_meta,
        "target_candidates": target_candidates,
        "protected_attribute_candidates": protected_candidates,
        "quality_warnings": quality_warnings,
        "profile_status": profile_status
    }

    # Ensure complete JSON serializability
    return to_py_type(profile_report)
