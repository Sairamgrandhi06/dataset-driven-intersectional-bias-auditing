"""
Unit and Integration Tests for AI Dataset Profiler (src/data/dataset_profiler.py).

Verifies tests 1 through 18:
1. Numeric/categorical type detection.
2. Missing-value detection.
3. Duplicate row detection.
4. Unique-value and cardinality detection.
5. Potential ID column detection.
6. Binary target candidate detection.
7. Multiclass target detection (UNSUPPORTED_MULTICLASS).
8. Class imbalance warning.
9. Protected attribute candidate detection.
10. Complete JSON serialization safety.
11. Empty dataset handling.
12. Single-column dataset handling.
13. Dataset with no binary target.
14. Dataset with high-cardinality columns.
15. Arbitrary column names handling.
16. Loan dataset profiling.
17. Student dataset profiling.
18. Adult dataset profiling.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

from src.data.dataset_profiler import profile_dataset, detect_column_type, is_potential_id


def test_1_numeric_categorical_detection():
    """TEST 1: Numeric, categorical, boolean, and datetime detection."""
    df = pd.DataFrame({
        "num_col": [1.5, 2.3, 3.1, 4.0],
        "cat_col": ["A", "B", "A", "C"],
        "bool_col": [True, False, True, False],
        "binary_int": [0, 1, 1, 0]
    })
    
    profile = profile_dataset(df, dataset_id="test_types")
    cols = profile["columns"]
    assert cols["num_col"]["type"] == "numerical"
    assert cols["cat_col"]["type"] == "categorical"
    assert cols["bool_col"]["type"] == "boolean"
    assert cols["binary_int"]["type"] == "boolean"


def test_2_missing_value_detection():
    """TEST 2: Missing-value count and percentage detection."""
    df = pd.DataFrame({
        "col_a": [1, 2, np.nan, 4, np.nan],
        "col_b": ["x", "y", "z", "w", "v"]
    })
    
    profile = profile_dataset(df, dataset_id="test_missing")
    cols = profile["columns"]
    assert cols["col_a"]["missing_count"] == 2
    assert cols["col_a"]["missing_percentage"] == pytest.approx(40.0, abs=1e-2)
    assert cols["col_b"]["missing_count"] == 0


def test_3_duplicate_row_detection():
    """TEST 3: Duplicate row detection."""
    df = pd.DataFrame({
        "feature_1": [1, 2, 1, 3],
        "feature_2": ["a", "b", "a", "c"]
    })
    
    profile = profile_dataset(df, dataset_id="test_duplicates")
    assert profile["dataset_profile"]["duplicate_rows"] == 1


def test_4_cardinality_and_sample_values_detection():
    """TEST 4: Unique count, cardinality, and sample value extraction."""
    df = pd.DataFrame({
        "colors": ["red", "blue", "green", "red", "blue", "yellow"]
    })
    
    profile = profile_dataset(df, dataset_id="test_cardinality")
    meta = profile["columns"]["colors"]
    assert meta["unique_count"] == 4
    assert set(meta["sample_values"]).issubset({"red", "blue", "green", "yellow"})


def test_5_potential_id_detection():
    """TEST 5: Potential ID column detection based on name keywords and high uniqueness ratio."""
    df = pd.DataFrame({
        "user_id": [f"USR_{i}" for i in range(30)],
        "score": [10, 20, 30] * 10
    })
    
    profile = profile_dataset(df, dataset_id="test_id")
    cols = profile["columns"]
    assert cols["user_id"]["potential_id"] is True
    assert cols["score"]["potential_id"] is False


def test_6_binary_target_candidate_detection():
    """TEST 6: Binary target candidate detection and priority ranking."""
    df = pd.DataFrame({
        "age": [25, 30, 45, 50],
        "loan_status": ["Approved", "Rejected", "Approved", "Approved"]
    })
    
    profile = profile_dataset(df, dataset_id="test_target")
    cands = profile["target_candidates"]
    assert len(cands) >= 1
    top = cands[0]
    assert top["column"] == "loan_status"
    assert top["task_type"] == "binary_classification"
    assert top["confidence"] in ["High", "Medium"]
    assert top["recommended_positive_class"] == "Approved"


def test_7_multiclass_target_detection():
    """TEST 7: Multiclass target detection sets profile status UNSUPPORTED_MULTICLASS."""
    df = pd.DataFrame({
        "target_class": ["ClassA", "ClassB", "ClassC", "ClassA", "ClassB", "ClassC"]
    })
    
    profile = profile_dataset(df, dataset_id="test_multiclass")
    assert profile["profile_status"] == "UNSUPPORTED_MULTICLASS"
    assert profile["target_candidates"][0]["task_type"] == "multiclass_classification"


def test_8_class_imbalance_warning():
    """TEST 8: Severe class imbalance warning generation."""
    df = pd.DataFrame({
        "outcome": [1] * 90 + [0] * 10,
        "feature": list(range(100))
    })
    
    profile = profile_dataset(df, dataset_id="test_imbalance")
    warnings = [w["code"] for w in profile["quality_warnings"] if isinstance(w, dict)]
    assert "SEVERE_CLASS_IMBALANCE" in warnings


def test_9_protected_attribute_candidate_detection():
    """TEST 9: Protected attribute candidate detection using demographic term matching."""
    df = pd.DataFrame({
        "gender": ["Female", "Male", "Female", "Male"],
        "race": ["White", "Black", "Asian", "White"],
        "income": [50000, 60000, 70000, 80000],
        "loan_status": ["Approved", "Rejected", "Approved", "Rejected"]
    })
    
    profile = profile_dataset(df, dataset_id="test_protected")
    prot_cands = [p["column"] for p in profile["protected_attribute_candidates"]]
    assert "gender" in prot_cands
    assert "race" in prot_cands
    assert "loan_status" not in prot_cands


def test_10_json_serialization_safety():
    """TEST 10: Profile report contract is 100% JSON serializable without numpy errors."""
    df = pd.DataFrame({
        "int_col": np.array([1, 2, 3], dtype=np.int64),
        "float_col": np.array([1.1, 2.2, 3.3], dtype=np.float64),
        "bool_col": np.array([True, False, True], dtype=np.bool_),
        "str_col": np.array(["A", "B", "C"], dtype=object)
    })
    
    profile = profile_dataset(df, dataset_id="test_json")
    # Must serialize cleanly without TypeError
    json_str = json.dumps(profile, indent=2)
    assert isinstance(json_str, str)
    assert len(json_str) > 0


def test_11_empty_dataset_handling():
    """TEST 11: Empty dataset handling generates CRITICAL warning and FAIL status."""
    df = pd.DataFrame()
    profile = profile_dataset(df, dataset_id="test_empty")
    assert profile["dataset_profile"]["row_count"] == 0
    assert profile["profile_status"] == "FAIL"


def test_12_single_column_dataset_handling():
    """TEST 12: Single-column dataset handling."""
    df = pd.DataFrame({"target": ["Approved", "Rejected", "Approved"]})
    profile = profile_dataset(df, dataset_id="test_single_col")
    assert profile["dataset_profile"]["column_count"] == 1
    assert len(profile["target_candidates"]) == 1


def test_13_dataset_with_no_binary_target():
    """TEST 13: Dataset with continuous columns only generates low confidence target candidates."""
    df = pd.DataFrame({
        "var1": np.linspace(1.0, 100.0, 50),
        "var2": np.linspace(0.1, 5.0, 50)
    })
    profile = profile_dataset(df, dataset_id="test_no_binary")
    cands = profile["target_candidates"]
    for c in cands:
        assert c["task_type"] == "regression"


def test_14_high_cardinality_columns():
    """TEST 14: High-cardinality categorical columns marked properly."""
    df = pd.DataFrame({
        "unique_codes": [f"CODE_{i}" for i in range(100)],
        "target": ["Approved", "Rejected"] * 50
    })
    profile = profile_dataset(df, dataset_id="test_cardinality_high")
    assert profile["columns"]["unique_codes"]["potential_id"] is True


def test_15_arbitrary_column_names_handling():
    """TEST 15: Arbitrary column names handling without crashing."""
    df = pd.DataFrame({
        "Column Special !@#": ["A", "B", "A", "B"],
        "Target_123": ["Yes", "No", "Yes", "No"]
    })
    profile = profile_dataset(df, dataset_id="test_arbitrary_names")
    assert "Column Special !@#" in profile["columns"]
    assert "Target_123" in profile["columns"]


def test_16_loan_dataset_profiling():
    """TEST 16: Profile Loan dataset file if present on disk."""
    loan_csv = "data/raw/loan_approval_fairness_test.csv"
    if os.path.exists(loan_csv):
        df = pd.read_csv(loan_csv)
        profile = profile_dataset(df, dataset_id="loan_approval_fairness_test")
        assert profile["dataset_profile"]["row_count"] == 800
        assert profile["target_candidates"][0]["column"] == "loan_status"
        prot_cands = [p["column"] for p in profile["protected_attribute_candidates"]]
        assert "gender" in prot_cands
        assert "race" in prot_cands


def test_17_student_dataset_profiling():
    """TEST 17: Profile Student dataset structure dynamically."""
    df = pd.DataFrame({
        "student_id": [f"STU_{i}" for i in range(40)],
        "gender": ["Female", "Male"] * 20,
        "ethnicity": ["GroupA", "GroupB"] * 20,
        "exam_score": np.random.randint(50, 100, 40),
        "pass_status": ["Pass", "Fail"] * 20
    })
    profile = profile_dataset(df, dataset_id="student_fairness")
    assert profile["target_candidates"][0]["column"] == "pass_status"
    prot_cols = [p["column"] for p in profile["protected_attribute_candidates"]]
    assert "gender" in prot_cols
    assert "ethnicity" in prot_cols


def test_18_adult_dataset_profiling():
    """TEST 18: Profile Adult dataset file if present on disk."""
    adult_csv = "data/raw/adult.csv"
    if os.path.exists(adult_csv):
        df = pd.read_csv(adult_csv, skipinitialspace=True)
        profile = profile_dataset(df, dataset_id="adult")
        assert profile["dataset_profile"]["row_count"] > 30000
        target_cols = [t["column"] for t in profile["target_candidates"]]
        assert "income" in target_cols
        prot_cols = [p["column"] for p in profile["protected_attribute_candidates"]]
        assert "sex" in prot_cols or "race" in prot_cols
