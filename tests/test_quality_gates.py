"""
Unit tests for Data Quality Gates module (src/data/quality_gates.py).
"""

import os
import pytest
import pandas as pd

from src.data.quality_gates import (
    compute_file_sha256,
    load_quality_config,
    run_data_quality_gates
)


def test_compute_file_sha256_existing():
    """Verify compute_file_sha256 returns valid 64-char hex hash for existing raw dataset."""
    raw_path = "data/raw/adult.csv"
    if os.path.exists(raw_path):
        file_hash = compute_file_sha256(raw_path)
        assert len(file_hash) == 64
        assert file_hash != "UNKNOWN"


def test_compute_file_sha256_missing():
    """Verify compute_file_sha256 returns UNKNOWN for missing file paths."""
    assert compute_file_sha256("non_existent_file_99.csv") == "UNKNOWN"


def test_quality_gates_pass_on_valid_data():
    """Verify quality gates pass 15/15 on valid Adult dataset."""
    report = run_data_quality_gates()
    assert report["overall_status"] == "PASS"
    assert report["quality_gates_passed"] == 15
    assert report["quality_gates_failed"] == 0
    assert len(report["checks"]) == 15


def test_quality_gates_missing_target():
    """Verify quality gates fail when target column is missing from DataFrame."""
    df_invalid = pd.DataFrame({
        "sex": ["Male", "Female"],
        "race": ["White", "Black"],
        "age": [30, 40]
    })
    report = run_data_quality_gates(df=df_invalid)
    assert report["overall_status"] == "FAIL"
    checks: list[dict] = report["checks"]
    assert any(c["name"] == "target_exists" and c["status"] == "FAIL" for c in checks)


def test_quality_gates_missing_protected_attributes():
    """Verify quality gates fail when protected attributes are missing."""
    df_invalid = pd.DataFrame({
        "income": ["<=50K", ">50K"],
        "age": [30, 40]
    })
    report = run_data_quality_gates(df=df_invalid)
    assert report["overall_status"] == "FAIL"
    checks: list[dict] = report["checks"]
    assert any(c["name"] == "protected_attributes_exist" and c["status"] == "FAIL" for c in checks)


def test_quality_gates_non_binary_target():
    """Verify quality gates fail when target column is non-binary (3 classes)."""
    df_invalid = pd.DataFrame({
        "income": ["<=50K", ">50K", "High"],
        "sex": ["Male", "Female", "Male"],
        "race": ["White", "Black", "White"]
    })
    report = run_data_quality_gates(df=df_invalid)
    assert report["overall_status"] == "FAIL"
    checks: list[dict] = report["checks"]
    assert any(c["name"] == "target_binary_compatible" and c["status"] == "FAIL" for c in checks)


def test_quality_gates_row_count_threshold():
    """Verify row count threshold check fails when rows are below minimum requirement."""
    df_small = pd.DataFrame({
        "income": ["<=50K", ">50K"],
        "sex": ["Male", "Female"],
        "race": ["White", "Black"]
    })
    report = run_data_quality_gates(df=df_small)
    assert report["overall_status"] == "FAIL"
    checks: list[dict] = report["checks"]
    assert any(c["name"] == "row_count_above_minimum" and c["status"] == "FAIL" for c in checks)
