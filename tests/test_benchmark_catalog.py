"""
Unit & Integration Tests for Explicit Registered Benchmark Catalog Architecture.
"""

import os
import json
import pytest
import pandas as pd

from src.data.benchmark_catalog import (
    load_registered_benchmarks,
    get_available_benchmarks,
    get_benchmark_display_names,
    get_benchmark_by_id_or_name,
    is_registered_benchmark,
    resolve_benchmark_dataframe
)
from dashboard.utils import (
    get_project_root,
    get_raw_dataset_dataframe
)


def test_benchmark_catalog_contains_only_supported_benchmarks():
    """Verify that the benchmark catalog contains ONLY explicitly supported benchmarks."""
    catalog = load_registered_benchmarks()
    
    # Authoritative supported benchmarks
    assert "adult_census_income" in catalog
    assert "students" in catalog
    
    # Internal, experiment, and test dataset IDs must NOT be in the catalog
    forbidden_ids = [
        "loan_approval",
        "loan_approval_fairness_test",
        "loan_approval_fairness_test_1",
        "loan_monitoring_shifted_batch",
        "test_ds_unique",
        "test_interactive_run",
        "test_loan_demo_crit",
        "final_project_loan_demo__1",
        "test_interactive_ds"
    ]
    for fid in forbidden_ids:
        assert fid not in catalog, f"Historical/test dataset '{fid}' must not appear in benchmark catalog!"


def test_benchmark_selector_display_names_isolated():
    """Verify benchmark dropdown options do not expose internal historical registry entries."""
    display_names = get_benchmark_display_names()
    
    assert "Adult Census Income" in display_names
    assert "Student Academic Performance" in display_names
    
    for name in display_names:
        assert "loan" not in name.lower()
        assert "test" not in name.lower()
        assert "shifted" not in name.lower()
        assert "demo" not in name.lower()


def test_benchmark_by_id_or_name_resolution():
    """Verify case-insensitive lookup by ID and display name."""
    bm1 = get_benchmark_by_id_or_name("adult_census_income")
    assert bm1 is not None
    assert bm1["dataset_id"] == "adult_census_income"
    assert bm1["display_name"] == "Adult Census Income"

    bm2 = get_benchmark_by_id_or_name("Adult Census Income")
    assert bm2 is not None
    assert bm2["dataset_id"] == "adult_census_income"

    bm3 = get_benchmark_by_id_or_name("students")
    assert bm3 is not None
    assert bm3["dataset_id"] == "students"

    bm4 = get_benchmark_by_id_or_name("Student Academic Performance")
    assert bm4 is not None
    assert bm4["dataset_id"] == "students"

    # Negative lookup
    assert get_benchmark_by_id_or_name("loan_approval") is None
    assert get_benchmark_by_id_or_name("test_ds_unique") is None


def test_resolve_benchmark_dataframe_adult():
    """Verify Adult Census Income benchmark resolves valid DataFrame and config."""
    df, cfg, ds_id = resolve_benchmark_dataframe("adult_census_income")
    assert ds_id == "adult_census_income"
    assert df is not None
    assert len(df) > 1000
    assert "income" in df.columns
    assert cfg is not None
    assert cfg["target"]["column"] == "income"


def test_resolve_benchmark_dataframe_students():
    """Verify Student Academic Performance benchmark resolves valid DataFrame and config."""
    df, cfg, ds_id = resolve_benchmark_dataframe("students")
    assert ds_id == "students"
    assert df is not None
    assert len(df) > 1000
    assert "passed" in df.columns
    assert cfg is not None
    assert cfg["target"]["column"] == "passed"


def test_is_registered_benchmark_predicate():
    """Verify predicate correctly classifies benchmarks vs arbitrary/historical IDs."""
    assert is_registered_benchmark("adult_census_income") is True
    assert is_registered_benchmark("Adult Census Income") is True
    assert is_registered_benchmark("students") is True
    assert is_registered_benchmark("Student Academic Performance") is True

    assert is_registered_benchmark("loan_approval") is False
    assert is_registered_benchmark("custom_uploaded_csv") is False
    assert is_registered_benchmark("loan_monitoring_shifted_batch") is False


def test_get_raw_dataset_dataframe_resolves_benchmarks():
    """Verify get_raw_dataset_dataframe transparently loads registered benchmark data."""
    df_adult = get_raw_dataset_dataframe("adult_census_income")
    assert df_adult is not None
    assert len(df_adult) > 0

    df_students = get_raw_dataset_dataframe("students")
    assert df_students is not None
    assert len(df_students) > 0


def test_bundled_benchmark_files_exist_on_disk():
    """Verify physical CSV files exist in data/benchmarks/."""
    root = get_project_root()
    adult_csv = os.path.join(root, "data", "benchmarks", "adult_census_income.csv")
    students_csv = os.path.join(root, "data", "benchmarks", "students.csv")

    assert os.path.exists(adult_csv), f"Missing bundled benchmark file: {adult_csv}"
    assert os.path.exists(students_csv), f"Missing bundled benchmark file: {students_csv}"

    # Confirm non-empty
    assert os.path.getsize(adult_csv) > 100000
    assert os.path.getsize(students_csv) > 100000


def test_benchmark_configs_point_to_valid_existing_paths():
    """Verify config files point to paths that actually exist on disk."""
    root = get_project_root()
    
    # Adult config
    adult_cfg_path = os.path.join(root, "config", "adult_census_income_config.json")
    assert os.path.exists(adult_cfg_path)
    with open(adult_cfg_path, "r", encoding="utf-8") as f:
        adult_cfg = json.load(f)
    abs_adult_raw = os.path.join(root, adult_cfg["path"]) if not os.path.isabs(adult_cfg["path"]) else adult_cfg["path"]
    assert os.path.exists(abs_adult_raw), f"Adult dataset path in config does not exist: {abs_adult_raw}"

    # Students config
    students_cfg_path = os.path.join(root, "config", "students_config.json")
    assert os.path.exists(students_cfg_path)
    with open(students_cfg_path, "r", encoding="utf-8") as f:
        students_cfg = json.load(f)
    abs_students_raw = os.path.join(root, students_cfg["path"]) if not os.path.isabs(students_cfg["path"]) else students_cfg["path"]
    assert os.path.exists(abs_students_raw), f"Students dataset path in config does not exist: {abs_students_raw}"
