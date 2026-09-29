"""
Expected-Result Oracle Helper Module.

Provides functions for loading, parsing, and validating frozen Phase 3 expected audit results
from 'evidence/reference/intersectional_expected_results.json'.
"""

import json
import os


def get_expected_results_path():
    """Return path to frozen expected-results oracle file."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(root, "evidence", "reference", "intersectional_expected_results.json")


def load_expected_results():
    """
    Load frozen expected-results oracle dictionary.

    Returns:
        dict: Expected results structure.
    """
    path = get_expected_results_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected results oracle file missing at '{path}'")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_reference_dataset(dataset_identifier: str) -> bool:
    """Return True if dataset_identifier corresponds to the frozen Adult reference dataset."""
    ident = dataset_identifier.lower().strip()
    return ident in ["adult_income", "adult_census_income", "adult", "data/raw/adult.csv"]


def verify_expected_oracle_integrity():
    """
    Verify structural integrity of the frozen expected-result oracle for the reference dataset.

    Returns:
        bool: True if oracle is structurally complete and valid.
    """
    oracle = load_expected_results()
    required_keys = ["oracle_metadata", "expected_group_summary", "expected_group_metrics", "expected_disparities"]
    for key in required_keys:
        if key not in oracle:
            return False

    meta = oracle["oracle_metadata"]
    if meta.get("test_sample_count") != 6033:
        return False

    if meta.get("min_group_size_threshold") != 30:
        return False

    return True
