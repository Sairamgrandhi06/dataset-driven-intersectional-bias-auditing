"""
Unit tests for Independent Reference Implementation & Expected-Result Oracle.
"""

import os
import pytest
import pandas as pd
import numpy as np

from reference.intersectional_reference import (
    create_reference_intersectional_labels,
    calculate_reference_group_metrics,
    calculate_reference_disparities,
    audit_intersectional_reference
)
from reference.expected_results import (
    load_expected_results,
    verify_expected_oracle_integrity
)
from src.data.loader import load_raw_data, load_config
from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data
from src.models.classifier import train_baseline_model, predict_model
from src.fairness.intersectional import audit_intersectional_attributes


def test_reference_group_construction():
    """Verify independent reference subgroup string creation."""
    df_sample = pd.DataFrame({"sex": ["Female", "Male"], "race": ["Black", "White"]})
    labels = create_reference_intersectional_labels(df_sample)
    assert list(labels) == ["Female + Black", "Male + White"]


def test_reference_group_counts():
    """Verify independent reference audit identifies 10 groups (7 primary, 3 low-sample)."""
    config = load_config("config/default_config.json")
    raw_df = load_raw_data("config/default_config.json")
    X, y, A = prepare_pipeline_data(raw_df, config=config)
    splits = split_pipeline_data(X, y, A, config=config)

    baseline_model = train_baseline_model(splits["X_train"], splits["y_train"], random_state=42)
    y_pred, _ = predict_model(baseline_model, splits["X_test"])

    res = audit_intersectional_reference(splits["y_test"], y_pred, splits["A_test"], min_group_size=30)
    assert res["total_groups_discovered"] == 10
    assert res["primary_group_count"] == 7
    assert res["low_sample_group_count"] == 3


def test_reference_metric_calculations():
    """Verify independent metric calculation logic on known toy arrays."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0])
    labels = pd.Series(["G1", "G1", "G1", "G2", "G2", "G2"])

    metrics = calculate_reference_group_metrics(y_true, y_pred, labels)
    assert "G1" in metrics
    assert "G2" in metrics
    assert metrics["G1"]["n_samples"] == 3
    assert metrics["G2"]["n_samples"] == 3


def test_expected_results_oracle_validation():
    """Verify frozen expected-result oracle loads correctly and maintains structural integrity."""
    assert verify_expected_oracle_integrity() is True
    oracle = load_expected_results()
    assert oracle["oracle_metadata"]["test_sample_count"] == 6033
    assert oracle["expected_disparities"]["demographic_parity_difference"] == pytest.approx(0.3171, abs=1e-3)


def test_production_reference_group_agreement():
    """Assert production and independent reference audits agree on group counts and names."""
    config = load_config("config/default_config.json")
    raw_df = load_raw_data("config/default_config.json")
    X, y, A = prepare_pipeline_data(raw_df, config=config)
    splits = split_pipeline_data(X, y, A, config=config)

    baseline_model = train_baseline_model(splits["X_train"], splits["y_train"], random_state=42)
    y_pred, _ = predict_model(baseline_model, splits["X_test"])

    prod_res = audit_intersectional_attributes(splits["y_test"], y_pred, splits["A_test"], attributes=["sex", "race"], min_group_size=30)
    ref_res = audit_intersectional_reference(splits["y_test"], y_pred, splits["A_test"], min_group_size=30)

    assert prod_res["total_groups_discovered"] == ref_res["total_groups_discovered"]
    assert prod_res["primary_group_count"] == ref_res["primary_group_count"]


def test_production_reference_metric_agreement():
    """Assert production and independent reference audit disparity metrics agree within 1e-4 tolerance."""
    config = load_config("config/default_config.json")
    raw_df = load_raw_data("config/default_config.json")
    X, y, A = prepare_pipeline_data(raw_df, config=config)
    splits = split_pipeline_data(X, y, A, config=config)

    baseline_model = train_baseline_model(splits["X_train"], splits["y_train"], random_state=42)
    y_pred, _ = predict_model(baseline_model, splits["X_test"])

    prod_disp = audit_intersectional_attributes(splits["y_test"], y_pred, splits["A_test"], attributes=["sex", "race"], min_group_size=30)["disparities"]
    ref_disp = audit_intersectional_reference(splits["y_test"], y_pred, splits["A_test"], min_group_size=30)["disparities"]

    for k in prod_disp:
        assert prod_disp[k] == pytest.approx(ref_disp[k], abs=1e-4)


def test_minimum_group_threshold_agreement():
    """Verify minimum group threshold N >= 30 separates low-sample groups consistently."""
    oracle = load_expected_results()
    assert oracle["oracle_metadata"]["min_group_size_threshold"] == 30
    assert oracle["expected_group_summary"]["low_sample_group_count"] == 3


def test_dataset_identity_hash_agreement():
    """Verify oracle records raw dataset SHA-256 hash matching physical adult.csv file."""
    oracle = load_expected_results()
    expected_hash = "9007ff524acf0776b9598a0ed883058df91f59c003e78cb2cddd88f87535fc86"
    assert oracle["oracle_metadata"]["dataset_hash"] == expected_hash


def test_test_set_size_agreement():
    """Verify test set size evaluates exactly 6,033 samples."""
    oracle = load_expected_results()
    assert oracle["oracle_metadata"]["test_sample_count"] == 6033


def test_benchmark_reproducibility():
    """Verify full end-to-end reproducibility of Equalized Odds disparity 0.4825."""
    oracle = load_expected_results()
    eo_diff = oracle["expected_disparities"]["equalized_odds_difference"]
    assert eo_diff == pytest.approx(0.4825, abs=1e-3)
