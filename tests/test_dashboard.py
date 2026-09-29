"""
Unit tests for Dashboard Utilities and Data Loaders.
"""

import os
import pytest
import pandas as pd

from dashboard.utils import (
    load_json_file,
    load_csv_file,
    get_figure_path,
    format_val,
    get_status_style
)


def test_load_json_file_valid():
    """Verify load_json_file reads existing processed JSON results."""
    data = load_json_file("tradeoff_results.json")
    assert data is not None
    assert isinstance(data, dict)
    assert "performance_metrics" in data
    assert "fairness_metrics" in data


def test_load_json_file_missing():
    """Verify load_json_file handles missing files gracefully returning None."""
    data = load_json_file("non_existent_file_xyz.json")
    assert data is None


def test_load_csv_file_valid():
    """Verify load_csv_file reads existing processed CSV summary tables."""
    df = load_csv_file("tradeoff_summary.csv")
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert "metric_name" in df.columns
    assert "impact_status" in df.columns


def test_load_csv_file_missing():
    """Verify load_csv_file handles missing files gracefully returning None."""
    df = load_csv_file("non_existent_file_xyz.csv")
    assert df is None


def test_get_figure_path_existing():
    """Verify get_figure_path returns valid path for existing figures."""
    fig_path = get_figure_path("baseline_vs_mitigated_calibration.png")
    assert fig_path is not None
    assert os.path.exists(fig_path)


def test_get_figure_path_missing():
    """Verify get_figure_path returns None for missing figures."""
    fig_path = get_figure_path("non_existent_figure_123.png")
    assert fig_path is None


def test_format_val():
    """Verify format_val correctly formats floating numbers and percentages."""
    assert format_val(0.83988) == "0.8399"
    assert format_val(0.83988, is_percentage=True) == "83.99%"
    assert format_val(None) == "N/A"


def test_get_status_style():
    """Verify status styling helper returns correct badges for Improved and Worsened."""
    s_imp = get_status_style("Improved")
    assert s_imp["label"] == "Improved"
    assert s_imp["color"] == "green"

    s_wor = get_status_style("Worsened")
    assert s_wor["label"] == "Worsened"
    assert s_wor["color"] == "red"

    s_neu = get_status_style("UnknownStatus")
    assert s_neu["label"] == "Neutral"


def test_format_val_edge_cases():
    """Verify format_val handles float('nan') and None values safely."""
    assert format_val(float("nan")) == "N/A"
    assert format_val(None) == "N/A"

