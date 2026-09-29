"""
Unit & Integration Tests for Dataset Generalization Engine.

Verifies that the core data, model, and auditing pipeline operates on arbitrary
tabular CSV datasets without dataset-specific assumptions or code alterations.
"""

import os
import json
import tempfile
import pytest
import numpy as np
import pandas as pd

from src.data.dataset_config import DatasetConfig
from src.data.loader import load_raw_dataset_from_config, compute_file_sha256
from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data, encode_target
from src.fairness.single_attribute import run_single_attribute_audits
from src.fairness.intersectional import create_intersectional_attribute, audit_intersectional_attributes
from src.models.classifier import train_baseline_model, predict_model, evaluate_performance_metrics
from src.mitigation.mitigator import train_mitigated_model, predict_mitigated_model
from src.run_dataset import run_dataset_pipeline
from reference.expected_results import is_reference_dataset


@pytest.fixture
def synthetic_custom_csv(tmp_path):
    """Fixture generating a synthetic CSV dataset with custom column names."""
    df = pd.DataFrame({
        "applicant_id": [f"ID_{i}" for i in range(200)],
        "age": np.random.randint(18, 70, size=200),
        "score": np.random.normal(600, 50, size=200),
        "gender": np.random.choice(["Female", "Male"], size=200),
        "region": np.random.choice(["North", "South", "East"], size=200),
        "approved": np.random.choice(["Yes", "No"], size=200, p=[0.4, 0.6])
    })
    file_path = tmp_path / "custom_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def synthetic_3way_csv(tmp_path):
    """Fixture generating a dataset with 3 protected attributes."""
    df = pd.DataFrame({
        "feature1": np.random.randn(300),
        "feature2": np.random.randn(300),
        "gender": np.random.choice(["Female", "Male"], size=300),
        "race": np.random.choice(["GroupA", "GroupB"], size=300),
        "age_group": np.random.choice(["Young", "Old"], size=300),
        "target_col": np.random.choice([0, 1], size=300)
    })
    file_path = tmp_path / "data_3way.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_custom_target_column(synthetic_custom_csv, tmp_path):
    """Verify loading and processing a dataset with custom target column ('approved') and positive class ('Yes')."""
    cfg_data = {
        "dataset_id": "custom_credit",
        "path": synthetic_custom_csv,
        "target": {
            "column": "approved",
            "positive_class": "Yes"
        },
        "protected_attributes": ["gender"],
        "id_columns": ["applicant_id"]
    }
    cfg = DatasetConfig.from_dict(cfg_data)

    df, meta = load_raw_dataset_from_config(cfg)
    assert meta["target_column"] == "approved"

    X, y, A = prepare_pipeline_data(df, config=cfg)
    assert y.name == "approved"
    assert set(y.unique()).issubset({0, 1})
    assert "applicant_id" not in X.columns
    assert "gender" not in X.columns


def test_custom_positive_class():
    """Verify target encoding handles various positive class types (strings, ints, bools)."""
    s_str = pd.Series(["Approved", "Rejected", "Approved", "Rejected"])
    y_encoded = encode_target(s_str, positive_value="Approved")
    assert list(y_encoded) == [1, 0, 1, 0]

    s_int = pd.Series([1, 0, 1, 0])
    y_encoded_int = encode_target(s_int, positive_value=1)
    assert list(y_encoded_int) == [1, 0, 1, 0]


def test_custom_protected_attributes(synthetic_custom_csv):
    """Verify protected attributes are selected dynamically from configuration."""
    cfg_data = {
        "dataset_id": "custom_credit",
        "path": synthetic_custom_csv,
        "target": {"column": "approved", "positive_class": "Yes"},
        "protected_attributes": ["gender", "region"]
    }
    cfg = DatasetConfig.from_dict(cfg_data)
    df, _ = load_raw_dataset_from_config(cfg)
    X, y, A = prepare_pipeline_data(df, config=cfg)

    assert list(A.columns) == ["gender", "region"]
    assert "gender" not in X.columns
    assert "region" not in X.columns


def test_three_way_intersection(synthetic_3way_csv):
    """Verify intersectional auditing supports 3 protected attributes dynamically."""
    cfg_data = {
        "dataset_id": "three_way",
        "path": synthetic_3way_csv,
        "target": {"column": "target_col", "positive_class": 1},
        "protected_attributes": ["gender", "race", "age_group"]
    }
    cfg = DatasetConfig.from_dict(cfg_data)
    df, _ = load_raw_dataset_from_config(cfg)
    X, y, A = prepare_pipeline_data(df, config=cfg)

    inter_series = create_intersectional_attribute(A, attributes=cfg.protected_attributes)
    assert inter_series.name == "gender + race + age_group"
    # Should have up to 2*2*2 = 8 subgroups
    assert len(inter_series.unique()) <= 8

    y_pred = np.random.choice([0, 1], size=len(y))
    audit = audit_intersectional_attributes(y, y_pred, A, attributes=cfg.protected_attributes, min_group_size=10)
    assert audit["intersection_definition"] == "gender x race x age_group"


def test_missing_protected_attribute(synthetic_custom_csv):
    """Verify error is raised if configured protected attribute does not exist in dataset."""
    cfg_data = {
        "dataset_id": "missing_attr",
        "path": synthetic_custom_csv,
        "target": {"column": "approved", "positive_class": "Yes"},
        "protected_attributes": ["non_existent_attribute"]
    }
    cfg = DatasetConfig.from_dict(cfg_data)
    with pytest.raises(ValueError, match="not found in dataset"):
        load_raw_dataset_from_config(cfg)


def test_invalid_target(synthetic_custom_csv):
    """Verify error is raised if configured target column does not exist in dataset."""
    cfg_data = {
        "dataset_id": "missing_target",
        "path": synthetic_custom_csv,
        "target": {"column": "non_existent_target", "positive_class": 1},
        "protected_attributes": ["gender"]
    }
    cfg = DatasetConfig.from_dict(cfg_data)
    with pytest.raises(ValueError, match="Target column 'non_existent_target' not found"):
        load_raw_dataset_from_config(cfg)


def test_multiclass_rejection(tmp_path):
    """Verify explicit ValueError is raised when target contains > 2 unique classes."""
    df_multi = pd.DataFrame({
        "feat": [1, 2, 3, 4, 5, 6],
        "gender": ["F", "M", "F", "M", "F", "M"],
        "class3": ["Low", "Medium", "High", "Low", "Medium", "High"]
    })
    path = str(tmp_path / "multi_class.csv")
    df_multi.to_csv(path, index=False)

    cfg_data = {
        "dataset_id": "multiclass_test",
        "path": path,
        "target": {"column": "class3", "positive_class": "High"},
        "protected_attributes": ["gender"]
    }
    cfg = DatasetConfig.from_dict(cfg_data)
    df, _ = load_raw_dataset_from_config(cfg)

    with pytest.raises(ValueError, match="Multiclass target not supported"):
        prepare_pipeline_data(df, config=cfg)


def test_dataset_hash_changes(tmp_path):
    """Verify compute_file_sha256 produces different hashes for different CSV contents."""
    p1 = tmp_path / "file1.csv"
    p2 = tmp_path / "file2.csv"

    p1.write_text("col1,col2\n1,2\n")
    p2.write_text("col1,col2\n1,3\n")

    h1 = compute_file_sha256(str(p1))
    h2 = compute_file_sha256(str(p2))

    assert len(h1) == 64
    assert len(h2) == 64
    assert h1 != h2


def test_reference_benchmark_only_for_reference_dataset():
    """Verify reference dataset helper correctly scopes Adult benchmark checks."""
    assert is_reference_dataset("adult_income") is True
    assert is_reference_dataset("adult") is True
    assert is_reference_dataset("custom_dataset") is False
    assert is_reference_dataset("compas") is False


def test_results_are_dataset_specific(synthetic_custom_csv, tmp_path):
    """Verify run_dataset_pipeline creates dataset-specific directory and result JSON."""
    config_file = tmp_path / "custom_config.json"
    cfg_data = {
        "dataset_id": "test_ds_unique",
        "path": synthetic_custom_csv,
        "target": {"column": "approved", "positive_class": "Yes"},
        "protected_attributes": ["gender"],
        "intersectional": {"min_group_size": 5}
    }
    config_file.write_text(json.dumps(cfg_data))

    res = run_dataset_pipeline(str(config_file))

    assert res["dataset_id"] == "test_ds_unique"
    assert os.path.exists("results/test_ds_unique/run_results.json")
    assert os.path.exists("evidence/runs/test_ds_unique/run_manifest.json")
