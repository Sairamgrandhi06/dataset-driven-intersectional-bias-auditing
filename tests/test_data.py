"""
Unit tests for Data Loading and Preprocessing Modules.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.loader import validate_schema, load_config
from src.data.preprocessor import (
    clean_missing_values,
    encode_target,
    prepare_pipeline_data,
    split_pipeline_data
)

@pytest.fixture
def sample_raw_dataframe():
    """Fixture providing a mock synthetic dataframe adhering to the Adult dataset schema."""
    data = {
        "age": [25, 50, 38, 42, 28, 35],
        "workclass": ["Private", "Self-emp", "Private", "Private", "Private", "?"],
        "fnlwgt": [226802, 89814, 215646, 154374, 338409, 120000],
        "education": ["11th", "Masters", "HS-grad", "Some-college", "Bachelors", "10th"],
        "education_num": [7, 14, 9, 10, 13, 6],
        "marital_status": ["Never-married", "Married-civ-spouse", "Divorced", "Never-married", "Bachelors", "Never-married"],
        "occupation": ["Machine-op-inspct", "Exec-managerial", "Handlers-cleaners", "Prof-specialty", "Sales", "?"],
        "relationship": ["Own-child", "Husband", "Not-in-family", "Unmarried", "Wife", "Own-child"],
        "race": ["Black", "White", "White", "Black", "Black", "White"],
        "sex": ["Male", "Male", "Male", "Female", "Female", "Female"],
        "capital_gain": [0, 0, 0, 0, 0, 0],
        "capital_loss": [0, 0, 0, 0, 0, 0],
        "hours_per_week": [40, 50, 40, 40, 40, 20],
        "native_country": ["United-States", "United-States", "United-States", "United-States", "United-States", "United-States"],
        "income": ["<=50K", ">50K", "<=50K", ">50K", "<=50K", "<=50K"]
    }
    return pd.DataFrame(data)

def test_validate_schema_success(sample_raw_dataframe):
    """Verify schema validation passes when all required columns are present."""
    expected_cols = list(sample_raw_dataframe.columns)
    assert validate_schema(sample_raw_dataframe, expected_cols) is True

def test_validate_schema_missing_column(sample_raw_dataframe):
    """Verify schema validation raises ValueError if a column is missing."""
    df_missing = sample_raw_dataframe.drop(columns=["income"])
    expected_cols = list(sample_raw_dataframe.columns)
    with pytest.raises(ValueError, match="Schema Validation Error"):
        validate_schema(df_missing, expected_cols)

def test_clean_missing_values(sample_raw_dataframe):
    """Verify that rows with '?' missing value markers are dropped."""
    df_cleaned = clean_missing_values(sample_raw_dataframe)
    # The last row has '?' in workclass and occupation
    assert len(df_cleaned) == 5
    assert "?" not in df_cleaned.values

def test_encode_target():
    """Verify binary target encoding produces 1 for >50K and 0 for <=50K."""
    series = pd.Series(["<=50K", ">50K", "<=50K.", ">50K."])
    encoded = encode_target(series, positive_value=">50K")
    expected = pd.Series([0, 1, 0, 1], name=series.name)
    pd.testing.assert_series_equal(encoded, expected)

def test_prepare_pipeline_data_decoupling(sample_raw_dataframe):
    """Verify decoupling of features X, target y, and protected attributes A."""
    config = {
        "dataset": {
            "target_column": "income",
            "positive_target_value": ">50K",
            "protected_attributes": ["sex", "race"],
            "drop_protected_from_features": True
        }
    }
    
    X, y, A = prepare_pipeline_data(sample_raw_dataframe, config=config)
    
    # 1. Row count match
    assert len(X) == len(y) == len(A) == 5
    
    # 2. Target binary
    assert set(y.unique()).issubset({0, 1})
    
    # 3. Protected attributes preserved in A
    assert list(A.columns) == ["sex", "race"]
    
    # 4. Protected attributes removed from feature matrix X
    assert "sex" not in X.columns
    assert "race" not in X.columns
    assert "sex_Male" not in X.columns
    assert "race_White" not in X.columns

def test_split_pipeline_data(sample_raw_dataframe):
    """Verify train/test splitting maintains proper row alignment across X, y, and A."""
    config = {
        "dataset": {
            "target_column": "income",
            "positive_target_value": ">50K",
            "protected_attributes": ["sex", "race"],
            "drop_protected_from_features": True
        },
        "test_size": 0.2,
        "random_seed": 42
    }
    
    X, y, A = prepare_pipeline_data(sample_raw_dataframe, config=config)
    splits = split_pipeline_data(X, y, A, config=config)
    
    # Verify split keys exist
    for key in ["X_train", "X_test", "y_train", "y_test", "A_train", "A_test"]:
        assert key in splits
        
    # Verify shapes and index alignment
    assert len(splits["X_train"]) == len(splits["y_train"]) == len(splits["A_train"])
    assert len(splits["X_test"]) == len(splits["y_test"]) == len(splits["A_test"])
    
    # Check index equality between features and protected attributes in test split
    pd.testing.assert_index_equal(splits["X_test"].index, splits["A_test"].index)
    pd.testing.assert_index_equal(splits["y_test"].index, splits["A_test"].index)
