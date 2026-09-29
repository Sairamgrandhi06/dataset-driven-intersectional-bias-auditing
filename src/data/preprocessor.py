"""
Data Preprocessing Module for Intersectional Bias Auditing.

Handles:
1. Missing value cleanup ('?', NaN, nulls).
2. Generalized binary target variable encoding with multiclass validation.
3. Decoupling model features (X), binary target outcome (y), and protected attributes (A).
4. Feature matrix preprocessing (excluding target, protected attributes, and ignored/ID columns).
5. Reproducible train/test splitting maintaining strict index alignment across X, y, and A.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Union, Optional
from sklearn.model_selection import train_test_split

from src.data.loader import load_config
from src.data.dataset_config import DatasetConfig


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace missing value markers ('?', 'NaN', 'null', etc.) with NaN and drop incomplete rows.

    Parameters:
        df (pd.DataFrame): Input raw dataframe.

    Returns:
        pd.DataFrame: Dataframe with missing values dropped.
    """
    df_cleaned = df.copy()
    # Replace common missing string markers with NaN
    df_cleaned.replace(["?", "NaN", "nan", "null", "NULL", "None"], np.nan, inplace=True)

    initial_len = len(df_cleaned)
    df_cleaned.dropna(inplace=True)
    dropped_count = initial_len - len(df_cleaned)

    if dropped_count > 0:
        print(f"Preprocessor: Dropped {dropped_count} rows containing missing values.")

    return df_cleaned


def encode_target(
    series: pd.Series,
    positive_value: Any = ">50K",
    negative_value: Optional[Any] = None
) -> pd.Series:
    """
    Encode categorical/numerical target variable into binary integers (1 for positive class, 0 otherwise).
    Validates that the target contains at most 2 unique non-null classes.

    Parameters:
        series (pd.Series): Target column values.
        positive_value (Any): Value representing the positive outcome (e.g. '>50K', 1, 'Yes', 'Approved').
        negative_value (Any, optional): Explicit negative outcome value.

    Returns:
        pd.Series: Binary integer target series (0 or 1).
    """
    clean_series = series.astype(str).str.strip().str.rstrip(".")
    clean_pos = str(positive_value).strip().rstrip(".")

    unique_vals = clean_series.unique()
    if len(unique_vals) > 2:
        raise ValueError(
            f"Multiclass target not supported! Target column '{series.name}' contains "
            f"{len(unique_vals)} unique classes: {list(unique_vals)}. Only binary classification targets are supported."
        )
    if len(unique_vals) == 0:
        raise ValueError(f"Target column '{series.name}' contains no non-null values.")

    # Match positive class value (case-insensitive for string targets)
    clean_series_lower = clean_series.str.lower()
    clean_pos_lower = clean_pos.lower()
    binary_y = (clean_series_lower == clean_pos_lower).astype(int)
    binary_y.name = series.name
    return binary_y


def prepare_pipeline_data(
    df: pd.DataFrame,
    config: Optional[Union[DatasetConfig, Dict[str, Any]]] = None,
    config_path: str = "config/default_config.json"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Process raw dataframe into decoupled features X, target y, and protected attributes A.

    Parameters:
        df (pd.DataFrame): Raw clean DataFrame.
        config (DatasetConfig or dict, optional): Configuration settings.
        config_path (str, optional): Path to configuration file.

    Returns:
        tuple: (X_encoded, y, A) where:
            - X_encoded (pd.DataFrame): One-hot encoded feature matrix.
            - y (pd.Series): Binary target labels (0 or 1).
            - A (pd.DataFrame): Raw protected attributes preserved for auditing.
    """
    if config is None:
        raw_cfg = load_config(config_path)
        cfg = DatasetConfig.from_dict(raw_cfg)
    elif isinstance(config, dict):
        cfg = DatasetConfig.from_dict(config)
    else:
        cfg = config

    target_col = cfg.target.column
    protected_cols = cfg.protected_attributes
    drop_protected = cfg.drop_protected_from_features
    positive_val = cfg.target.positive_class
    ignore_cols = set(cfg.ignore_columns + cfg.id_columns)

    # 1. Clean missing values
    df_clean = clean_missing_values(df)

    # 2. Extract target variable y
    y = encode_target(df_clean[target_col], positive_value=positive_val)
    y.name = target_col

    # 3. Extract protected attributes dataframe A
    A = df_clean[protected_cols].copy()

    # 4. Prepare feature dataframe X
    cols_to_drop = [target_col]
    if drop_protected:
        cols_to_drop.extend(protected_cols)
    for ic in ignore_cols:
        if ic in df_clean.columns and ic not in cols_to_drop:
            cols_to_drop.append(ic)

    X_raw = df_clean.drop(columns=cols_to_drop, errors="ignore").copy()

    # 5. One-hot encode categorical features in X
    if X_raw.shape[1] == 0:
        X_encoded = X_raw.copy()
    else:
        X_encoded = pd.get_dummies(X_raw, drop_first=True)

    return X_encoded, y, A


def split_pipeline_data(
    X: pd.DataFrame,
    y: pd.Series,
    A: pd.DataFrame,
    config: Optional[Union[DatasetConfig, Dict[str, Any]]] = None,
    config_path: str = "config/default_config.json"
) -> Dict[str, Any]:
    """
    Perform reproducible stratified train-test split for features (X), target (y),
    and protected attributes (A) preserving row alignment.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target series.
        A (pd.DataFrame): Protected attributes dataframe.
        config (DatasetConfig or dict, optional): Configuration settings.
        config_path (str, optional): Configuration file path.

    Returns:
        dict: Dictionary containing train/test splits:
            - 'X_train', 'X_test', 'y_train', 'y_test', 'A_train', 'A_test'
    """
    if config is None:
        raw_cfg = load_config(config_path)
        cfg = DatasetConfig.from_dict(raw_cfg)
    elif isinstance(config, dict):
        cfg = DatasetConfig.from_dict(config)
    else:
        cfg = config

    test_size = cfg.test_size
    random_seed = cfg.random_state

    # Check if stratification is possible
    stratify_param = y
    if y is not None and len(y) > 0:
        counts = y.value_counts()
        n_classes = len(counts)
        calc_test_count = int(np.ceil(len(y) * test_size)) if isinstance(test_size, float) else test_size
        if (counts.min() < 2) or (calc_test_count < n_classes):
            stratify_param = None

    # Split indices first to ensure absolute alignment across X, y, and A
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_param
    )

    splits = {
        "X_train": X.iloc[idx_train].copy(),
        "X_test": X.iloc[idx_test].copy(),
        "y_train": y.iloc[idx_train].copy(),
        "y_test": y.iloc[idx_test].copy(),
        "A_train": A.iloc[idx_train].copy(),
        "A_test": A.iloc[idx_test].copy()
    }

    return splits
