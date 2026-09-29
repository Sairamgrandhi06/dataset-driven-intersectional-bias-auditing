"""
Data Loader Module for Intersectional Bias Auditing.

Handles reading tabular datasets from CSV paths based on DatasetConfig, performing
validation checks (schema, missing values, duplicates, column types), calculating dataset
SHA-256 hashes, and generating comprehensive dataset metadata.
"""

import os
import json
import hashlib
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import requests

from src.data.dataset_config import DatasetConfig


def load_config(config_path="config/default_config.json"):
    """
    Load project configuration settings from a JSON file.

    Parameters:
        config_path (str): Path to configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return ""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_dataset_if_needed(raw_path, url, columns=None):
    """
    Download raw dataset from URL if it does not already exist locally.

    Parameters:
        raw_path (str): Destination path for CSV file.
        url (str): Source URL for dataset.
        columns (list, optional): Header column names for dataset without headers.

    Returns:
        str: Absolute or relative path to raw file.
    """
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    if not os.path.exists(raw_path):
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        temp_file = raw_path + ".tmp"
        with open(temp_file, "wb") as f:
            f.write(response.content)

        if columns:
            df = pd.read_csv(temp_file, header=None, names=columns, skipinitialspace=True)
            df.to_csv(raw_path, index=False)
        else:
            df = pd.read_csv(temp_file)
            df.to_csv(raw_path, index=False)

        if os.path.exists(temp_file):
            os.remove(temp_file)
        print(f"Dataset successfully saved to {raw_path}")
    else:
        print(f"Dataset already exists locally at {raw_path}")

    return raw_path


def validate_schema(df: pd.DataFrame, expected_columns: list) -> bool:
    """Verify that the input DataFrame contains all expected columns."""
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema Validation Error! Missing columns: {missing_cols}")
    return True


def inspect_dataset_metadata(df: pd.DataFrame, config: DatasetConfig) -> Dict[str, Any]:
    """
    Generate dataset metadata summary including shape, missing values, column types,
    and target / protected attribute validation.
    """
    target_col = config.target.column
    protected_cols = config.protected_attributes

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Columns: {list(df.columns)}")

    missing_protected = [c for c in protected_cols if c not in df.columns]
    if missing_protected:
        raise ValueError(f"Protected attributes {missing_protected} not found in dataset. Columns: {list(df.columns)}")

    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category", "string", "bool"]).columns)

    missing_per_col = {col: int(df[col].isna().sum()) for col in df.columns}
    total_missing = sum(missing_per_col.values())
    duplicate_rows = int(df.duplicated().sum())

    target_counts = df[target_col].value_counts().to_dict()
    # Convert keys to str for JSON serialization compatibility
    target_counts_str = {str(k): int(v) for k, v in target_counts.items()}

    return {
        "dataset_id": config.dataset_id,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "target_column": target_col,
        "target_distribution": target_counts_str,
        "protected_attributes": protected_cols,
        "total_missing_values": total_missing,
        "missing_per_column": missing_per_col,
        "duplicate_rows": duplicate_rows
    }


def load_raw_dataset_from_config(config: DatasetConfig, base_dir: str = ".") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load tabular CSV dataset specified by DatasetConfig, strip string whitespace,
    validate columns, and return (df, metadata).
    """
    raw_path = config.path
    if not os.path.exists(raw_path) and not os.path.isabs(raw_path):
        candidate = os.path.join(base_dir, raw_path)
        if os.path.exists(candidate):
            raw_path = candidate

    if not os.path.exists(raw_path):
        # Fallback to downloading if legacy URL is present in fallback config
        if hasattr(config, "url") and getattr(config, "url"):
            download_dataset_if_needed(raw_path, getattr(config, "url"))
        else:
            raise FileNotFoundError(f"Dataset CSV file not found at: '{raw_path}'")

    df = pd.read_csv(raw_path)

    # Clean whitespace from all string/object columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    metadata = inspect_dataset_metadata(df, config)
    metadata["file_sha256"] = compute_file_sha256(raw_path)

    return df, metadata


def load_raw_data(config_path="config/default_config.json") -> pd.DataFrame:
    """
    Legacy wrapper: Load raw dataset as pandas DataFrame based on configuration settings.

    Parameters:
        config_path (str): Path to project configuration JSON.

    Returns:
        pd.DataFrame: Cleaned raw dataframe with stripped string columns.
    """
    config_dict = load_config(config_path)
    cfg = DatasetConfig.from_dict(config_dict)

    ds_config = config_dict.get("dataset", {})
    raw_path = ds_config.get("raw_path", cfg.path)
    url = ds_config.get("url")
    columns = ds_config.get("columns")

    if url and columns and not os.path.exists(raw_path):
        download_dataset_if_needed(raw_path, url, columns)

    df, _ = load_raw_dataset_from_config(cfg)
    return df
