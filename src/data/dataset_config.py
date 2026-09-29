"""
Dataset Configuration Contract Module.

Provides structured parsing, validation, and serialization for tabular dataset configurations
in the Intersectional Bias Auditing framework.
"""

import os
import re
import json
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional, Tuple


@dataclass
class DatasetTargetConfig:
    column: str
    positive_class: Any
    negative_class: Optional[Any] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetTargetConfig":
        if "column" not in data or "positive_class" not in data:
            raise ValueError("Target config requires 'column' and 'positive_class'.")
        return cls(
            column=str(data["column"]),
            positive_class=data["positive_class"],
            negative_class=data.get("negative_class")
        )

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "column": self.column,
            "positive_class": self.positive_class
        }
        if self.negative_class is not None:
            res["negative_class"] = self.negative_class
        return res


@dataclass
class IntersectionalConfig:
    min_group_size: int = 30

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "IntersectionalConfig":
        if not data:
            return cls()
        return cls(min_group_size=int(data.get("min_group_size", 30)))

    def to_dict(self) -> Dict[str, Any]:
        return {"min_group_size": self.min_group_size}


@dataclass
class ModelConfig:
    algorithm: str = "LogisticRegression"
    max_iter: int = 1000
    solver: str = "lbfgs"
    random_state: int = 42

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], global_seed: int = 42) -> "ModelConfig":
        if not data:
            return cls(random_state=global_seed)
        return cls(
            algorithm=str(data.get("algorithm", "LogisticRegression")),
            max_iter=int(data.get("max_iter", 1000)),
            solver=str(data.get("solver", "lbfgs")),
            random_state=int(data.get("random_state", global_seed))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "max_iter": self.max_iter,
            "solver": self.solver,
            "random_state": self.random_state
        }


@dataclass
class DatasetConfig:
    dataset_id: str
    path: str
    target: DatasetTargetConfig
    protected_attributes: List[str]
    ignore_columns: List[str] = field(default_factory=list)
    id_columns: List[str] = field(default_factory=list)
    test_size: float = 0.2
    random_state: int = 42
    intersectional: IntersectionalConfig = field(default_factory=IntersectionalConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    fairness_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "demographic_parity_difference_max": 0.10,
        "disparate_impact_min": 0.80,
        "equal_opportunity_difference_max": 0.10,
        "equalized_odds_difference_max": 0.10,
        "fpr_difference_max": 0.10
    })
    calibration_settings: Dict[str, Any] = field(default_factory=lambda: {"n_bins": 10})
    drop_protected_from_features: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        """Parse DatasetConfig from dictionary or legacy config dictionary."""
        # Support legacy config structures (default_config.json)
        if "dataset" in data and isinstance(data["dataset"], dict):
            ds = data["dataset"]
            target_config = DatasetTargetConfig(
                column=ds.get("target_column", "income"),
                positive_class=ds.get("positive_target_value", ">50K")
            )
            dataset_id = str(ds.get("name", "adult_income")).lower().replace(" ", "_")
            path = str(ds.get("raw_path", "data/raw/adult.csv"))
            protected_attrs = list(ds.get("protected_attributes", ["sex", "race"]))
            drop_protected = bool(ds.get("drop_protected_from_features", True))
            test_size = float(data.get("test_size", 0.2))
            random_seed = int(data.get("random_seed", 42))

            return cls(
                dataset_id=dataset_id,
                path=path,
                target=target_config,
                protected_attributes=protected_attrs,
                test_size=test_size,
                random_state=random_seed,
                drop_protected_from_features=drop_protected,
                model_config=ModelConfig(random_state=random_seed)
            )

        if "dataset_id" not in data or "path" not in data or "target" not in data or "protected_attributes" not in data:
            raise ValueError("DatasetConfig requires 'dataset_id', 'path', 'target', and 'protected_attributes'.")

        target_obj = DatasetTargetConfig.from_dict(data["target"])
        protected_attrs = [str(a) for a in data["protected_attributes"]]
        if not protected_attrs:
            raise ValueError("At least one protected attribute must be specified.")

        random_state = int(data.get("random_state", 42))
        return cls(
            dataset_id=str(data["dataset_id"]),
            path=str(data["path"]),
            target=target_obj,
            protected_attributes=protected_attrs,
            ignore_columns=[str(c) for c in data.get("ignore_columns", [])],
            id_columns=[str(c) for c in data.get("id_columns", [])],
            test_size=float(data.get("test_size", 0.2)),
            random_state=random_state,
            intersectional=IntersectionalConfig.from_dict(data.get("intersectional")),
            model_config=ModelConfig.from_dict(data.get("model_config"), global_seed=random_state),
            fairness_thresholds=data.get("fairness_thresholds", {
                "demographic_parity_difference_max": 0.10,
                "disparate_impact_min": 0.80,
                "equal_opportunity_difference_max": 0.10,
                "equalized_odds_difference_max": 0.10,
                "fpr_difference_max": 0.10
            }),
            calibration_settings=data.get("calibration_settings", {"n_bins": 10}),
            drop_protected_from_features=bool(data.get("drop_protected_from_features", True))
        )

    @classmethod
    def from_json(cls, json_path: str) -> "DatasetConfig":
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Dataset configuration file not found at: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "path": self.path,
            "target": self.target.to_dict(),
            "protected_attributes": self.protected_attributes,
            "ignore_columns": self.ignore_columns,
            "id_columns": self.id_columns,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "intersectional": self.intersectional.to_dict(),
            "model_config": self.model_config.to_dict(),
            "fairness_thresholds": self.fairness_thresholds,
            "calibration_settings": self.calibration_settings,
            "drop_protected_from_features": self.drop_protected_from_features
        }


def load_dataset_config_by_id(dataset_id: str, base_dir: str = ".") -> DatasetConfig:
    """
    Locate and load DatasetConfig object by dataset_id with strict dataset isolation.

    Parameters:
        dataset_id (str): Target dataset identifier.
        base_dir (str): Base workspace directory.

    Returns:
        DatasetConfig: Loaded dataset configuration instance.
    """
    clean_id = dataset_id.lower().strip()
    config_dir = os.path.join(base_dir, "config")

    # 1. Candidate file paths based on naming conventions
    candidate_filenames = [
        f"{clean_id}_config.json",
        f"{clean_id}.json",
    ]
    if clean_id in ["loan_approval", "loan", "loan_approval_dataset"]:
        candidate_filenames.extend(["loan_config.json", "loan_approval_config.json"])
    elif clean_id in ["student_performance", "student", "student_performance_dataset"]:
        candidate_filenames.extend(["student_config.json", "student_performance_config.json"])
    elif clean_id in ["adult_census_income", "adult_income", "adult"]:
        candidate_filenames.extend(["default_config.json", "adult_config.json", "adult_census_income_config.json"])

    for c_name in candidate_filenames:
        c_path = os.path.join(config_dir, c_name)
        if os.path.exists(c_path):
            try:
                cfg = DatasetConfig.from_json(c_path)
                if cfg.dataset_id.lower().strip() == clean_id or clean_id in cfg.dataset_id.lower():
                    return cfg
            except Exception:
                pass

    # 2. Search config directory for any JSON whose dataset_id field matches clean_id
    if os.path.exists(config_dir):
        for f_name in sorted(os.listdir(config_dir)):
            if f_name.endswith(".json"):
                f_path = os.path.join(config_dir, f_name)
                try:
                    with open(f_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and str(data.get("dataset_id", "")).lower().strip() == clean_id:
                        return DatasetConfig.from_json(f_path)
                except Exception:
                    pass

    # 3. Adult reference dataset fallback
    if clean_id in ["adult_census_income", "adult_income", "adult"]:
        def_path = os.path.join(config_dir, "default_config.json")
        if os.path.exists(def_path):
            return DatasetConfig.from_json(def_path)

    # 4. Check for data/raw/{clean_id}.csv
    raw_dir = os.path.join(base_dir, "data", "raw")
    raw_candidates = [
        os.path.join(raw_dir, f"{clean_id}.csv"),
        os.path.join(raw_dir, f"{clean_id.split('_')[0]}.csv")
    ]
    for r_path in raw_candidates:
        if os.path.exists(r_path):
            return DatasetConfig(
                dataset_id=clean_id,
                path=r_path,
                target=DatasetTargetConfig(column="target", positive_class=1),
                protected_attributes=[]
            )

    # Generic fallback - STRICT: never uses default_config.json for non-adult
    return DatasetConfig(
        dataset_id=clean_id,
        path="",
        target=DatasetTargetConfig(column="target", positive_class=1),
        protected_attributes=["protected_attr"]
    )


def validate_dataset_configuration_contract(
    df: pd.DataFrame,
    config_dict: Dict[str, Any]
) -> Tuple[bool, List[str], List[str]]:
    """
    Perform pre-training configuration validation against the raw dataset DataFrame.

    Validates:
    - Target column exists, is not an ID, is not protected, has at least 2 distinct classes.
    - Positive class is present in target column classes.
    - Protected attributes exist, are not empty, and do not overlap with ID columns.
    - ID columns exist in DataFrame.
    - Warns if numeric predictive features are placed in ID/Ignore.
    - At least one predictive feature remains after excluding target, protected, and ID columns.

    Returns:
        tuple: (is_valid: bool, errors: List[str], warnings: List[str])
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        errors.append("CONFIGURATION_INVALID: Dataset DataFrame is empty or None.")
        return False, errors, warnings

    target_info = config_dict.get("target")
    if isinstance(target_info, dict):
        target_col = target_info.get("column")
        pos_class = target_info.get("positive_class")
    else:
        target_col = config_dict.get("target_column") or (str(target_info) if target_info else None)
        pos_class = config_dict.get("positive_class")

    protected_attrs = list(config_dict.get("protected_attributes") or [])
    id_cols = list(config_dict.get("id_columns") or [])
    all_cols = list(df.columns)

    # 1. Target Column Validations
    if not target_col:
        errors.append("CONFIGURATION_INVALID: Target column must be specified.")
    elif target_col not in all_cols:
        errors.append(f"CONFIGURATION_INVALID: Target column '{target_col}' was not found in dataset columns.")
    else:
        if target_col in id_cols:
            errors.append(f"CONFIGURATION_INVALID: Target column '{target_col}' cannot simultaneously be configured as an ID/Ignore column.")
        if target_col in protected_attrs:
            errors.append(f"CONFIGURATION_INVALID: Target column '{target_col}' cannot simultaneously be configured as a protected demographic attribute.")

        valid_targets = df[target_col].dropna().unique()
        if len(valid_targets) < 2 and len(df) >= 10:
            errors.append(f"CONFIGURATION_INVALID: Target column '{target_col}' contains only {len(valid_targets)} unique value(s). Binary classification requires at least 2 distinct classes.")

        if pos_class is not None and len(valid_targets) >= 2:
            pos_str = str(pos_class).strip().lower()
            unique_strs = [str(u).strip().lower() for u in valid_targets]
            if pos_str not in unique_strs:
                errors.append(f"CONFIGURATION_INVALID: Specified positive class value '{pos_class}' does not exist in target column '{target_col}'. Available values: {list(valid_targets)}")

    # 2. Protected Attributes Validations
    if not protected_attrs:
        errors.append("CONFIGURATION_INVALID: At least one protected demographic attribute must be selected.")
    else:
        for p in protected_attrs:
            if p not in all_cols:
                errors.append(f"CONFIGURATION_INVALID: Protected attribute '{p}' was not found in dataset columns.")
            elif p in id_cols:
                errors.append(f"CONFIGURATION_INVALID: Protected attribute '{p}' cannot simultaneously be configured as an ID/Ignore column.")

    # 3. ID / Ignore Column Validations & Suspicious Predictive Feature Warnings
    for c in id_cols:
        if c not in all_cols:
            errors.append(f"CONFIGURATION_INVALID: ID/Ignore column '{c}' was not found in dataset columns.")
        else:
            col_series = df[c]
            if pd.api.types.is_numeric_dtype(col_series):
                c_clean = str(c).lower().strip()
                tokens = set(re.split(r'[^a-z0-9]+', c_clean))
                is_named_id = bool(
                    (tokens & {"id", "uuid", "guid", "ssn", "identifier", "pk", "uid"})
                    or c_clean.endswith("_id")
                    or c_clean.startswith("id_")
                    or (c_clean.endswith("id") and len(c_clean) > 2 and c_clean not in ["paid", "unpaid", "grid", "solid", "liquid", "valid", "invalid", "hybrid", "pyramid", "acid", "lipid", "mid"])
                    or c_clean in ["id", "uuid", "guid", "ssn", "identifier", "pk", "uid"]
                )
                if not is_named_id:
                    warnings.append(f"'{c}' appears to be a numeric predictive feature rather than an identifier. Review the ID/Ignore selection before training.")

    # 4. Feature Matrix Check (Remaining features)
    if not errors:
        excluded_cols = set([target_col] + id_cols + protected_attrs)
        remaining_features = [c for c in all_cols if c not in excluded_cols]
        if len(remaining_features) == 0:
            errors.append("CONFIGURATION_INVALID: No predictive features remain after excluding target, protected attributes, and ID/Ignore columns. At least one feature is required.")

    is_valid = (len(errors) == 0)
    return is_valid, errors, warnings


