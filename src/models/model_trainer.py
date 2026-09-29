"""
Model Trainer, Preflight Validation & Cross-Validation Engine.

Performs dataset preflight validation, dataset-specific candidate model training,
and 5-fold Stratified Cross-Validation strictly on training data (X_train, y_train).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import clone

from src.models.model_registry import get_candidate_model


def validate_training_data_preflight(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame = None,
    y_test: pd.Series = None,
    cv_folds: int = 5
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate dataset eligibility and trainability prior to candidate model training.

    Parameters:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target labels.
        X_test (pd.DataFrame, optional): Test feature matrix.
        y_test (pd.Series, optional): Test target labels.
        cv_folds (int): Configured CV fold count.

    Returns:
        (bool, dict): (is_valid, validation_report_dict)
    """
    train_rows = len(y_train) if y_train is not None else 0
    test_rows = len(y_test) if y_test is not None else 0
    actual_rows = train_rows + test_rows

    target_classes = y_train.nunique() if y_train is not None else 0

    min_test_rows = 1
    min_target_classes = 2

    # StratifiedKFold requires at least cv_folds samples per class in training set
    min_samples_per_class = max(1, cv_folds)
    min_train_rows = min_target_classes * min_samples_per_class
    required_minimum = min_train_rows + min_test_rows

    if actual_rows < 2 or train_rows < 1:
        return False, {
            "status": "DATASET_TOO_SMALL_FOR_TRAINING",
            "message": f"Dataset has too few total samples ({actual_rows} total, {train_rows} train, {test_rows} test). Minimum required: {required_minimum}.",
            "required_minimum": required_minimum,
            "actual_rows": actual_rows,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "cv_folds": cv_folds,
            "target_classes_found": int(target_classes)
        }

    if target_classes < min_target_classes:
        return False, {
            "status": "DATASET_TOO_SMALL_FOR_TRAINING",
            "message": f"Training target column contains only {target_classes} class. Binary classification requires at least 2 distinct target classes.",
            "required_minimum": required_minimum,
            "actual_rows": actual_rows,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "cv_folds": cv_folds,
            "target_classes_found": int(target_classes)
        }

    # Check sample count for each class in y_train
    class_counts = y_train.value_counts().to_dict()
    for cls_val, cnt in class_counts.items():
        if cnt < cv_folds:
            return False, {
                "status": "DATASET_TOO_SMALL_FOR_TRAINING",
                "message": f"Class '{cls_val}' has only {cnt} training sample(s), but at least {cv_folds} samples per class are required for {cv_folds}-fold Stratified Cross-Validation.",
                "required_minimum": required_minimum,
                "actual_rows": actual_rows,
                "train_rows": train_rows,
                "test_rows": test_rows,
                "cv_folds": cv_folds,
                "target_classes_found": int(target_classes)
            }

    if train_rows < min_train_rows:
        return False, {
            "status": "DATASET_TOO_SMALL_FOR_TRAINING",
            "message": f"Training sample count ({train_rows}) is insufficient for {cv_folds}-fold CV. Required minimum: {min_train_rows} training samples.",
            "required_minimum": required_minimum,
            "actual_rows": actual_rows,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "cv_folds": cv_folds,
            "target_classes_found": int(target_classes)
        }

    return True, {
        "status": "VALID",
        "message": "Dataset preflight validation passed.",
        "required_minimum": required_minimum,
        "actual_rows": actual_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "cv_folds": cv_folds,
        "target_classes_found": int(target_classes)
    }


def train_and_cross_validate_candidate(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Perform Stratified K-Fold Cross-Validation on training data X_train, y_train and fit final model on X_train.

    Parameters:
        model_key (str): Candidate model key.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target labels.
        cv_folds (int): Number of CV folds (default: 5).
        random_seed (int): Random seed.

    Returns:
        dict: Candidate model result containing fitted model object (or None if failed), is_fitted boolean, and CV metrics.
    """
    base_model = get_candidate_model(model_key, random_seed=random_seed)
    n_samples = len(y_train)

    if n_samples < cv_folds or y_train.nunique() < 2:
        fitted = None
        try:
            base_model.fit(X_train, y_train)
            fitted = base_model
        except Exception:
            fitted = None

        return {
            "model_key": model_key,
            "fitted_model": fitted,
            "is_fitted": (fitted is not None),
            "failure_reason": None if fitted is not None else f"Dataset has insufficient samples ({n_samples}) or classes ({y_train.nunique()}) for cross-validation.",
            "cv_metrics": {
                "cv_accuracy_mean": 0.0,
                "cv_accuracy_std": 0.0,
                "cv_precision_mean": 0.0,
                "cv_recall_mean": 0.0,
                "cv_f1_mean": 0.0,
                "cv_roc_auc_mean": None,
                "folds_completed": 0
            }
        }

    try:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        accuracies, precisions, recalls, f1s, roc_aucs = [], [], [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            fold_model = clone(base_model)
            fold_model.fit(X_tr, y_tr)

            y_val_pred = fold_model.predict(X_val)

            acc = accuracy_score(y_val, y_val_pred)
            prec = precision_score(y_val, y_val_pred, zero_division=0)
            rec = recall_score(y_val, y_val_pred, zero_division=0)
            f1 = f1_score(y_val, y_val_pred, zero_division=0)

            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            if hasattr(fold_model, "predict_proba") and y_val.nunique() == 2:
                try:
                    y_val_prob = fold_model.predict_proba(X_val)[:, 1]
                    auc = roc_auc_score(y_val, y_val_prob)
                    roc_aucs.append(auc)
                except Exception:
                    pass

        # Fit final model on full training set
        final_model = clone(base_model)
        final_model.fit(X_train, y_train)

        cv_results = {
            "cv_accuracy_mean": float(np.mean(accuracies)),
            "cv_accuracy_std": float(np.std(accuracies)),
            "cv_precision_mean": float(np.mean(precisions)),
            "cv_recall_mean": float(np.mean(recalls)),
            "cv_f1_mean": float(np.mean(f1s)),
            "cv_roc_auc_mean": float(np.mean(roc_aucs)) if roc_aucs else None,
            "folds_completed": len(accuracies)
        }

        return {
            "model_key": model_key,
            "fitted_model": final_model,
            "is_fitted": True,
            "failure_reason": None,
            "cv_metrics": cv_results
        }
    except Exception as exc:
        return {
            "model_key": model_key,
            "fitted_model": None,
            "is_fitted": False,
            "failure_reason": f"Training exception: {str(exc)}",
            "cv_metrics": {
                "cv_accuracy_mean": 0.0,
                "cv_accuracy_std": 0.0,
                "cv_precision_mean": 0.0,
                "cv_recall_mean": 0.0,
                "cv_f1_mean": 0.0,
                "cv_roc_auc_mean": None,
                "folds_completed": 0
            }
        }
