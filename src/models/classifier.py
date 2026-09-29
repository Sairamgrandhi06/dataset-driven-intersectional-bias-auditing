"""
Classification Model Training and Evaluation Module for Intersectional Bias Auditing.

This module provides functions to:
1. Train a baseline Logistic Regression model using training features (X_train) and labels (y_train).
2. Generate binary predictions and positive class probabilities on test features (X_test).
3. Compute standard classification performance metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def train_baseline_model(X_train, y_train, random_state=42, max_iter=1000, solver="lbfgs"):
    """
    Train a baseline Logistic Regression classification model.

    Parameters:
        X_train (pd.DataFrame or np.ndarray): Training feature matrix.
        y_train (pd.Series or np.ndarray): Training binary target labels.
        random_state (int): Seed for reproducibility.
        max_iter (int): Maximum iterations for solver convergence.
        solver (str): Algorithm to use in optimization ('lbfgs', 'liblinear', etc.).

    Returns:
        LogisticRegression: Trained scikit-learn Logistic Regression model.
    """
    model = LogisticRegression(random_state=random_state, max_iter=max_iter, solver=solver)
    model.fit(X_train, y_train)
    return model


def predict_model(model, X_test):
    """
    Generate class predictions and positive class probabilities.

    Parameters:
        model: Trained classifier supporting predict and predict_proba.
        X_test (pd.DataFrame or np.ndarray): Test feature matrix.

    Returns:
        tuple: (y_pred, y_prob) where:
            - y_pred (np.ndarray): Binary predicted class labels (0 or 1).
            - y_prob (np.ndarray): Predicted probability for the positive class (1).
    """
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # Fallback to decision function normalized via sigmoid if predict_proba is unavailable
        df_vals = model.decision_function(X_test)
        y_prob = 1.0 / (1.0 + np.exp(-df_vals))
    else:
        y_prob = y_pred.astype(float)
        
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        tol = 1e-4
        if not (np.any(y_prob < -tol) or np.any(y_prob > 1.0 + tol)):
            y_prob = np.clip(y_prob, 0.0, 1.0)

    return y_pred, y_prob


def evaluate_performance_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate baseline machine learning classification metrics.

    Parameters:
        y_true (pd.Series or np.ndarray): Ground truth binary target labels.
        y_pred (pd.Series or np.ndarray): Predicted binary target labels.
        y_prob (pd.Series or np.ndarray, optional): Predicted positive class probabilities.

    Returns:
        dict: Performance metrics dictionary containing:
            - accuracy (float)
            - precision (float)
            - recall (float)
            - f1_score (float)
            - roc_auc (float or None)
            - confusion_matrix (dict with TN, FP, FN, TP and matrix list)
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, np.asarray(y_prob)))
        except ValueError:
            roc_auc = None

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
        }
    }

    return metrics
