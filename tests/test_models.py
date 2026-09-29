"""
Unit tests for Baseline Model Training and Metric Evaluation Modules.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.classifier import (
    train_baseline_model,
    predict_model,
    evaluate_performance_metrics
)


@pytest.fixture
def synthetic_training_data():
    """Fixture providing simple synthetic classification training features and targets."""
    np.random.seed(42)
    X = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randn(100),
        "feature_3": np.random.choice([0, 1], size=100)
    })
    y = pd.Series(np.random.choice([0, 1], size=100, p=[0.7, 0.3]))
    return X, y


def test_train_baseline_model(synthetic_training_data):
    """Verify that train_baseline_model fits and returns a LogisticRegression classifier."""
    X_train, y_train = synthetic_training_data
    model = train_baseline_model(X_train, y_train, random_state=42, max_iter=100)

    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "classes_")
    assert len(model.classes_) == 2


def test_predict_model(synthetic_training_data):
    """Verify predict_model outputs arrays of expected shape and valid probability range."""
    X_train, y_train = synthetic_training_data
    model = train_baseline_model(X_train, y_train, random_state=42)

    X_test = X_train.iloc[:20]
    y_pred, y_prob = predict_model(model, X_test)

    assert len(y_pred) == 20
    assert len(y_prob) == 20
    assert set(np.unique(y_pred)).issubset({0, 1})
    assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))


def test_evaluate_performance_metrics():
    """Verify evaluation metric computation against exact ground truth values."""
    # Deterministic test case:
    # y_true: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] (4 positive, 6 negative)
    # y_pred: [1, 1, 1, 0, 1, 0, 0, 0, 0, 0] (3 TP, 1 FN, 1 FP, 5 TN)
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.6, 0.2, 0.1, 0.3, 0.2, 0.1])

    metrics = evaluate_performance_metrics(y_true, y_pred, y_prob)

    # TP=3, TN=5, FP=1, FN=1
    # Accuracy = (3 + 5) / 10 = 0.8
    # Precision = 3 / (3 + 1) = 0.75
    # Recall = 3 / (3 + 1) = 0.75
    # F1 = 2 * 0.75 * 0.75 / 1.5 = 0.75
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["precision"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["f1_score"] == pytest.approx(0.75)
    assert metrics["roc_auc"] is not None
    assert metrics["confusion_matrix"]["TP"] == 3
    assert metrics["confusion_matrix"]["TN"] == 5
    assert metrics["confusion_matrix"]["FP"] == 1
    assert metrics["confusion_matrix"]["FN"] == 1
