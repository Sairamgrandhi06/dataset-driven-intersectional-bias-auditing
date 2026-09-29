"""
Unit tests for Bias Mitigation, Probability Bounds, and Trade-off Comparison Modules.
"""

import pytest
import numpy as np
import pandas as pd
from fairlearn.reductions import ExponentiatedGradient

from src.mitigation.mitigator import (
    get_fairness_constraint,
    train_mitigated_model,
    predict_mitigated_model,
    calculate_mitigation_tradeoff,
    generate_pareto_front_grid
)


@pytest.fixture
def synthetic_mitigation_data():
    """Fixture providing synthetic features, targets, and sensitive attributes."""
    np.random.seed(42)
    n_train = 200
    n_test = 50

    X_tr = pd.DataFrame({
        "feature_1": np.random.randn(n_train),
        "feature_2": np.random.randn(n_train)
    })
    y_tr = pd.Series(np.random.choice([0, 1], size=n_train, p=[0.6, 0.4]))
    sens_tr = pd.Series(np.random.choice(["GroupA", "GroupB"], size=n_train, p=[0.5, 0.5]))

    X_te = pd.DataFrame({
        "feature_1": np.random.randn(n_test),
        "feature_2": np.random.randn(n_test)
    })
    y_te = pd.Series(np.random.choice([0, 1], size=n_test, p=[0.6, 0.4]))
    sens_te = pd.Series(np.random.choice(["GroupA", "GroupB"], size=n_test, p=[0.5, 0.5]))

    return X_tr, y_tr, sens_tr, X_te, y_te, sens_te


def test_mitigation_configuration():
    """Verify get_fairness_constraint instantiates valid Fairlearn constraint moments."""
    c_eq = get_fairness_constraint("EqualizedOdds")
    assert c_eq is not None

    c_dp = get_fairness_constraint("DemographicParity")
    assert c_dp is not None

    with pytest.raises(ValueError, match="Unsupported constraint type"):
        get_fairness_constraint("InvalidConstraintName")


def test_training_only_mitigation(synthetic_mitigation_data):
    """Verify mitigation model trains exclusively on training data without touching test data."""
    X_tr, y_tr, sens_tr, X_te, y_te, sens_te = synthetic_mitigation_data

    model = train_mitigated_model(
        X_tr,
        y_tr,
        sensitive_features=sens_tr,
        constraint_type="EqualizedOdds",
        eps=0.05,
        random_state=42
    )

    assert isinstance(model, ExponentiatedGradient)
    # Ensure model prediction on test set preserves test set row count
    y_pred, y_prob = predict_mitigated_model(model, X_te)
    assert len(y_pred) == len(X_te) == 50
    assert len(y_prob) == 50


def test_prediction_generation(synthetic_mitigation_data):
    """Verify predictions and probability bounds [0, 1]."""
    X_tr, y_tr, sens_tr, X_te, y_te, sens_te = synthetic_mitigation_data

    model = train_mitigated_model(X_tr, y_tr, sensitive_features=sens_tr, eps=0.05)
    y_pred, y_prob = predict_mitigated_model(model, X_te)

    assert set(np.unique(y_pred)).issubset({0, 1})
    assert len(y_prob) == len(X_te)
    assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))


def test_tradeoff_comparison_calculations():
    """Verify baseline vs mitigated comparison calculation structure and deltas."""
    base_metrics = {"accuracy": 0.84, "precision": 0.75, "recall": 0.60, "f1_score": 0.667, "roc_auc": 0.88}
    mit_metrics = {"accuracy": 0.76, "precision": 0.65, "recall": 0.55, "f1_score": 0.595, "roc_auc": 0.78}

    base_audit = {
        "disparities": {
            "demographic_parity_difference": 0.31,
            "disparate_impact_ratio": 0.15,
            "equalized_odds_difference": 0.48,
            "equal_opportunity_difference": 0.48,
            "false_positive_rate_difference": 0.14
        }
    }

    mit_audit = {
        "disparities": {
            "demographic_parity_difference": 0.04,
            "disparate_impact_ratio": 0.60,
            "equalized_odds_difference": 0.12,
            "equal_opportunity_difference": 0.12,
            "false_positive_rate_difference": 0.05
        }
    }

    tradeoff = calculate_mitigation_tradeoff(base_metrics, mit_metrics, base_audit, mit_audit)

    # Accuracy delta = 0.76 - 0.84 = -0.08
    assert tradeoff["performance_comparison"]["accuracy"]["delta"] == pytest.approx(-0.08)

    # Equalized Odds fairness gain = 0.48 - 0.12 = 0.36
    assert tradeoff["fairness_comparison"]["equalized_odds_difference"]["fairness_gain"] == pytest.approx(0.36)

    # Demographic Parity fairness gain = 0.31 - 0.04 = 0.27
    assert tradeoff["fairness_comparison"]["demographic_parity_difference"]["fairness_gain"] == pytest.approx(0.27)


def test_reproducibility(synthetic_mitigation_data):
    """Verify deterministic training and prediction output with fixed random seed."""
    X_tr, y_tr, sens_tr, X_te, _, _ = synthetic_mitigation_data

    m1 = train_mitigated_model(X_tr, y_tr, sensitive_features=sens_tr, random_state=42)
    m2 = train_mitigated_model(X_tr, y_tr, sensitive_features=sens_tr, random_state=42)

    p1, prob1 = predict_mitigated_model(m1, X_te)
    p2, prob2 = predict_mitigated_model(m2, X_te)

    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_almost_equal(prob1, prob2)
