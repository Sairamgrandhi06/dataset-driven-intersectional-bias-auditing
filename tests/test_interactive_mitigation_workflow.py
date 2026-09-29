"""
Unit and Integration Tests for Interactive Bias Mitigation, Live Logging,
Pre/Post Comparative Evaluations, and Model Changes Extraction.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.mitigation.mitigator import (
    train_mitigated_model_with_logging,
    extract_mitigation_model_changes,
    calculate_mitigation_tradeoff,
    predict_mitigated_model
)
from dashboard.utils import (
    execute_interactive_mitigation_workflow,
    get_raw_dataset_dataframe
)


@pytest.fixture
def synthetic_interactive_data():
    """Synthetic dataset fixture for testing interactive mitigation."""
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "num_feature_1": np.random.normal(0, 1, n),
        "num_feature_2": np.random.uniform(10, 50, n),
        "cat_feature_1": np.random.choice(["TypeA", "TypeB", "TypeC"], size=n),
        "gender": np.random.choice(["Female", "Male"], size=n, p=[0.5, 0.5]),
        "race": np.random.choice(["Group1", "Group2"], size=n, p=[0.6, 0.4]),
        "outcome": np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    })
    return df


def test_train_mitigated_model_with_logging():
    """Verify train_mitigated_model_with_logging captures structured live logs."""
    np.random.seed(42)
    n = 150
    X_train = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y_train = pd.Series(np.random.choice([0, 1], size=n))
    sens = pd.Series(np.random.choice(["A", "B"], size=n))

    captured_logs = []
    def custom_log(msg):
        captured_logs.append(msg)

    model, logs = train_mitigated_model_with_logging(
        X_train=X_train,
        y_train=y_train,
        sensitive_features=sens,
        constraint_type="EqualizedOdds",
        eps=0.02,
        max_iter=20,
        logger_callback=custom_log
    )

    assert model is not None
    assert len(logs) > 0
    assert len(captured_logs) > 0
    assert any("Fairlearn" in l for l in logs)
    assert any("EqualizedOdds" in l for l in logs)


def test_extract_mitigation_model_changes():
    """Verify extract_mitigation_model_changes extracts ensemble weights and parameters."""
    np.random.seed(42)
    n = 150
    X_train = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y_train = pd.Series(np.random.choice([0, 1], size=n))
    sens = pd.Series(np.random.choice(["A", "B"], size=n))

    model, _ = train_mitigated_model_with_logging(
        X_train=X_train,
        y_train=y_train,
        sensitive_features=sens,
        constraint_type="EqualizedOdds",
        eps=0.05,
        max_iter=15
    )

    changes = extract_mitigation_model_changes(model, X_train, y_train, feature_names=["f1", "f2"])
    assert changes is not None
    assert "ensemble_composition" in changes
    assert len(changes["ensemble_composition"]) > 0
    assert changes["active_predictors_count"] >= 1
    assert "summary" in changes


def test_calculate_mitigation_tradeoff_with_percentage_gains():
    """Verify calculate_mitigation_tradeoff computes percentage reductions and group deltas."""
    base_perf = {"accuracy": 0.85, "precision": 0.80, "recall": 0.60, "f1_score": 0.686, "roc_auc": 0.88}
    mit_perf = {"accuracy": 0.82, "precision": 0.75, "recall": 0.65, "f1_score": 0.696, "roc_auc": 0.85}

    base_audit = {
        "disparities": {
            "equalized_odds_difference": 0.12,
            "demographic_parity_difference": 0.20,
            "disparate_impact_ratio": 0.40,
            "equal_opportunity_difference": 0.10
        },
        "all_group_metrics": {
            "Female + Group1": {"true_positive_rate": 0.50, "false_positive_rate": 0.05, "selection_rate": 0.10, "sample_count": 50},
            "Male + Group1": {"true_positive_rate": 0.62, "false_positive_rate": 0.12, "selection_rate": 0.25, "sample_count": 60}
        }
    }

    mit_audit = {
        "disparities": {
            "equalized_odds_difference": 0.03,
            "demographic_parity_difference": 0.06,
            "disparate_impact_ratio": 0.75,
            "equal_opportunity_difference": 0.03
        },
        "all_group_metrics": {
            "Female + Group1": {"true_positive_rate": 0.58, "false_positive_rate": 0.06, "selection_rate": 0.15, "sample_count": 50},
            "Male + Group1": {"true_positive_rate": 0.61, "false_positive_rate": 0.09, "selection_rate": 0.20, "sample_count": 60}
        }
    }

    report = calculate_mitigation_tradeoff(base_perf, mit_perf, base_audit, mit_audit)

    # Accuracy drop: -0.03
    assert report["performance_comparison"]["accuracy"]["delta"] == pytest.approx(-0.03)

    # EOD gain: 0.12 - 0.03 = 0.09 (75% reduction)
    eod_comp = report["fairness_comparison"]["equalized_odds_difference"]
    assert eod_comp["fairness_gain"] == pytest.approx(0.09)
    assert eod_comp["gain_pct"] == pytest.approx(75.0)

    # Group opportunity breakdown
    assert len(report["group_opportunity_breakdown"]) == 2
    f_grp = [g for g in report["group_opportunity_breakdown"] if "Female" in g["group"]][0]
    assert f_grp["tpr_delta"] == pytest.approx(0.08)


def test_execute_interactive_mitigation_workflow_synthetic(synthetic_interactive_data):
    """Verify execute_interactive_mitigation_workflow runs end-to-end with live logging."""
    df = synthetic_interactive_data
    logs = []

    res = execute_interactive_mitigation_workflow(
        dataset_id="test_synthetic",
        df=df,
        target_column="outcome",
        positive_class=1,
        protected_attributes=["gender", "race"],
        base_model_key="logistic_regression",
        constraint_type="EqualizedOdds",
        eps=0.05,
        max_iter=15,
        test_size=0.25,
        random_seed=42,
        min_group_size=10,
        logger_callback=lambda m: logs.append(m)
    )

    assert res["status"] == "SUCCESS"
    assert "baseline" in res
    assert "mitigated" in res
    assert "tradeoff" in res
    assert "model_changes" in res["mitigated"]
    assert len(logs) > 0
    assert res["baseline"]["performance"]["accuracy"] > 0.0
    assert res["mitigated"]["performance"]["accuracy"] > 0.0


def test_execute_post_processing_mitigation(synthetic_interactive_data):
    """Verify execute_interactive_mitigation_workflow supports post-processing ThresholdOptimizer."""
    df = synthetic_interactive_data
    logs = []

    res = execute_interactive_mitigation_workflow(
        dataset_id="test_postproc",
        df=df,
        target_column="outcome",
        positive_class=1,
        protected_attributes=["gender"],
        base_model_key="logistic_regression",
        strategy_type="post_processing",
        constraint_type="EqualizedOdds",
        test_size=0.25,
        random_seed=42,
        min_group_size=10,
        logger_callback=lambda m: logs.append(m)
    )

    assert res["status"] == "SUCCESS"
    assert res["strategy_type"] == "post_processing"
    assert "group_threshold_adjustments" in res["mitigated"]["model_changes"]
    assert len(logs) > 0


def test_execute_pareto_frontier_run(synthetic_interactive_data):
    """Verify execute_pareto_frontier_run computes trade-off frontier points."""
    from dashboard.utils import execute_pareto_frontier_run
    df = synthetic_interactive_data

    pts = execute_pareto_frontier_run(
        dataset_id="test_pareto",
        df=df,
        target_column="outcome",
        positive_class=1,
        protected_attributes=["gender"],
        constraint_type="EqualizedOdds",
        eps_grid=[0.01, 0.05]
    )

    assert len(pts) == 2
    assert "accuracy" in pts[0]
    assert "equalized_odds_difference" in pts[0]

