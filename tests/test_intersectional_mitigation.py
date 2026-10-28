"""
Regression Test Suite for Step 4 Intersectional Bias Mitigation.

Verifies:
1. Dropdown scope selection (options, formatting, and baseline scope retrieval).
2. Correct intersectional group construction consistent with Step 3.
3. Actual application of the fairness constraint to combined groups.
4. Before-and-after metric calculations (accuracy, F1, DPD, EOD, DIR, calibration).
5. Low-sample group handling (insufficient coverage detection, statistically defensible partitioning).
6. Both supported mitigation strategies (ExponentiatedGradient & ThresholdOptimizer).
7. Backward compatibility of gender and ethnicity mitigation.
8. Result persistence and downstream dashboard consistency (Step 5 Trade-off & Step 9 Governance).
"""

import pytest
import numpy as np
import pandas as pd

from dashboard.utils import (
    load_dataset_run_result,
    normalize_run_results,
    get_step4_baseline_fairness_scope,
    get_step4_comparative_evaluations,
    get_step5_tradeoff_analysis_data,
    get_step9_governance_dashboard_data,
    execute_interactive_mitigation_workflow
)
from src.fairness.intersectional import (
    create_intersectional_attribute,
    audit_intersectional_attributes
)


# ==============================================================================
# 1. DROPDOWN SCOPE SELECTION
# ==============================================================================
def test_1_dropdown_scope_selection():
    """Verify dropdown scope options, formatting, and baseline metrics retrieval for intersectional scope."""
    raw_run = load_dataset_run_result("students")
    assert raw_run is not None
    norm = normalize_run_results(raw_run)

    single_data = norm.get("single_attribute", {})
    inter_data = norm.get("intersectional", {})
    available_attrs = list(single_data.keys())
    prot_attrs = inter_data.get("attributes_combined") or available_attrs
    inter_term = " Ã— ".join([a.replace("_", " ").title() for a in prot_attrs])

    # Formatter testing
    def format_scope_label(scope_key: str) -> str:
        if scope_key == "intersectional":
            return f"Intersectional Compound Scope ({inter_term})"
        return f"Baseline {scope_key.replace('_', ' ').title()} Scope"

    assert format_scope_label("gender") == "Baseline Gender Scope"
    assert format_scope_label("ethnicity") == "Baseline Ethnicity Scope"
    assert format_scope_label("intersectional") == "Intersectional Compound Scope (Gender Ã— Ethnicity)"

    # Baseline scope resolution
    scope = get_step4_baseline_fairness_scope(norm, selected_attribute="intersectional")
    assert scope["scope_type"] == "intersectional"
    assert scope["scope_name"] == "Intersectional"
    assert "Intersectional Compound Scope" in scope["scope_label"]
    assert "intersectional" in scope["available_attributes"]
    assert "gender" in scope["available_attributes"]
    assert "ethnicity" in scope["available_attributes"]

    # Intersectional baseline disparity matches Step 3
    assert scope["intersectional_estimable"] is True
    assert scope["intersectional_status"] == "ESTIMABLE"
    assert scope["demographic_parity_difference"] == pytest.approx(0.1526, abs=1e-3)
    assert scope["equalized_odds_difference"] == pytest.approx(0.5000, abs=1e-3)
    assert scope["disparate_impact_ratio"] == pytest.approx(0.8452, abs=1e-3)


# ==============================================================================
# 2. CORRECT INTERSECTIONAL GROUP CONSTRUCTION
# ==============================================================================
def test_2_correct_intersectional_group_construction():
    """Verify compound groups are constructed dynamically and consistently with Step 3."""
    A_df = pd.DataFrame({
        "gender": ["F", "M", "Non-binary", " F ", "M"],
        "ethnicity": ["A", "B", "C", "D", " E "]
    })

    combined = create_intersectional_attribute(A_df, attributes=["gender", "ethnicity"], join_str=" + ")
    expected = ["F + A", "M + B", "Non-binary + C", "F + D", "M + E"]
    assert list(combined) == expected
    assert combined.name == "gender + ethnicity"


# ==============================================================================
# 3. ACTUAL APPLICATION OF FAIRNESS CONSTRAINT TO COMBINED GROUPS
# ==============================================================================
def test_3_actual_application_of_fairness_constraint_to_combined_groups():
    """Verify that selecting intersectional scope constrains combined groups and logs 15 intersections."""
    logs = []
    res = execute_interactive_mitigation_workflow(
        dataset_id="students",
        strategy_type="in_processing",
        constraint_type="DemographicParity",
        target_scope="intersectional",
        eps=0.02,
        max_iter=10,
        logger_callback=lambda m: logs.append(m)
    )

    assert res["status"] == "SUCCESS"
    assert res["target_scope"] == "intersectional"
    assert res["mitigated"]["target_scope"] == "intersectional"
    assert res["scope_name"] == "Intersectional Compound Scope"

    full_log = " ".join(logs + res.get("logs", []))
    assert "Detected 15 distinct sensitive group intersections in training set." in full_log
    assert "[STATISTICAL SAFETY] Intersectional scope evaluated across 11 primary groups" in full_log
    assert "DemographicParity" in full_log


# ==============================================================================
# 4. BEFORE-AND-AFTER METRIC CALCULATIONS
# ==============================================================================
def test_4_before_and_after_intersectional_metrics():
    """Verify Step 4 comparative evaluations populate before/after metrics and deltas for intersectional scope."""
    mock_active_mit = {
        "status": "SUCCESS",
        "dataset_id": "students",
        "target_scope": "intersectional",
        "baseline": {
            "performance": {"accuracy": 1.0000, "f1_score": 1.0000},
            "calibration": {"brier_score": 0.0000, "ece": 0.0000},
            "fairness": {
                "coverage_status": {"code": "AVAILABLE"},
                "disparities": {
                    "demographic_parity_difference": 0.1052,
                    "equalized_odds_difference": 0.0000,
                    "disparate_impact_ratio": 0.8918
                }
            }
        },
        "mitigated": {
            "performance": {"accuracy": 0.9950, "f1_score": 0.9950},
            "calibration": {"brier_score": 0.0050, "ece": 0.0020},
            "fairness": {
                "coverage_status": {"code": "AVAILABLE"},
                "disparities": {
                    "demographic_parity_difference": 0.0350,
                    "equalized_odds_difference": 0.0000,
                    "disparate_impact_ratio": 0.9600
                }
            }
        }
    }

    comp = get_step4_comparative_evaluations(mock_active_mit, selected_attribute="intersectional")
    assert comp["has_executed"] is True
    table = {row["Metric"]: row for row in comp["table_data"]}

    assert "Accuracy" in table
    assert table["Accuracy"]["Before"] == "1.0000"
    assert table["Accuracy"]["After"] == "0.9950"

    assert "Demographic Parity Difference (Intersectional)" in table
    assert table["Demographic Parity Difference (Intersectional)"]["Before"] == "0.1052"
    assert table["Demographic Parity Difference (Intersectional)"]["After"] == "0.0350"
    assert table["Demographic Parity Difference (Intersectional)"]["Delta (Î”)"] == "-0.0702"

    assert "Disparate Impact Ratio (Intersectional)" in table
    assert table["Disparate Impact Ratio (Intersectional)"]["Before"] == "0.8918"

    assert comp["fairness_gains"]["scope_name"] == "Intersectional"
    assert comp["fairness_gains"]["dpd_gain"] == pytest.approx(0.1052 - 0.0350, abs=1e-4)


# ==============================================================================
# 5. LOW-SAMPLE GROUP HANDLING
# ==============================================================================
def test_5_low_sample_group_handling():
    """Verify loan dataset raises clear error when intersectional constraint is requested with N < 30."""
    raw_run = load_dataset_run_result("final_project_loan_demo__1")
    norm = normalize_run_results(raw_run)

    # In Step 4 baseline, loan intersectional scope reports NOT_ESTIMABLE with None disparities
    scope = get_step4_baseline_fairness_scope(norm, selected_attribute="intersectional")
    assert scope["intersectional_estimable"] is False
    assert scope["intersectional_status"] == "NOT_ESTIMABLE"
    assert scope["demographic_parity_difference"] is None
    assert scope["equalized_odds_difference"] is None

    # Attempting to execute intersectional mitigation raises clear descriptive ValueError
    with pytest.raises(ValueError) as excinfo:
        execute_interactive_mitigation_workflow(
            dataset_id="final_project_loan_demo__1",
            target_scope="intersectional",
            max_iter=5
        )
    assert "Intersectional mitigation constraint cannot be applied reliably" in str(excinfo.value)
    assert "N >= 30" in str(excinfo.value)


# ==============================================================================
# 6. BOTH SUPPORTED MITIGATION STRATEGIES
# ==============================================================================
def test_6_both_mitigation_strategies_on_intersectional_scope():
    """Verify both ExponentiatedGradient and ThresholdOptimizer support intersectional scope."""
    # Synthetic dataset with 4 balanced intersectional groups meeting N >= 30 in test set
    np.random.seed(42)
    n = 800
    df = pd.DataFrame({
        "num1": np.random.randn(n),
        "gender": ["F"] * 400 + ["M"] * 400,
        "race": (["A"] * 200 + ["B"] * 200) * 2,
        "label": np.random.choice([0, 1], size=n, p=[0.4, 0.6])
    })

    # Strategy 1: in_processing (ExponentiatedGradient)
    res_in = execute_interactive_mitigation_workflow(
        dataset_id="synth_in",
        df=df,
        target_column="label",
        positive_class=1,
        protected_attributes=["gender", "race"],
        strategy_type="in_processing",
        constraint_type="DemographicParity",
        target_scope="intersectional",
        min_group_size=30,
        max_iter=5
    )
    assert res_in["status"] == "SUCCESS"
    assert "ExponentiatedGradient" in res_in["mitigated"]["algorithm"]

    # Strategy 2: post_processing (ThresholdOptimizer)
    res_post = execute_interactive_mitigation_workflow(
        dataset_id="synth_post",
        df=df,
        target_column="label",
        positive_class=1,
        protected_attributes=["gender", "race"],
        strategy_type="post_processing",
        constraint_type="DemographicParity",
        target_scope="intersectional",
        min_group_size=30
    )
    assert res_post["status"] == "SUCCESS"
    assert "ThresholdOptimizer" in res_post["mitigated"]["algorithm"]


# ==============================================================================
# 7. BACKWARD COMPATIBILITY OF GENDER AND ETHNICITY MITIGATION
# ==============================================================================
def test_7_backward_compatibility_gender_and_ethnicity_scopes():
    """Verify selecting single attribute scopes constrains only that attribute without altering behavior."""
    raw_run = load_dataset_run_result("students")
    norm = normalize_run_results(raw_run)

    # Baseline Gender scope
    scope_g = get_step4_baseline_fairness_scope(norm, selected_attribute="gender")
    assert scope_g["scope_name"] == "Gender"
    assert scope_g["scope_label"] == "Baseline Gender Fairness"
    assert scope_g["demographic_parity_difference"] == pytest.approx(0.0491, abs=1e-3)

    # Baseline Ethnicity scope
    scope_e = get_step4_baseline_fairness_scope(norm, selected_attribute="ethnicity")
    assert scope_e["scope_name"] == "Ethnicity"
    assert scope_e["scope_label"] == "Baseline Ethnicity Fairness"
    assert scope_e["demographic_parity_difference"] == pytest.approx(0.0197, abs=1e-3)

    # Execution with single attribute gender
    logs_g = []
    res_g = execute_interactive_mitigation_workflow(
        dataset_id="students",
        strategy_type="in_processing",
        constraint_type="EqualizedOdds",
        target_scope="gender",
        eps=0.01,
        max_iter=5,
        logger_callback=lambda m: logs_g.append(m)
    )
    assert res_g["target_scope"] == "gender"
    assert any("Detected 3 distinct sensitive group intersections" in l for l in logs_g)

    # Execution with single attribute ethnicity
    logs_e = []
    res_e = execute_interactive_mitigation_workflow(
        dataset_id="students",
        strategy_type="in_processing",
        constraint_type="EqualizedOdds",
        target_scope="ethnicity",
        eps=0.01,
        max_iter=5,
        logger_callback=lambda m: logs_e.append(m)
    )
    assert res_e["target_scope"] == "ethnicity"
    assert any("Detected 5 distinct sensitive group intersections" in l for l in logs_e)


# ==============================================================================
# 8. RESULT PERSISTENCE AND DOWNSTREAM DASHBOARD CONSISTENCY
# ==============================================================================
def test_8_downstream_dashboard_consistency():
    """Verify Step 5 Trade-off Analysis and Step 9 Governance data reflect intersectional scope consistently."""
    mock_active_mit = {
        "status": "SUCCESS",
        "dataset_id": "students",
        "target_scope": "intersectional",
        "strategy_type": "in_processing",
        "baseline": {
            "performance": {"accuracy": 1.0000, "f1_score": 1.0000},
            "calibration": {"brier_score": 0.0000, "ece": 0.0000},
            "fairness": {
                "coverage_status": {"code": "AVAILABLE"},
                "disparities": {
                    "demographic_parity_difference": 0.1052,
                    "equalized_odds_difference": 0.0000,
                    "disparate_impact_ratio": 0.8918
                }
            }
        },
        "mitigated": {
            "algorithm": "Fairlearn ExponentiatedGradient (Random Forest)",
            "constraint_type": "DemographicParity",
            "target_scope": "intersectional",
            "eps": 0.01,
            "performance": {"accuracy": 0.9950, "f1_score": 0.9950},
            "calibration": {"brier_score": 0.0050, "ece": 0.0020},
            "fairness": {
                "coverage_status": {"code": "AVAILABLE"},
                "disparities": {
                    "demographic_parity_difference": 0.0350,
                    "equalized_odds_difference": 0.0000,
                    "disparate_impact_ratio": 0.9600
                }
            }
        }
    }

    # Step 5 Consistency
    step5 = get_step5_tradeoff_analysis_data(
        active_ds="students",
        selected_attribute="intersectional",
        session_mit_res=mock_active_mit
    )
    assert step5["has_mitigation"] is True
    assert step5["scope_name"] == "Intersectional"
    assert "intersectional" in step5["available_attrs"]

    fair_card = step5["summary_cards"]["fairness"]
    assert fair_card["scope_name"] == "Intersectional"

    # Step 9 Governance Consistency
    step9 = get_step9_governance_dashboard_data(
        dataset_id="students",
        session_mit_res=mock_active_mit
    )
    qv = step9["evaluator_quick_view"]
    assert "Intersectional Compound Scope" in qv["q5_mitigation"]
    assert step9["mitigation_info"]["scope"] == "Intersectional Compound Scope"


# ==============================================================================
# 9. HONEST REPORTING WHEN MITIGATION TARGET UNMET OR UNCHANGED
# ==============================================================================
def test_9_honest_reporting_when_mitigation_target_unmet_or_unchanged():
    """Verify UI and workflow honestly distinguish 'mitigation executed' from 'target achieved'."""
    res = execute_interactive_mitigation_workflow(
        dataset_id="students",
        strategy_type="in_processing",
        constraint_type="EqualizedOdds",
        target_scope="intersectional",
        eps=0.01,
        max_iter=5,
        random_seed=42
    )

    fgate = res.get("fairness_gate", {})
    mit_dpd = res["mitigated"]["fairness"]["disparities"]["demographic_parity_difference"]
    mit_eod = res["mitigated"]["fairness"]["disparities"]["equalized_odds_difference"]
    assert mit_dpd == pytest.approx(0.0596, abs=1e-3)
    assert mit_eod == pytest.approx(0.5500, abs=1e-3)
    assert fgate["threshold"] == 0.10

    # The gate must honestly report TARGET_UNMET, NOT PASS!
    assert fgate["target_achieved"] is False
    assert "TARGET UNMET" in fgate["gate_badge"]

    # Comparative table compliance column
    comp = get_step4_comparative_evaluations(res, selected_attribute="intersectional")
    row_map = {r["Metric"]: r for r in comp["table_data"]}
    eod_row = row_map["Equalized Odds Difference (Intersectional)"]
    assert eod_row["Before"] == "0.5000"
    assert eod_row["After"] == "0.5500"
    assert "EXCEEDS THRESHOLD" in eod_row["Compliance"]
    assert "Worsened" in eod_row["Status"]


# ==============================================================================
# 10. VALID PROBABILITY-BASED CALIBRATION METRICS (NO ZERO COLLAPSE)
# ==============================================================================
def test_10_valid_probability_calibration_metrics():
    """Verify that mitigated model probabilities are continuous soft probabilities and Brier/ECE are non-zero."""
    res = execute_interactive_mitigation_workflow(
        dataset_id="students",
        strategy_type="in_processing",
        constraint_type="DemographicParity",
        target_scope="intersectional",
        eps=0.01,
        max_iter=5,
        random_seed=42
    )
    mit_cal = res["mitigated"]["calibration"]
    brier = mit_cal.get("brier_score")
    ece = mit_cal.get("ece")

    # Brier score and ECE must be genuine positive floats, NOT collapsed to 0.0000
    assert brier is not None and brier > 0.0001
    assert ece is not None and ece > 0.0001


# ==============================================================================
# 11. THRESHOLD OPTIMIZER DEGENERATE GROUP HANDLING
# ==============================================================================
def test_11_threshold_optimizer_degenerate_group_handling():
    """Verify ThresholdOptimizer catches single-class intersectional groups and provides clear diagnostics."""
    with pytest.raises(ValueError) as excinfo:
        execute_interactive_mitigation_workflow(
            dataset_id="students",
            strategy_type="post_processing",
            constraint_type="DemographicParity",
            target_scope="intersectional",
            min_group_size=30
        )
    err = str(excinfo.value)
    assert "degenerate single-class labels detected" in err or "cannot construct ROC tradeoff curves" in err
    assert "ExponentiatedGradient" in err


# ==============================================================================
# 12. GROUP METRICS RECOMPUTATION PRESERVATION (TPR, FPR, FNR)
# ==============================================================================
def test_12_group_metrics_recomputation_preservation():
    """Verify that all intersectional groups preserve sample_count, TPR, FPR, FNR, and selection rates."""
    res = execute_interactive_mitigation_workflow(
        dataset_id="students",
        strategy_type="in_processing",
        constraint_type="DemographicParity",
        target_scope="intersectional",
        eps=0.01,
        max_iter=5,
        random_seed=42
    )
    all_groups = res["mitigated"]["fairness"]["all_group_metrics"]
    assert len(all_groups) == 15

    for g_name, g_data in all_groups.items():
        assert "sample_count" in g_data
        assert "selection_rate" in g_data
        assert "true_positive_rate" in g_data
        assert "false_positive_rate" in g_data
        assert "false_negative_rate" in g_data
        assert 0.0 <= g_data["selection_rate"] <= 1.0
        assert 0.0 <= g_data["true_positive_rate"] <= 1.0


# ==============================================================================
# 13. NO 'NONE: NONE' MESSAGE AND SAFE GATE FALLBACK
# ==============================================================================
def test_13_no_none_none_message_and_safe_gate_fallback():
    """Verify that even with an empty or unpopulated gate, comparative evaluations synthesize a valid gate."""
    # Simulate a legacy or unpopulated mitigation result
    mock_legacy_mit = {
        "status": "SUCCESS",
        "baseline": {
            "performance": {"accuracy": 1.0, "f1_score": 1.0},
            "fairness": {"disparities": {"demographic_parity_difference": 0.1052, "equalized_odds_difference": 0.0}},
            "calibration": {"brier_score": 0.0009, "ece": 0.0082}
        },
        "mitigated": {
            "performance": {"accuracy": 1.0, "f1_score": 1.0},
            "fairness": {"disparities": {"demographic_parity_difference": 0.1052, "equalized_odds_difference": 0.0}},
            "calibration": {"brier_score": 0.0013, "ece": 0.0105}
        }
        # Notice: 'fairness_gate' is completely missing!
    }

    comp = get_step4_comparative_evaluations(mock_legacy_mit, selected_attribute="intersectional")
    fgate = comp.get("fairness_gate", {})
    assert fgate is not None
    assert isinstance(fgate, dict)

    badge = fgate.get("gate_badge")
    explanation = fgate.get("explanation")
    assert badge is not None
    assert explanation is not None
    assert badge != "None"
    assert explanation != "None"

    # Simulate UI message formatting
    msg = f"Mitigation algorithm executed, but {badge}: {explanation}"
    assert "None: None" not in msg
    assert "TARGET UNMET" in msg


# ==============================================================================
# 14. EXACT USER SCENARIO: 40 ITERATIONS DEMOGRAPHIC PARITY ON STUDENTS
# ==============================================================================
def test_14_exact_user_configuration_target_unmet_and_continuous_calibration():
    """Verify exact user run: students, Random Forest, DemographicParity, max_iter=10 (fast proxy for 40)."""
    res = execute_interactive_mitigation_workflow(
        dataset_id="students",
        base_model_key="random_forest",
        strategy_type="in_processing",
        constraint_type="DemographicParity",
        target_scope="intersectional",
        eps=0.01,
        max_iter=10,
        random_seed=42
    )

    fgate = res.get("fairness_gate", {})
    mit_dpd = res["mitigated"]["fairness"]["disparities"]["demographic_parity_difference"]
    mit_eod = res["mitigated"]["fairness"]["disparities"]["equalized_odds_difference"]
    # With clean features (no target leakage), DPD improves to <= 0.1000:
    assert mit_dpd <= 0.10
    assert mit_dpd == pytest.approx(0.0596, abs=1e-3)
    # Trade-off: EOD exceeds 0.10 threshold:
    assert mit_eod > 0.10

    # Gate must report TARGET UNMET (not PASS!) because secondary thresholds are exceeded
    assert fgate.get("target_achieved") is False
    assert "TARGET UNMET" in fgate.get("gate_badge", "")
    assert fgate.get("explanation") is not None
    assert "None: None" not in f"{fgate.get('gate_badge')}: {fgate.get('explanation')}"

    # Continuous soft probability calibration: Brier and ECE must be strictly positive
    cal = res["mitigated"]["calibration"]
    assert cal["brier_score"] > 0.0001
    assert cal["ece"] > 0.0001


# ==============================================================================
# 15. TARGET LEAKAGE REGRESSION: FEATURES & TARGET SEPARATION
# ==============================================================================
def test_15_students_target_leakage_features_excluded():
    """
    Regression test proving overall_gpa and student_id are excluded from model features,
    passed is the binary target outcome, and train/test splits remain deterministic.
    """
    import os
    from src.data.loader import load_config
    from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data

    students_csv = os.path.join("data", "raw", "students.csv")
    if not os.path.exists(students_csv):
        pytest.skip("students.csv not found")

    cfg = load_config("config/students_config.json")
    df = pd.read_csv(students_csv)

    X, y, A = prepare_pipeline_data(df, config=cfg)

    # 1. overall_gpa must be excluded from feature matrix
    assert "overall_gpa" not in X.columns
    assert not any("overall_gpa" in col for col in X.columns)

    # 2. student_id must be excluded from feature matrix
    assert "student_id" not in X.columns
    assert not any("student_id" in col for col in X.columns)

    # 3. passed must be excluded from feature matrix
    assert "passed" not in X.columns
    assert not any("passed" in col for col in X.columns)

    # 4. passed is the target variable y
    assert y.name == "passed"
    assert set(y.unique()).issubset({0, 1})

    # 5. Protected attributes gender & ethnicity in A
    assert list(A.columns) == ["gender", "ethnicity"]

    # 6. Feature dimensions: 28 features (down from 29 when overall_gpa was leaked)
    assert X.shape[1] == 28

    # 7. Train/test split preservation (8000 train / 2000 test, seed 42)
    splits = split_pipeline_data(X, y, A, config=cfg)
    assert splits["X_train"].shape == (8000, 28)
    assert splits["X_test"].shape == (2000, 28)
    assert cfg["test_size"] == 0.2
    assert cfg["random_state"] == 42
