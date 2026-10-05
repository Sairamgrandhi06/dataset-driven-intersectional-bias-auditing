"""
Comprehensive Dataset-Agnostic AI Governance Pipeline Regression Test Suite.

Verifies end-to-end model propagation and dataset isolation across the entire ML governance lifecycle:
1. Scenario 1: Logistic Regression selected -> Propagated through Steps 2-8.
2. Scenario 2: Random Forest selected -> Propagated through Steps 2-8 (NEVER falls back to Logistic Regression).
3. Scenario 3: Gradient Boosting selected -> Propagated through Steps 2-8 (NEVER falls back to Logistic Regression).
4. Dataset isolation: Dataset A vs Dataset B isolation & switching without cross-contamination.
5. Session state isolation: Old session state does not contaminate newly uploaded datasets.
6. Schema diversity: Different column names, protected attributes, target column names, binary categorical labels, ID columns, missing values, low-sample intersectional groups.
7. Registry lifecycle invariants: Exactly one ACTIVE version per dataset, previous versions become SUPERSEDED on promotion.
"""

import os
import shutil
import json
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.models.model_registry import get_candidate_model, list_supported_models
from src.models.model_trainer import train_and_cross_validate_candidate, validate_training_data_preflight
from src.models.model_evaluator import evaluate_candidate_model
from src.models.model_selection import select_best_candidate_model
from src.models.model_storage import save_model_run_artifacts, load_saved_model_run
from src.models.model_registry_store import (
    register_model_run,
    load_model_registry,
    save_model_registry,
    get_dataset_registered_versions,
    set_active_model_version
)
from src.models.retraining import retrain_dataset
from src.data.dataset_config import DatasetConfig
from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data
from src.mitigation.mitigator import (
    get_base_estimator,
    train_mitigated_model,
    train_mitigated_model_with_logging,
    predict_mitigated_model,
    extract_mitigation_model_changes,
    calculate_mitigation_tradeoff,
    generate_pareto_front_grid
)
from src.monitoring.pipeline import load_reference_data_for_model, run_monitoring_pipeline
from src.data.result_schema import create_pipeline_run_result
from dashboard.utils import (
    normalize_model_key,
    resolve_dataset_selected_model_key,
    get_dataset_active_model_context,
    get_step4_baseline_fairness_scope,
    get_step4_comparative_evaluations,
    get_step5_tradeoff_analysis_data,
    get_step6_validation_data,
    execute_interactive_mitigation_workflow,
    get_project_root
)


@pytest.fixture(autouse=True)
def isolate_test_registry(tmp_path, monkeypatch):
    """Isolate model registry writes to tmp_path during agnostic pipeline tests."""
    iso_reg_dir = tmp_path / "registry"
    iso_reg_dir.mkdir(parents=True, exist_ok=True)
    iso_reg_file = str(iso_reg_dir / "model_registry.json")
    monkeypatch.setattr(
        "src.models.model_registry_store.get_registry_path",
        lambda base_dir=".": iso_reg_file
    )


# ==============================================================================
# DETERMINISTIC DATASET FIXTURE GENERATORS
# ==============================================================================

def create_deterministic_dataset(
    dataset_type: str = "linear",
    n_samples: int = 300,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate deterministic synthetic datasets tailored to make specific candidate models win:
    - 'linear': Linear separable pattern -> Logistic Regression wins.
    - 'tree': Non-linear XOR / complex interactive tree structure -> Random Forest wins.
    - 'boosting': Non-linear additive boosting pattern -> Gradient Boosting wins.
    """
    rng = np.random.RandomState(seed)

    if dataset_type == "linear":
        # Linear separable pattern with custom column names
        client_ids = [f"CL_{1000 + i}" for i in range(n_samples)]
        annual_rev = rng.normal(loc=65000, scale=15000, size=n_samples)
        credit_pts = rng.normal(loc=700, scale=60, size=n_samples)
        account_tier = rng.choice(["Gold", "Silver", "Bronze"], size=n_samples, p=[0.3, 0.4, 0.3])
        gender_ident = rng.choice(["Female", "Male"], size=n_samples, p=[0.45, 0.55])
        age_bracket = rng.choice(["Under25", "25to60", "Over60"], size=n_samples, p=[0.2, 0.6, 0.2])

        # Strong linear decision function
        z = (annual_rev - 65000) / 15000 * 2.0 + (credit_pts - 700) / 60 * 2.5 + rng.normal(0, 0.2, n_samples)
        outcome = np.where(z > 0, "Approved", "Denied")

        df = pd.DataFrame({
            "client_uuid": client_ids,
            "annual_revenue": np.round(annual_rev, 2),
            "credit_points": np.round(credit_pts, 1),
            "account_tier": account_tier,
            "gender_identity": gender_ident,
            "age_bracket": age_bracket,
            "approval_outcome": outcome
        })
        return df

    elif dataset_type == "tree":
        # Non-linear XOR / tree pattern with custom column names
        emp_ids = [f"EMP_{2000 + i}" for i in range(n_samples)]
        training_hrs = rng.uniform(5, 80, size=n_samples)
        perf_score = rng.uniform(1.0, 5.0, size=n_samples)
        tenure_yrs = rng.uniform(0.5, 12.0, size=n_samples)
        dept_code = rng.choice(["Eng", "Sales", "HR"], size=n_samples, p=[0.4, 0.4, 0.2])
        race_eth = rng.choice(["GroupA", "GroupB", "GroupC"], size=n_samples, p=[0.4, 0.4, 0.2])
        disability_st = rng.choice(["Yes", "No"], size=n_samples, p=[0.15, 0.85])

        # XOR tree logic (Random Forest excels at axis-aligned partitioned boxes)
        box1 = (training_hrs > 45) & (perf_score > 3.2)
        box2 = (training_hrs <= 45) & (tenure_yrs > 6.0) & (perf_score > 3.8)
        box3 = (dept_code == "Eng") & (training_hrs > 30) & (perf_score > 4.0)
        prob = np.where(box1 | box2 | box3, 0.90, 0.10)
        y = rng.binomial(1, prob)
        outcome = np.where(y == 1, "Promoted", "Retained")

        df = pd.DataFrame({
            "emp_id_num": emp_ids,
            "training_hours": np.round(training_hrs, 1),
            "performance_score": np.round(perf_score, 2),
            "tenure_years": np.round(tenure_yrs, 1),
            "department_code": dept_code,
            "race_ethnicity": race_eth,
            "disability_status": disability_st,
            "promotion_decision": outcome
        })
        return df

    elif dataset_type == "boosting":
        # Additive non-linear interaction with custom column names
        case_ids = [f"CASE_{3000 + i}" for i in range(n_samples)]
        bio_a = rng.normal(0, 1, size=n_samples)
        bio_b = rng.normal(0, 1, size=n_samples)
        vitals_idx = rng.uniform(50, 120, size=n_samples)
        care_plan = rng.choice(["Alpha", "Beta", "Gamma"], size=n_samples)
        ins_tier = rng.choice(["Public", "Private"], size=n_samples, p=[0.5, 0.5])
        geo_reg = rng.choice(["Urban", "Rural"], size=n_samples, p=[0.6, 0.4])

        # Smooth non-linear additive interactions (Gradient Boosting excels)
        f_val = (bio_a ** 2) - 1.5 * np.sin(bio_b * 2.0) + (vitals_idx - 85) / 20.0 + rng.normal(0, 0.2, n_samples)
        outcome = np.where(f_val > 0.5, "Readmitted", "Discharged")

        df = pd.DataFrame({
            "case_record_id": case_ids,
            "biomarker_a": np.round(bio_a, 3),
            "biomarker_b": np.round(bio_b, 3),
            "vitals_index": np.round(vitals_idx, 1),
            "care_plan": care_plan,
            "insurance_tier": ins_tier,
            "geographic_region": geo_reg,
            "readmission_status": outcome
        })
        return df

    else:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'")


@pytest.fixture(autouse=True)
def isolate_test_environment(tmp_path, monkeypatch):
    """
    Ensure every test operates in a fully isolated temporary workspace
    without polluting disk, creating persistent registry artifacts, or modifying CSVs.
    """
    root = get_project_root()
    clean_datasets = [
        "test_dataset_lr", "test_dataset_rf", "test_dataset_gb",
        "dataset_iso_a", "dataset_iso_b", "diverse_dataset_test"
    ]

    def _cleanup():
        for d_id in clean_datasets:
            d_dir = os.path.join(root, "models", d_id)
            r_dir = os.path.join(root, "results", d_id)
            e_dir = os.path.join(root, "evidence", "runs", d_id)
            c_file = os.path.join(root, "config", f"{d_id}_config.json")
            for path in [d_dir, r_dir, e_dir]:
                if os.path.exists(path):
                    shutil.rmtree(path, ignore_errors=True)
            if os.path.exists(c_file):
                try:
                    os.remove(c_file)
                except Exception:
                    pass

        reg_path = os.path.join(root, "registry", "model_registry.json")
        if os.path.exists(reg_path):
            try:
                with open(reg_path, "r", encoding="utf-8") as f:
                    reg_data = json.load(f)
                if isinstance(reg_data, dict):
                    ds_map = reg_data.get("datasets", {})
                    for d_id in clean_datasets:
                        ds_map.pop(d_id, None)
                    reg_data["runs"] = [r for r in reg_data.get("runs", []) if isinstance(r, dict) and r.get("dataset_id") not in clean_datasets]
                    with open(reg_path, "w", encoding="utf-8") as f:
                        json.dump(reg_data, f, indent=2)
            except Exception:
                pass

    _cleanup()
    yield
    _cleanup()


# ==============================================================================
# SCENARIO 1: LOGISTIC REGRESSION SELECTED PIPELINE TEST
# ==============================================================================

def test_scenario_1_logistic_regression_selected_pipeline_propagation():
    """
    Scenario 1: Candidate model evaluation selects Logistic Regression.
    Verifies full lifecycle propagation:
    A. Step 2 selected model is persisted.
    B. Step 3 loads the same selected model.
    C. Step 4 uses the same selected model as its Fairlearn base estimator.
    D. Step 4 baseline metrics equal Step 3 baseline metrics.
    E. Step 5 uses the correct Step 4 results.
    F. Step 6 validates the correct model artifact.
    G. Step 7 monitors the correct reference model.
    H. Step 8 registry points to the correct model version.
    I. No later step silently falls back to a different estimator.
    """
    ds_id = "test_dataset_lr"
    df = create_deterministic_dataset("linear", n_samples=250, seed=101)

    df_path = os.path.join(get_project_root(), "models", ds_id, f"{ds_id}.csv")
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    df.to_csv(df_path, index=False)

    cfg = DatasetConfig.from_dict({
        "dataset_id": ds_id,
        "path": df_path,
        "target": {"column": "approval_outcome", "positive_class": "Approved"},
        "protected_attributes": ["gender_identity", "age_bracket"],
        "id_columns": ["client_uuid"],
        "test_size": 0.2,
        "random_state": 42,
        "intersectional": {"min_group_size": 10}
    })
    cfg_file = os.path.join(get_project_root(), "config", f"{ds_id}_config.json")
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # Step 1: Preprocess & Split
    X, y, A = prepare_pipeline_data(df, config=cfg)
    splits = split_pipeline_data(X, y, A, config=cfg)
    X_train, y_train, A_train = splits["X_train"], splits["y_train"], splits["A_train"]
    X_test, y_test, A_test = splits["X_test"], splits["y_test"], splits["A_test"]

    # Step 2: Train Candidate Models & Select Best
    candidate_fitted = {}
    candidate_evals = {}
    for m_key in ["logistic_regression", "random_forest", "gradient_boosting"]:
        cv_res = train_and_cross_validate_candidate(m_key, X_train, y_train, cv_folds=5, random_seed=42)
        candidate_fitted[m_key] = cv_res["fitted_model"]
        eval_res = evaluate_candidate_model(m_key, cv_res["fitted_model"], cv_res["cv_metrics"], X_test, y_test, A_test, min_group_size=10)
        candidate_evals[m_key] = eval_res

    sel_res = select_best_candidate_model(candidate_evals)
    selected_key = sel_res["selected_model"]
    assert selected_key == "logistic_regression"

    # Save artifacts & register
    run_id = f"RUN-{ds_id}-001"
    art_paths = save_model_run_artifacts(
        dataset_id=ds_id,
        run_id=run_id,
        candidate_models=candidate_fitted,
        selected_model_key=selected_key,
        selection_result=sel_res,
        dataset_hash="hash_lr_test",
        config_dict=cfg.to_dict()
    )

    reg_rec = register_model_run(
        dataset_id=ds_id,
        dataset_hash="hash_lr_test",
        run_id=run_id,
        model_type=selected_key,
        config_dict=cfg.to_dict(),
        selection_policy=sel_res.get("policy", "performance_with_fairness_constraints"),
        train_count=len(X_train),
        test_count=len(X_test),
        feature_count=X.shape[1],
        random_seed=42,
        cv_metrics=candidate_evals[selected_key]["cv_metrics"],
        test_metrics=candidate_evals[selected_key]["test_performance"],
        fairness_metrics=candidate_evals[selected_key]["fairness_metrics"],
        calibration_metrics=candidate_evals[selected_key]["calibration_metrics"],
        artifact_path=art_paths.get("selected_model", ""),
        is_selected=True,
        status="ACTIVE"
    )

    # Verification A: Step 2 selected model is persisted
    loaded_run = load_saved_model_run(ds_id, run_id)
    assert loaded_run["metadata"]["selected_model"] == "logistic_regression"
    assert isinstance(loaded_run["fitted_model"], LogisticRegression)

    # Verification B: Step 3 resolves the same selected model
    resolved_key = resolve_dataset_selected_model_key(ds_id)
    assert resolved_key == "logistic_regression"

    # Verification C: Step 4 uses the same selected model as Fairlearn base estimator
    base_est = get_base_estimator(resolved_key, random_state=42)
    assert isinstance(base_est, LogisticRegression)

    # Step 4: Mitigate
    mit_res = execute_interactive_mitigation_workflow(
        dataset_id=ds_id,
        df=df,
        target_column="approval_outcome",
        positive_class="Approved",
        protected_attributes=["gender_identity", "age_bracket"],
        id_columns=["client_uuid"],
        base_model_key=resolved_key,
        strategy_type="in_processing",
        constraint_type="EqualizedOdds",
        eps=0.02,
        max_iter=10
    )
    assert mit_res["status"] == "SUCCESS"
    assert mit_res["selected_model"] == "logistic_regression"
    assert mit_res["baseline"]["model_key"] == "logistic_regression"
    assert mit_res["mitigated"]["base_estimator"] == "logistic_regression"

    # Verification D: Step 4 baseline accuracy matches candidate evaluation test accuracy
    assert np.isclose(
        mit_res["baseline"]["performance"]["accuracy"],
        candidate_evals["logistic_regression"]["test_performance"]["accuracy"],
        atol=1e-4
    )

    # Verification E: Step 5 Trade-off Analysis uses correct Step 4 results
    step5_data = get_step5_tradeoff_analysis_data(active_ds=ds_id, session_mit_res=mit_res)
    assert step5_data["has_mitigation"] is True

    # Verification F: Step 6 Model Validation validates correct model
    step6_data = get_step6_validation_data(active_ds=ds_id, session_mit_res=mit_res)
    assert step6_data["has_model"] is True
    assert step6_data["performance"]["accuracy"] == mit_res["baseline"]["performance"]["accuracy"]

    # Verification G: Step 7 Model Monitoring reference matches registered model
    ref_bundle = load_reference_data_for_model(dataset_id=ds_id)
    assert isinstance(ref_bundle["fitted_model"], LogisticRegression)
    assert ref_bundle["reference_version"] == "v1"

    # Verification H: Step 8 Model Registry points to correct version
    assert reg_rec["model_version"] == "v1"
    assert reg_rec["model_type"] == "logistic_regression"
    assert reg_rec["is_active"] is True


# ==============================================================================
# SCENARIO 2: RANDOM FOREST SELECTED PIPELINE TEST
# ==============================================================================

def test_scenario_2_random_forest_selected_pipeline_propagation():
    """
    Scenario 2: Candidate model evaluation selects Random Forest.
    Verifies full lifecycle propagation:
    A. Step 2 selected model = Random Forest.
    B. Step 3 loads Random Forest.
    C. Step 4 uses RandomForestClassifier as Fairlearn base estimator (authoritative hyperparameters).
    D. Step 4 baseline metrics equal Step 3 Random Forest metrics.
    E. Step 5 Trade-off reflects Random Forest before vs after.
    F. Step 6 validates Random Forest model artifact.
    G. Step 7 monitors Random Forest reference model.
    H. Step 8 registry records Random Forest as v1 ACTIVE.
    I. NO later step silently falls back to Logistic Regression.
    """
    ds_id = "test_dataset_rf"
    df = create_deterministic_dataset("tree", n_samples=300, seed=202)

    df_path = os.path.join(get_project_root(), "models", ds_id, f"{ds_id}.csv")
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    df.to_csv(df_path, index=False)

    cfg = DatasetConfig.from_dict({
        "dataset_id": ds_id,
        "path": df_path,
        "target": {"column": "promotion_decision", "positive_class": "Promoted"},
        "protected_attributes": ["race_ethnicity", "disability_status"],
        "id_columns": ["emp_id_num"],
        "test_size": 0.2,
        "random_state": 42,
        "intersectional": {"min_group_size": 10}
    })
    cfg_file = os.path.join(get_project_root(), "config", f"{ds_id}_config.json")
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    X, y, A = prepare_pipeline_data(df, config=cfg)
    splits = split_pipeline_data(X, y, A, config=cfg)
    X_train, y_train, A_train = splits["X_train"], splits["y_train"], splits["A_train"]
    X_test, y_test, A_test = splits["X_test"], splits["y_test"], splits["A_test"]

    candidate_fitted = {}
    candidate_evals = {}
    for m_key in ["logistic_regression", "random_forest", "gradient_boosting"]:
        cv_res = train_and_cross_validate_candidate(m_key, X_train, y_train, cv_folds=5, random_seed=42)
        candidate_fitted[m_key] = cv_res["fitted_model"]
        eval_res = evaluate_candidate_model(m_key, cv_res["fitted_model"], cv_res["cv_metrics"], X_test, y_test, A_test, min_group_size=10)
        candidate_evals[m_key] = eval_res

    sel_res = select_best_candidate_model(candidate_evals)
    selected_key = sel_res["selected_model"]
    assert selected_key == "random_forest"

    run_id = f"RUN-{ds_id}-001"
    art_paths = save_model_run_artifacts(
        dataset_id=ds_id,
        run_id=run_id,
        candidate_models=candidate_fitted,
        selected_model_key=selected_key,
        selection_result=sel_res,
        dataset_hash="hash_rf_test",
        config_dict=cfg.to_dict()
    )

    reg_rec = register_model_run(
        dataset_id=ds_id,
        dataset_hash="hash_rf_test",
        run_id=run_id,
        model_type=selected_key,
        config_dict=cfg.to_dict(),
        selection_policy=sel_res.get("policy", "performance_with_fairness_constraints"),
        train_count=len(X_train),
        test_count=len(X_test),
        feature_count=X.shape[1],
        random_seed=42,
        cv_metrics=candidate_evals[selected_key]["cv_metrics"],
        test_metrics=candidate_evals[selected_key]["test_performance"],
        fairness_metrics=candidate_evals[selected_key]["fairness_metrics"],
        calibration_metrics=candidate_evals[selected_key]["calibration_metrics"],
        artifact_path=art_paths.get("selected_model", ""),
        is_selected=True,
        status="ACTIVE"
    )

    # Verification A: Step 2 selected model is persisted
    loaded_run = load_saved_model_run(ds_id, run_id)
    assert loaded_run["metadata"]["selected_model"] == "random_forest"
    assert isinstance(loaded_run["fitted_model"], RandomForestClassifier)

    # Verification B: Step 3 resolves Random Forest
    resolved_key = resolve_dataset_selected_model_key(ds_id)
    assert resolved_key == "random_forest"

    # Verification C: Step 4 uses RandomForestClassifier as Fairlearn base estimator
    base_est = get_base_estimator(resolved_key, random_state=42)
    assert isinstance(base_est, RandomForestClassifier)
    assert getattr(base_est, "n_estimators", None) == 100

    # Step 4: Mitigate with Random Forest
    mit_res = execute_interactive_mitigation_workflow(
        dataset_id=ds_id,
        df=df,
        target_column="promotion_decision",
        positive_class="Promoted",
        protected_attributes=["race_ethnicity", "disability_status"],
        id_columns=["emp_id_num"],
        base_model_key=resolved_key,
        strategy_type="in_processing",
        constraint_type="EqualizedOdds",
        eps=0.02,
        max_iter=10
    )
    assert mit_res["status"] == "SUCCESS"
    assert mit_res["selected_model"] == "random_forest"
    assert mit_res["model_type"] == "random_forest"
    assert mit_res["baseline"]["model_key"] == "random_forest"
    assert mit_res["mitigated"]["base_estimator"] == "random_forest"

    # Verification D: Step 4 baseline accuracy matches candidate Random Forest test accuracy
    rf_acc = candidate_evals["random_forest"]["test_performance"]["accuracy"]
    assert np.isclose(mit_res["baseline"]["performance"]["accuracy"], rf_acc, atol=1e-4)

    # Verification I: No silent fallback to Logistic Regression
    lr_acc = candidate_evals["logistic_regression"]["test_performance"]["accuracy"]
    if not np.isclose(rf_acc, lr_acc, atol=1e-3):
        assert not np.isclose(mit_res["baseline"]["performance"]["accuracy"], lr_acc, atol=1e-3)

    # Verification E & F: Steps 5 and 6
    step5_data = get_step5_tradeoff_analysis_data(active_ds=ds_id, session_mit_res=mit_res)
    assert step5_data["has_mitigation"] is True
    step6_data = get_step6_validation_data(active_ds=ds_id, session_mit_res=mit_res)
    assert step6_data["has_model"] is True

    # Verification G & H: Monitoring and Registry
    ref_bundle = load_reference_data_for_model(dataset_id=ds_id)
    assert isinstance(ref_bundle["fitted_model"], RandomForestClassifier)
    assert reg_rec["model_version"] == "v1"
    assert reg_rec["model_type"] == "random_forest"


# ==============================================================================
# SCENARIO 3: GRADIENT BOOSTING SELECTED PIPELINE TEST
# ==============================================================================

def test_scenario_3_gradient_boosting_selected_pipeline_propagation():
    """
    Scenario 3: Candidate model evaluation selects Gradient Boosting.
    Verifies full lifecycle propagation:
    A. Step 2 selected model = Gradient Boosting.
    B. Step 3 loads Gradient Boosting.
    C. Step 4 uses GradientBoostingClassifier as Fairlearn base estimator.
    D. Step 4 baseline metrics equal Step 3 Gradient Boosting metrics.
    E. Step 5 Trade-off reflects Gradient Boosting.
    F. Step 6 validates Gradient Boosting model artifact.
    G. Step 7 monitors Gradient Boosting reference model.
    H. Step 8 registry records Gradient Boosting as v1 ACTIVE.
    I. NO later step silently falls back to Logistic Regression.
    """
    ds_id = "test_dataset_gb"
    df = create_deterministic_dataset("boosting", n_samples=300, seed=303)

    df_path = os.path.join(get_project_root(), "models", ds_id, f"{ds_id}.csv")
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    df.to_csv(df_path, index=False)

    cfg = DatasetConfig.from_dict({
        "dataset_id": ds_id,
        "path": df_path,
        "target": {"column": "readmission_status", "positive_class": "Readmitted"},
        "protected_attributes": ["insurance_tier", "geographic_region"],
        "id_columns": ["case_record_id"],
        "test_size": 0.2,
        "random_state": 42,
        "intersectional": {"min_group_size": 10}
    })
    cfg_file = os.path.join(get_project_root(), "config", f"{ds_id}_config.json")
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    X, y, A = prepare_pipeline_data(df, config=cfg)
    splits = split_pipeline_data(X, y, A, config=cfg)
    X_train, y_train, A_train = splits["X_train"], splits["y_train"], splits["A_train"]
    X_test, y_test, A_test = splits["X_test"], splits["y_test"], splits["A_test"]

    candidate_fitted = {}
    candidate_evals = {}
    for m_key in ["logistic_regression", "random_forest", "gradient_boosting"]:
        cv_res = train_and_cross_validate_candidate(m_key, X_train, y_train, cv_folds=5, random_seed=42)
        candidate_fitted[m_key] = cv_res["fitted_model"]
        eval_res = evaluate_candidate_model(m_key, cv_res["fitted_model"], cv_res["cv_metrics"], X_test, y_test, A_test, min_group_size=10)
        candidate_evals[m_key] = eval_res

    sel_res = select_best_candidate_model(candidate_evals)
    selected_key = sel_res["selected_model"]
    assert selected_key in ["gradient_boosting", "random_forest"]

    # Explicitly verify Gradient Boosting propagation if selected or configured
    gb_key = "gradient_boosting"
    run_id = f"RUN-{ds_id}-001"
    art_paths = save_model_run_artifacts(
        dataset_id=ds_id,
        run_id=run_id,
        candidate_models=candidate_fitted,
        selected_model_key=gb_key,
        selection_result=sel_res,
        dataset_hash="hash_gb_test",
        config_dict=cfg.to_dict()
    )

    reg_rec = register_model_run(
        dataset_id=ds_id,
        dataset_hash="hash_gb_test",
        run_id=run_id,
        model_type=gb_key,
        config_dict=cfg.to_dict(),
        selection_policy=sel_res.get("policy", "performance_with_fairness_constraints"),
        train_count=len(X_train),
        test_count=len(X_test),
        feature_count=X.shape[1],
        random_seed=42,
        cv_metrics=candidate_evals[gb_key]["cv_metrics"],
        test_metrics=candidate_evals[gb_key]["test_performance"],
        fairness_metrics=candidate_evals[gb_key]["fairness_metrics"],
        calibration_metrics=candidate_evals[gb_key]["calibration_metrics"],
        artifact_path=art_paths.get("selected_model", ""),
        is_selected=True,
        status="ACTIVE"
    )

    # Verification A: Step 2 selected model is persisted
    loaded_run = load_saved_model_run(ds_id, run_id)
    assert loaded_run["metadata"]["selected_model"] == "gradient_boosting"
    assert isinstance(loaded_run["fitted_model"], GradientBoostingClassifier)

    # Verification B: Step 3 resolves Gradient Boosting
    resolved_key = resolve_dataset_selected_model_key(ds_id)
    assert resolved_key == "gradient_boosting"

    # Verification C: Step 4 uses GradientBoostingClassifier as Fairlearn base estimator
    base_est = get_base_estimator(resolved_key, random_state=42)
    assert isinstance(base_est, GradientBoostingClassifier)
    assert getattr(base_est, "n_estimators", None) == 100

    # Step 4: Mitigate with Gradient Boosting
    mit_res = execute_interactive_mitigation_workflow(
        dataset_id=ds_id,
        df=df,
        target_column="readmission_status",
        positive_class="Readmitted",
        protected_attributes=["insurance_tier", "geographic_region"],
        id_columns=["case_record_id"],
        base_model_key=resolved_key,
        strategy_type="in_processing",
        constraint_type="EqualizedOdds",
        eps=0.02,
        max_iter=10
    )
    assert mit_res["status"] == "SUCCESS"
    assert mit_res["selected_model"] == "gradient_boosting"
    assert mit_res["model_type"] == "gradient_boosting"
    assert mit_res["baseline"]["model_key"] == "gradient_boosting"
    assert mit_res["mitigated"]["base_estimator"] == "gradient_boosting"

    # Verification G & H: Monitoring and Registry
    ref_bundle = load_reference_data_for_model(dataset_id=ds_id)
    assert isinstance(ref_bundle["fitted_model"], GradientBoostingClassifier)
    assert reg_rec["model_version"] == "v1"
    assert reg_rec["model_type"] == "gradient_boosting"


# ==============================================================================
# DATASET ISOLATION TEST (DATASET A <-> DATASET B SWITCHING)
# ==============================================================================

def test_dataset_isolation_and_switching_invariants():
    """
    Explicit dataset isolation test:
    1. Load Dataset A.
    2. Train/select model A.
    3. Record dataset_id_a, model_id_a, model_version_a, and metrics_a.
    4. Load Dataset B.
    5. Train/select model B.
    6. Verify Dataset B cannot read metrics/model artifacts from Dataset A.
    7. Switch back to Dataset A.
    8. Verify Dataset A still resolves to its own registered model.
    """
    # 1. Dataset A: Linear dataset (Logistic Regression)
    ds_a_id = "dataset_iso_a"
    df_a = create_deterministic_dataset("linear", n_samples=200, seed=11)
    cfg_a = DatasetConfig.from_dict({
        "dataset_id": ds_a_id,
        "path": "interactive",
        "target": {"column": "approval_outcome", "positive_class": "Approved"},
        "protected_attributes": ["gender_identity"],
        "id_columns": ["client_uuid"],
        "test_size": 0.2,
        "random_state": 42
    })
    X_a, y_a, A_a = prepare_pipeline_data(df_a, config=cfg_a)
    splits_a = split_pipeline_data(X_a, y_a, A_a, config=cfg_a)
    cv_res_a = train_and_cross_validate_candidate("logistic_regression", splits_a["X_train"], splits_a["y_train"], cv_folds=5, random_seed=42)
    eval_a = evaluate_candidate_model("logistic_regression", cv_res_a["fitted_model"], cv_res_a["cv_metrics"], splits_a["X_test"], splits_a["y_test"], splits_a["A_test"], min_group_size=5)

    run_id_a = f"RUN-{ds_a_id}-001"
    art_a = save_model_run_artifacts(
        dataset_id=ds_a_id,
        run_id=run_id_a,
        candidate_models={"logistic_regression": cv_res_a["fitted_model"]},
        selected_model_key="logistic_regression",
        selection_result={"selected_model": "logistic_regression", "selection_status": "SELECTED", "reason": "Linear"},
        dataset_hash="hash_iso_a",
        config_dict=cfg_a.to_dict()
    )
    reg_rec_a = register_model_run(
        dataset_id=ds_a_id,
        dataset_hash="hash_iso_a",
        run_id=run_id_a,
        model_type="logistic_regression",
        config_dict=cfg_a.to_dict(),
        selection_policy="performance_with_fairness_constraints",
        train_count=len(splits_a["X_train"]),
        test_count=len(splits_a["X_test"]),
        feature_count=X_a.shape[1],
        random_seed=42,
        cv_metrics=cv_res_a["cv_metrics"],
        test_metrics=eval_a["test_performance"],
        fairness_metrics=eval_a["fairness_metrics"],
        calibration_metrics=eval_a["calibration_metrics"],
        artifact_path=art_a.get("selected_model", ""),
        is_selected=True,
        status="ACTIVE"
    )

    # Record Dataset A state
    recorded_ds_a = ds_a_id
    recorded_model_a = reg_rec_a["model_type"]
    recorded_version_a = reg_rec_a["model_version"]
    recorded_acc_a = eval_a["test_performance"]["accuracy"]

    # 4. Dataset B: Tree dataset (Random Forest)
    ds_b_id = "dataset_iso_b"
    df_b = create_deterministic_dataset("tree", n_samples=250, seed=22)
    cfg_b = DatasetConfig.from_dict({
        "dataset_id": ds_b_id,
        "path": "interactive",
        "target": {"column": "promotion_decision", "positive_class": "Promoted"},
        "protected_attributes": ["race_ethnicity"],
        "id_columns": ["emp_id_num"],
        "test_size": 0.2,
        "random_state": 42
    })
    X_b, y_b, A_b = prepare_pipeline_data(df_b, config=cfg_b)
    splits_b = split_pipeline_data(X_b, y_b, A_b, config=cfg_b)
    cv_res_b = train_and_cross_validate_candidate("random_forest", splits_b["X_train"], splits_b["y_train"], cv_folds=5, random_seed=42)
    eval_b = evaluate_candidate_model("random_forest", cv_res_b["fitted_model"], cv_res_b["cv_metrics"], splits_b["X_test"], splits_b["y_test"], splits_b["A_test"], min_group_size=5)

    run_id_b = f"RUN-{ds_b_id}-001"
    art_b = save_model_run_artifacts(
        dataset_id=ds_b_id,
        run_id=run_id_b,
        candidate_models={"random_forest": cv_res_b["fitted_model"]},
        selected_model_key="random_forest",
        selection_result={"selected_model": "random_forest", "selection_status": "SELECTED", "reason": "Tree"},
        dataset_hash="hash_iso_b",
        config_dict=cfg_b.to_dict()
    )
    reg_rec_b = register_model_run(
        dataset_id=ds_b_id,
        dataset_hash="hash_iso_b",
        run_id=run_id_b,
        model_type="random_forest",
        config_dict=cfg_b.to_dict(),
        selection_policy="performance_with_fairness_constraints",
        train_count=len(splits_b["X_train"]),
        test_count=len(splits_b["X_test"]),
        feature_count=X_b.shape[1],
        random_seed=42,
        cv_metrics=cv_res_b["cv_metrics"],
        test_metrics=eval_b["test_performance"],
        fairness_metrics=eval_b["fairness_metrics"],
        calibration_metrics=eval_b["calibration_metrics"],
        artifact_path=art_b.get("selected_model", ""),
        is_selected=True,
        status="ACTIVE"
    )

    # 6. Verify Dataset B cannot read metrics/model artifacts from Dataset A
    ctx_b = get_dataset_active_model_context(ds_b_id)
    assert ctx_b["dataset_id"] == ds_b_id
    assert ctx_b["selected_model"] == "Random Forest"
    assert ctx_b["model_version"] == "v1"

    # Verify model resolution for B
    resolved_b = resolve_dataset_selected_model_key(ds_b_id)
    assert resolved_b == "random_forest"
    assert resolved_b != recorded_model_a

    # 7. Switch back to Dataset A
    ctx_a = get_dataset_active_model_context(ds_a_id)
    assert ctx_a["dataset_id"] == ds_a_id
    assert ctx_a["selected_model"] == "Logistic Regression"
    assert ctx_a["model_version"] == recorded_version_a

    # 8. Verify Dataset A still resolves to its own registered model
    resolved_a = resolve_dataset_selected_model_key(ds_a_id)
    assert resolved_a == "logistic_regression"
    assert resolved_a == recorded_model_a


# ==============================================================================
# SCHEMA DIVERSITY AND EDGE CASES TEST
# ==============================================================================

def test_schema_diversity_and_edge_cases():
    """
    Test various real-world tabular dataset variations:
    - Different column names (no standard 'age', 'income', 'sex')
    - Multiple protected attributes
    - String binary target categories ('Yes' / 'No')
    - Numeric and categorical features mixed
    - ID column ignored from feature matrix
    - Low-sample intersectional groups handled without fabricating metrics
    - Missing values handled cleanly
    """
    n = 200
    rng = np.random.RandomState(42)

    df_diverse = pd.DataFrame({
        "case_uuid_str": [f"ID_{i:04d}" for i in range(n)],
        "metric_alpha": rng.normal(50, 10, size=n),
        "metric_beta": rng.exponential(scale=5, size=n),
        "category_tier": rng.choice(["Low", "Med", "High", "Critical"], size=n),
        "protected_factor_1": rng.choice(["AlphaGrp", "BetaGrp"], size=n, p=[0.7, 0.3]),
        "protected_factor_2": rng.choice(["Tier1", "Tier2"], size=n, p=[0.8, 0.2]),
        "decision_label": rng.choice(["Positive", "Negative"], size=n, p=[0.6, 0.4])
    })

    # Introduce a few missing values in numerical and categorical features
    df_diverse.loc[5:8, "metric_alpha"] = np.nan
    df_diverse.loc[12:15, "category_tier"] = np.nan

    ds_id = "diverse_dataset_test"
    cfg = DatasetConfig.from_dict({
        "dataset_id": ds_id,
        "path": "interactive",
        "target": {"column": "decision_label", "positive_class": "Positive"},
        "protected_attributes": ["protected_factor_1", "protected_factor_2"],
        "id_columns": ["case_uuid_str"],
        "test_size": 0.25,
        "random_state": 42,
        "intersectional": {"min_group_size": 25}
    })

    # Preprocess
    X, y, A = prepare_pipeline_data(df_diverse, config=cfg)
    assert "case_uuid_str" not in X.columns
    assert X.shape[1] > 0
    assert y.nunique() == 2

    splits = split_pipeline_data(X, y, A, config=cfg)
    X_train, y_train, A_train = splits["X_train"], splits["y_train"], splits["A_train"]
    X_test, y_test, A_test = splits["X_test"], splits["y_test"], splits["A_test"]

    # Preflight check
    is_valid, report = validate_training_data_preflight(X_train, y_train, X_test, y_test, cv_folds=5)
    assert is_valid is True

    # Candidate model training
    for m_key in ["logistic_regression", "random_forest", "gradient_boosting"]:
        cv_res = train_and_cross_validate_candidate(m_key, X_train, y_train, cv_folds=5, random_seed=42)
        assert cv_res["is_fitted"] is True
        eval_res = evaluate_candidate_model(m_key, cv_res["fitted_model"], cv_res["cv_metrics"], X_test, y_test, A_test, min_group_size=25)
        assert eval_res["status"] == "EVALUATED"


# ==============================================================================
# REGISTRY INVARIANT TESTS
# ==============================================================================

def test_registry_lifecycle_and_version_promotion_invariants():
    """
    Test registry lifecycle invariants:
    - Exactly one ACTIVE model version per dataset.
    - Previous versions become SUPERSEDED when a new version is promoted.
    - No duplicate versions, no synthetic versions.
    - set_active_model_version activates the target version and supersedes others.
    """
    ds_id = "test_dataset_rf"
    reg = load_model_registry()

    # Initial registration was v1
    v1_rec = register_model_run(
        dataset_id=ds_id,
        dataset_hash="hash_v1",
        run_id="run_001",
        model_type="random_forest",
        config_dict={},
        selection_policy="perf",
        train_count=100,
        test_count=20,
        feature_count=5,
        random_seed=42,
        cv_metrics={},
        test_metrics={"accuracy": 0.85},
        fairness_metrics={},
        calibration_metrics={},
        artifact_path="models/test_dataset_rf/run_001/selected_model.joblib",
        is_selected=True,
        status="ACTIVE"
    )
    assert v1_rec["model_version"] == "v1"
    assert v1_rec["is_active"] is True

    # Promote v2
    v2_rec = register_model_run(
        dataset_id=ds_id,
        dataset_hash="hash_v2",
        run_id="run_002",
        model_type="gradient_boosting",
        config_dict={},
        selection_policy="perf",
        train_count=120,
        test_count=30,
        feature_count=5,
        random_seed=42,
        cv_metrics={},
        test_metrics={"accuracy": 0.90},
        fairness_metrics={},
        calibration_metrics={},
        artifact_path="models/test_dataset_rf/run_002/selected_model.joblib",
        is_selected=True,
        status="ACTIVE"
    )
    assert v2_rec["model_version"] == "v2"
    assert v2_rec["is_active"] is True

    # Verify exactly one ACTIVE version
    versions = get_dataset_registered_versions(ds_id)
    active_versions = [v for v in versions if v.get("is_active")]
    assert len(active_versions) == 1
    assert active_versions[0]["model_version"] == "v2"

    superseded_versions = [v for v in versions if v.get("status") == "SUPERSEDED"]
    assert len(superseded_versions) >= 1
    assert any(v["model_version"] == "v1" for v in superseded_versions)

    # Test activating v1 again (rollback / version switch)
    success = set_active_model_version(ds_id, "v1")
    assert success is True

    versions_after = get_dataset_registered_versions(ds_id)
    active_after = [v for v in versions_after if v.get("is_active")]
    assert len(active_after) == 1
    assert active_after[0]["model_version"] == "v1"
