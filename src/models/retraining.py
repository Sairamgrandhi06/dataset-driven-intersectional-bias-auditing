"""
Automated Retraining Engine.

Executes dataset-specific model retraining workflows, executing profiling, preprocessing,
candidate training, 5-fold cross-validation, constraint selection, mitigation, calibration,
trade-off analysis, deterministic versioning, and registry lifecycle updates.
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, Any, Union

from src.data.dataset_config import DatasetConfig, load_dataset_config_by_id
from src.data.loader import load_raw_dataset_from_config
from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data
from src.models.model_trainer import train_and_cross_validate_candidate
from src.models.model_evaluator import evaluate_candidate_model
from src.models.model_selection import select_best_candidate_model
from src.models.model_storage import save_model_run_artifacts
from src.models.model_registry_store import generate_run_id, register_model_run
from src.fairness.single_attribute import run_single_attribute_audits
from src.fairness.intersectional import create_intersectional_attribute, audit_intersectional_attributes
from src.fairness.calibration import evaluate_calibration
from src.mitigation.mitigator import train_mitigated_model, predict_mitigated_model
from src.fairness.trade_off import compute_three_way_tradeoff
from src.data.result_schema import create_pipeline_run_result


from src.models.classifier import evaluate_performance_metrics
from src.fairness.trade_off import calculate_comprehensive_tradeoff


def retrain_dataset(
    dataset_id: str,
    config: Any = None,
    base_dir: str = "."
) -> Dict[str, Any]:
    """
    Execute dataset retraining workflow and register new model version in registry.

    Parameters:
        dataset_id (str): Dataset identifier string.
        config (str or dict): Path to dataset config file or configuration dictionary.
        base_dir (str): Base project directory path.

    Returns:
        dict: Standardized run results dictionary with versioning metadata.
    """
    # Load configuration object
    if isinstance(config, str):
        if os.path.exists(config):
            cfg = DatasetConfig.from_json(config)
        else:
            cfg = load_dataset_config_by_id(dataset_id, base_dir=base_dir)
    elif isinstance(config, DatasetConfig):
        cfg = config
    elif isinstance(config, dict):
        cfg = DatasetConfig.from_dict(config)
    else:
        cfg = load_dataset_config_by_id(dataset_id, base_dir=base_dir)

    # 1. Load Raw Dataset & Calculate Hash
    df_raw, metadata = load_raw_dataset_from_config(cfg, base_dir=base_dir)
    run_id = generate_run_id(dataset_id)

    # 2. Preprocess Data
    X, y, A = prepare_pipeline_data(df_raw, config=cfg)
    splits = split_pipeline_data(X, y, A, config=cfg)

    X_train, y_train, A_train = splits["X_train"], splits["y_train"], splits["A_train"]
    X_test, y_test, A_test = splits["X_test"], splits["y_test"], splits["A_test"]

    # 2.5 Preflight Validation
    from src.models.model_trainer import validate_training_data_preflight
    is_valid, pf_report = validate_training_data_preflight(X_train, y_train, X_test, y_test, cv_folds=5)
    if not is_valid:
        register_model_run(
            dataset_id=cfg.dataset_id,
            dataset_hash=metadata["file_sha256"],
            run_id=run_id,
            model_type="none",
            config_dict=cfg.to_dict(),
            selection_policy="none",
            train_count=len(X_train),
            test_count=len(X_test),
            feature_count=X.shape[1],
            random_seed=cfg.model_config.random_state,
            cv_metrics={},
            test_metrics={},
            fairness_metrics={},
            calibration_metrics={},
            artifact_path="",
            is_selected=False,
            status="FAILED",
            base_dir=base_dir
        )
        res = create_pipeline_run_result(
            dataset_id=cfg.dataset_id,
            dataset_hash=metadata["file_sha256"],
            config_version=getattr(cfg, "version", "1.0.0"),
            target_column=cfg.target.column,
            positive_class=cfg.target.positive_class,
            protected_attributes=cfg.protected_attributes,
            feature_count=X.shape[1],
            train_size=len(X_train),
            test_size=len(X_test),
            baseline_metrics={},
            fairness_metrics={},
            intersectional_metrics={},
            calibration_metrics={},
            errors=[pf_report["message"]]
        )
        res["run_id"] = run_id
        res["model_version"] = "NONE"
        res["model_status"] = "FAILED"
        res["status"] = pf_report["status"]
        res["selection_status"] = pf_report["status"]
        res["message"] = pf_report["message"]
        res["required_minimum"] = pf_report["required_minimum"]
        res["actual_rows"] = pf_report["actual_rows"]
        res["train_rows"] = pf_report["train_rows"]
        res["test_rows"] = pf_report["test_rows"]
        res["cv_folds"] = pf_report["cv_folds"]
        res["selected_model"] = None
        return res

    # 3. Train Candidate Models (LR, RF, GB) & Cross-Validate
    candidate_keys = ["logistic_regression", "random_forest", "gradient_boosting"]
    candidate_fitted = {}
    candidate_evals = {}

    for m_key in candidate_keys:
        cv_res = train_and_cross_validate_candidate(
            model_key=m_key,
            X_train=X_train,
            y_train=y_train,
            cv_folds=5,
            random_seed=cfg.model_config.random_state
        )
        fitted_m = cv_res["fitted_model"]
        candidate_fitted[m_key] = fitted_m

        eval_res = evaluate_candidate_model(
            model_key=m_key,
            fitted_model=fitted_m,
            cv_metrics=cv_res["cv_metrics"],
            X_test=X_test,
            y_test=y_test,
            A_test=A_test,
            min_group_size=cfg.intersectional.min_group_size
        )
        candidate_evals[m_key] = eval_res

    # 4. Perform Constraint-Based Model Selection
    selection_res = select_best_candidate_model(candidate_evals)
    selected_key = selection_res.get("selected_model")

    if not selected_key or selected_key not in candidate_fitted or candidate_fitted.get(selected_key) is None:
        register_model_run(
            dataset_id=cfg.dataset_id,
            dataset_hash=metadata["file_sha256"],
            run_id=run_id,
            model_type="none",
            config_dict=cfg.to_dict(),
            selection_policy=selection_res.get("policy", "performance_with_fairness_constraints"),
            train_count=len(X_train),
            test_count=len(X_test),
            feature_count=X.shape[1],
            random_seed=cfg.model_config.random_state,
            cv_metrics={},
            test_metrics={},
            fairness_metrics={},
            calibration_metrics={},
            artifact_path="",
            is_selected=False,
            status="FAILED",
            base_dir=base_dir
        )
        res = create_pipeline_run_result(
            dataset_id=cfg.dataset_id,
            dataset_hash=metadata["file_sha256"],
            config_version=getattr(cfg, "version", "1.0.0"),
            target_column=cfg.target.column,
            positive_class=cfg.target.positive_class,
            protected_attributes=cfg.protected_attributes,
            feature_count=X.shape[1],
            train_size=len(X_train),
            test_size=len(X_test),
            baseline_metrics={},
            fairness_metrics={},
            intersectional_metrics={},
            calibration_metrics={},
            errors=[selection_res.get("reason", "No valid fitted candidate model available.")]
        )
        res["run_id"] = run_id
        res["model_version"] = "NONE"
        res["model_status"] = "FAILED"
        res["status"] = "NO_VALID_MODEL"
        res["selection_status"] = "NO_VALID_MODEL"
        res["message"] = selection_res.get("reason")
        res["selected_model"] = None
        return res

    baseline_model = candidate_fitted[selected_key]
    selected_eval = candidate_evals[selected_key]

    # 5. Evaluate Held-Out Baseline Performance, Calibration & Audits
    y_base_pred = baseline_model.predict(X_test)
    y_base_prob = baseline_model.predict_proba(X_test)[:, 1] if hasattr(baseline_model, "predict_proba") else None

    base_perf = selected_eval["test_performance"]
    base_calib = evaluate_calibration(y_test, y_base_prob, n_bins=cfg.calibration_settings.get("n_bins", 10))
    single_audits = run_single_attribute_audits(np.array(y_test), y_base_pred, A_test)
    intersectional_audit = audit_intersectional_attributes(
        np.array(y_test), y_base_pred, A_test,
        attributes=cfg.protected_attributes,
        min_group_size=cfg.intersectional.min_group_size
    )

    # 6. Apply Fairlearn Mitigation & Calibration
    sens_train = create_intersectional_attribute(A_train, attributes=cfg.protected_attributes)
    sens_test = create_intersectional_attribute(A_test, attributes=cfg.protected_attributes)

    mitigated_model = train_mitigated_model(
        X_train, y_train, sens_train,
        random_state=cfg.model_config.random_state,
        max_iter=cfg.model_config.max_iter,
        base_estimator_type=selected_key
    )

    y_mit_pred, y_mit_prob = predict_mitigated_model(mitigated_model, X_test, random_state=0)
    mit_perf = evaluate_performance_metrics(np.array(y_test), y_mit_pred, y_mit_prob)
    mit_calib = evaluate_calibration(y_test, y_mit_prob, n_bins=cfg.calibration_settings.get("n_bins", 10))
    mit_intersectional = audit_intersectional_attributes(
        np.array(y_test), y_mit_pred, A_test,
        attributes=cfg.protected_attributes,
        min_group_size=cfg.intersectional.min_group_size
    )

    # 7. Calculate Three-Way Trade-off
    tradeoff_report = calculate_comprehensive_tradeoff(
        baseline_metrics=base_perf,
        mitigated_metrics=mit_perf,
        baseline_intersectional_audit=intersectional_audit,
        mitigated_intersectional_audit=mit_intersectional,
        baseline_calib=base_calib,
        mitigated_calib=mit_calib
    )

    # 8. Save Joblib Artifacts & Evidence
    artifact_paths = save_model_run_artifacts(
        dataset_id=cfg.dataset_id,
        run_id=run_id,
        candidate_models=candidate_fitted,
        selected_model_key=selected_key,
        selection_result=selection_res,
        dataset_hash=metadata["file_sha256"],
        config_dict=cfg.to_dict(),
        base_dir=base_dir
    )

    # 9. Register Model Run & Version in registry/model_registry.json
    reg_record = register_model_run(
        dataset_id=cfg.dataset_id,
        dataset_hash=metadata["file_sha256"],
        run_id=run_id,
        model_type=selected_key,
        config_dict=cfg.to_dict(),
        selection_policy=selection_res.get("policy", "performance_with_fairness_constraints"),
        train_count=len(X_train),
        test_count=len(X_test),
        feature_count=X.shape[1],
        random_seed=cfg.model_config.random_state,
        cv_metrics=selected_eval["cv_metrics"],
        test_metrics=base_perf,
        fairness_metrics=intersectional_audit["disparities"],
        calibration_metrics=base_calib,
        artifact_path=artifact_paths.get("selected_model", ""),
        is_selected=True,
        status="ACTIVE",
        base_dir=base_dir
    )

    # 10. Generate Standardized Pipeline Result JSON
    run_result = create_pipeline_run_result(
        dataset_id=cfg.dataset_id,
        dataset_hash=metadata["file_sha256"],
        config_version=getattr(cfg, "version", "1.0.0"),
        target_column=cfg.target.column,
        positive_class=cfg.target.positive_class,
        protected_attributes=cfg.protected_attributes,
        feature_count=X.shape[1],
        train_size=len(X_train),
        test_size=len(X_test),
        baseline_metrics=base_perf,
        single_attribute_metrics=single_audits,
        fairness_metrics=intersectional_audit["disparities"],
        intersectional_metrics=intersectional_audit,
        calibration_metrics=base_calib,
        mitigation_metrics=mit_intersectional["disparities"],
        tradeoff_metrics=tradeoff_report
    )

    # Attach model registry version & run_id metadata
    run_result["run_id"] = run_id
    run_result["model_version"] = reg_record["model_version"]
    run_result["model_status"] = reg_record["status"]

    return run_result
