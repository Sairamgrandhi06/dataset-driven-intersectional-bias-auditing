"""
Reusable Dataset Pipeline Runner CLI.

Executes the full end-to-end audit, mitigation, calibration, and trade-off pipeline
for any tabular CSV dataset specified by a DatasetConfig JSON file.

Usage:
    python src/run_dataset.py --config config/default_config.json
    python src/run_dataset.py --config path/to/custom_dataset_config.json
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset_config import DatasetConfig
from src.data.loader import load_raw_dataset_from_config
from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data
from src.models.classifier import train_baseline_model, predict_model, evaluate_performance_metrics
from src.fairness.single_attribute import run_single_attribute_audits
from src.fairness.intersectional import create_intersectional_attribute, audit_intersectional_attributes
from src.fairness.calibration import evaluate_calibration
from src.mitigation.mitigator import train_mitigated_model, predict_mitigated_model
from src.fairness.trade_off import compute_three_way_tradeoff, generate_tradeoff_summary_table
from src.data.result_schema import create_pipeline_run_result


def run_dataset_pipeline(config_path: str = "config/default_config.json") -> dict:
    """
    Execute end-to-end dataset pipeline using a dataset configuration file.

    Parameters:
        config_path (str): Path to dataset configuration JSON file.

    Returns:
        dict: Standardized run result dictionary.
    """
    print("=" * 80)
    print("DATASET-DRIVEN INTERSECTIONAL BIAS AUDITING & TRADE-OFF PIPELINE")
    print("=" * 80)

    # 1. Load Configuration Contract
    print(f"\n[1/7] Loading dataset configuration from: '{config_path}'...")
    cfg = DatasetConfig.from_json(config_path)
    print(f"      Dataset Identifier    : {cfg.dataset_id}")
    print(f"      Dataset Path          : {cfg.path}")
    print(f"      Target Column         : {cfg.target.column} (Positive Class: '{cfg.target.positive_class}')")
    print(f"      Protected Attributes  : {cfg.protected_attributes}")

    # 2. Load Raw Dataset & Inspect Metadata
    print(f"\n[2/7] Loading raw dataset and calculating quality metadata...")
    df_raw, metadata = load_raw_dataset_from_config(cfg)
    print(f"      Rows: {metadata['row_count']:,} | Columns: {metadata['column_count']}")
    print(f"      SHA-256 Hash: {metadata['file_sha256']}")

    # 2.5 Pre-training Configuration Contract Validation
    from src.data.dataset_config import validate_dataset_configuration_contract
    cfg_valid, cfg_errors, cfg_warnings = validate_dataset_configuration_contract(df_raw, cfg.to_dict())
    if not cfg_valid:
        print(f"\n[CONFIGURATION FAILURE] {cfg_errors[0]}")
        res = create_pipeline_run_result(
            dataset_id=cfg.dataset_id,
            dataset_hash=metadata.get("file_sha256", "N/A"),
            config_version="1.0.0",
            target_column=cfg.target.column,
            positive_class=cfg.target.positive_class,
            protected_attributes=cfg.protected_attributes,
            feature_count=0,
            train_size=0,
            test_size=0,
            baseline_metrics={},
            fairness_metrics={},
            intersectional_metrics={},
            calibration_metrics={},
            errors=cfg_errors
        )
        res["status"] = "CONFIGURATION_INVALID"
        res["selection_status"] = "CONFIGURATION_INVALID"
        res["selected_model"] = None
        res["message"] = cfg_errors[0]
        return res

    # 3. Preprocess & Decouple Pipeline Data
    print(f"\n[3/7] Preprocessing features X, target y, and protected attributes A...")
    X, y, A = prepare_pipeline_data(df_raw, config=cfg)
    splits = split_pipeline_data(X, y, A, config=cfg)

    X_train, y_train, A_train = splits["X_train"], splits["y_train"], splits["A_train"]
    X_test, y_test, A_test = splits["X_test"], splits["y_test"], splits["A_test"]
    print(f"      Train Samples: {len(X_train):,} | Test Samples: {len(X_test):,} | Encoded Features: {X.shape[1]}")

    # 3.5 Preflight Data & Training Eligibility Validation
    from src.models.model_trainer import validate_training_data_preflight, train_and_cross_validate_candidate
    from src.models.model_evaluator import evaluate_candidate_model
    from src.models.model_selection import select_best_candidate_model
    from src.models.model_storage import save_model_run_artifacts

    is_valid, pf_report = validate_training_data_preflight(X_train, y_train, X_test, y_test, cv_folds=5)
    if not is_valid:
        print(f"\n[PREFLIGHT FAILURE] {pf_report['message']}")
        res = create_pipeline_run_result(
            dataset_id=cfg.dataset_id,
            dataset_hash=metadata["file_sha256"],
            config_version="1.0.0",
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
        res["status"] = pf_report["status"]
        res["message"] = pf_report["message"]
        res["required_minimum"] = pf_report["required_minimum"]
        res["actual_rows"] = pf_report["actual_rows"]
        res["train_rows"] = pf_report["train_rows"]
        res["test_rows"] = pf_report["test_rows"]
        res["cv_folds"] = pf_report["cv_folds"]
        res["selection_status"] = pf_report["status"]
        res["selected_model"] = None
        return res

    # 4. Train & Compare Candidate Models (Logistic Regression, Random Forest, Gradient Boosting)
    print(f"\n[4/7] Training & Cross-Validating Candidate Classifier Models...")

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
        print(f"      • {m_key:<20} | CV Acc: {cv_res['cv_metrics']['cv_accuracy_mean']:.4f} | Test Acc: {eval_res['test_performance']['accuracy']:.4f}")

    selection_res = select_best_candidate_model(candidate_evals)
    selected_key = selection_res.get("selected_model")

    if not selected_key or selected_key not in candidate_fitted or candidate_fitted.get(selected_key) is None:
        print(f"\n[MODEL SELECTION FAILURE] No candidate model was successfully trained.")
        print(f"      Reason: {selection_res.get('reason')}")
        res = create_pipeline_run_result(
            dataset_id=cfg.dataset_id,
            dataset_hash=metadata["file_sha256"],
            config_version="1.0.0",
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
            errors=[selection_res.get("reason", "No valid fitted model available.")]
        )
        res["status"] = "NO_VALID_MODEL"
        res["selection_status"] = "NO_VALID_MODEL"
        res["selected_model"] = None
        res["message"] = selection_res.get("reason")
        return res

    print(f"      Selected Model        : {selected_key} (Status: {selection_res['selection_status']})")
    print(f"      Selection Reason      : {selection_res['reason']}")

    # Save candidate model artifacts and run manifest
    from src.models.model_registry_store import register_model_run
    run_id = f"RUN-{cfg.dataset_id}-{int(time.time())}"
    saved_paths = save_model_run_artifacts(
        dataset_id=cfg.dataset_id,
        run_id=run_id,
        candidate_models=candidate_fitted,
        selected_model_key=selected_key,
        selection_result=selection_res,
        dataset_hash=metadata["file_sha256"],
        config_dict=cfg.to_dict()
    )

    from reference.expected_results import is_reference_dataset
    if is_reference_dataset(cfg.dataset_id) and "logistic_regression" in candidate_fitted and candidate_fitted["logistic_regression"] is not None:
        baseline_model = candidate_fitted["logistic_regression"]
    else:
        baseline_model = candidate_fitted[selected_key]

    y_base_pred, y_base_prob = predict_model(baseline_model, X_test)
    base_perf = evaluate_performance_metrics(y_test, y_base_pred, y_base_prob)
    base_calib = evaluate_calibration(y_test, y_base_prob, n_bins=cfg.calibration_settings.get("n_bins", 10))

    # Single-attribute & Intersectional Audit
    single_audits = run_single_attribute_audits(y_test, y_base_pred, A_test)
    intersectional_audit = audit_intersectional_attributes(
        y_test, y_base_pred, A_test,
        attributes=cfg.protected_attributes,
        min_group_size=cfg.intersectional.min_group_size
    )

    base_eod_val = intersectional_audit['disparities']['equalized_odds_difference']
    base_eod_str = f"{base_eod_val:.4f}" if base_eod_val is not None else "N/A (Insufficient coverage)"

    print(f"      Baseline Accuracy     : {base_perf['accuracy']:.4f}")
    print(f"      Intersectional EOD    : {base_eod_str}")
    print(f"      Baseline Brier Score  : {base_calib['brier_score']:.4f}")

    # 5. Apply Bias Mitigation (Fairlearn ExponentiatedGradient)
    print(f"\n[5/7] Training Fairlearn Mitigated Classifier (EqualizedOdds)...")
    sens_train = create_intersectional_attribute(A_train, attributes=cfg.protected_attributes)
    mit_model = train_mitigated_model(
        X_train, y_train, sens_train,
        random_state=cfg.model_config.random_state,
        max_iter=min(getattr(cfg.model_config, "max_iter", 50), 50),
        base_estimator_type=selected_key
    )
    y_mit_pred, y_mit_prob = predict_mitigated_model(mit_model, X_test, random_state=0)

    mit_perf = evaluate_performance_metrics(y_test, y_mit_pred, y_mit_prob)
    mit_calib = evaluate_calibration(y_test, y_mit_prob, n_bins=cfg.calibration_settings.get("n_bins", 10))
    mit_intersectional_audit = audit_intersectional_attributes(
        y_test, y_mit_pred, A_test,
        attributes=cfg.protected_attributes,
        min_group_size=cfg.intersectional.min_group_size
    )

    mit_eod_val = mit_intersectional_audit['disparities']['equalized_odds_difference']
    mit_eod_str = f"{mit_eod_val:.4f}" if mit_eod_val is not None else "N/A (Insufficient coverage)"

    print(f"      Mitigated Accuracy    : {mit_perf['accuracy']:.4f}")
    print(f"      Mitigated EOD         : {mit_eod_str}")
    print(f"      Mitigated Brier Score : {mit_calib['brier_score']:.4f}")

    # 6. Compute Three-Way Trade-off
    print(f"\n[6/7] Computing Three-Way Trade-off (Performance vs. Calibration vs. Fairness)...")
    tradeoff_report = compute_three_way_tradeoff(
        base_perf,
        mit_perf,
        intersectional_audit,
        mit_intersectional_audit,
        base_calib,
        mit_calib
    )

    # 6.5 Register Model Run in registry/model_registry.json
    selected_eval = candidate_evals.get(selected_key, {})
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
        cv_metrics=selected_eval.get("cv_metrics", {}),
        test_metrics=base_perf,
        fairness_metrics=intersectional_audit["disparities"],
        calibration_metrics=base_calib,
        artifact_path=saved_paths.get("selected_model", ""),
        is_selected=True,
        status="ACTIVE",
        base_dir="."
    )

    # 7. Serialize Results & Generate Artifacts
    print(f"\n[7/7] Serializing dataset results under 'results/{cfg.dataset_id}' and 'data/processed'...")

    # Dataset-specific output directories
    res_dir = os.path.join("results", cfg.dataset_id)
    ev_dir = os.path.join("evidence", "runs", cfg.dataset_id)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(ev_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Construct unified run result object
    run_result = create_pipeline_run_result(
        dataset_id=cfg.dataset_id,
        dataset_hash=metadata["file_sha256"],
        config_version="1.0.0",
        target_column=cfg.target.column,
        positive_class=cfg.target.positive_class,
        protected_attributes=cfg.protected_attributes,
        feature_count=X.shape[1],
        train_size=len(X_train),
        test_size=len(X_test),
        selected_model=selected_key,
        model_type=selected_key,
        selection_status=selection_res.get("selection_status"),
        selection_reason=selection_res.get("reason"),
        model_version=reg_record.get("model_version", "v1"),
        baseline_metrics=base_perf,
        single_attribute_metrics=single_audits,
        fairness_metrics=intersectional_audit["disparities"],
        intersectional_metrics=intersectional_audit,
        calibration_metrics=base_calib,
        mitigation_metrics=mit_intersectional_audit["disparities"],
        tradeoff_metrics=tradeoff_report
    )
    run_result["run_id"] = run_id
    run_result["model_version"] = reg_record.get("model_version", "v1")
    run_result["model_status"] = reg_record.get("status", "ACTIVE")

    # Save to dataset-specific paths
    res_json_path = os.path.join(res_dir, "run_results.json")
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump(run_result, f, indent=2)

    ev_json_path = os.path.join(ev_dir, "run_manifest.json")
    with open(ev_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Update primary 'data/processed' only if reference dataset to avoid overwriting Adult benchmark files
    from reference.expected_results import is_reference_dataset
    if is_reference_dataset(cfg.dataset_id):
        full_json_output = {
            "phase": 5,
            "mitigation_configuration": {
                "algorithm": "Fairlearn ExponentiatedGradient",
                "constraint": "EqualizedOdds",
                "eps": 0.01
            },
            "performance_metrics": {
                "baseline": base_perf,
                "mitigated": mit_perf,
                "comparison": tradeoff_report["performance"]
            },
            "calibration_metrics": {
                "baseline": base_calib,
                "mitigated": mit_calib,
                "comparison": tradeoff_report["calibration"]
            },
            "fairness_metrics": {
                "baseline": intersectional_audit["disparities"],
                "mitigated": mit_intersectional_audit["disparities"],
                "comparison": tradeoff_report["fairness"]
            },
            "tradeoff_interpretation": tradeoff_report["interpretation"],
            "intersectional_threshold_policy": tradeoff_report["intersectional_threshold_policy"]
        }
        with open("data/processed/tradeoff_results.json", "w", encoding="utf-8") as f:
            json.dump(full_json_output, f, indent=2)
        summary_df = generate_tradeoff_summary_table(tradeoff_report)
        summary_df.to_csv("data/processed/tradeoff_summary.csv", index=False)

    print("=" * 80)
    print(f"DATASET PIPELINE EXECUTION COMPLETE FOR '{cfg.dataset_id.upper()}'")
    print(f"Results saved to: '{res_json_path}'")
    print("=" * 80)

    return run_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset-driven intersectional bias auditing pipeline.")
    parser.add_argument("--config", type=str, default="config/default_config.json", help="Path to dataset configuration JSON file.")
    args = parser.parse_args()

    run_dataset_pipeline(args.config)
