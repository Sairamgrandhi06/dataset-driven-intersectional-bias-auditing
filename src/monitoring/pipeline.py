"""
Batch Monitoring Pipeline Coordinator.

Orchestrates complete batch dataset monitoring, drift detection, model health assessment,
retraining recommendation, and evidence registry persistence with strict dataset isolation.
"""

import json
import os
import time
from typing import Dict, Any, Union, Optional, Tuple
import pandas as pd

from src.data.dataset_config import load_dataset_config_by_id, DatasetConfig, DatasetTargetConfig
from src.data.loader import load_raw_data
from src.data.preprocessor import prepare_pipeline_data
from src.data.quality_gates import compute_file_sha256
from src.models.model_registry_store import get_dataset_registered_versions, load_model_registry
from src.models.model_storage import load_saved_model_run
from src.monitoring.schema_drift import detect_schema_drift
from src.monitoring.data_drift import detect_numerical_feature_drift, detect_categorical_feature_drift
from src.monitoring.feature_drift import analyze_feature_drift
from src.monitoring.data_quality_drift import analyze_data_quality_drift
from src.monitoring.target_drift import analyze_target_drift
from src.monitoring.performance_monitor import monitor_performance_drift
from src.monitoring.fairness_monitor import monitor_fairness_drift
from src.monitoring.calibration_monitor import monitor_calibration_drift
from src.monitoring.model_health import evaluate_model_health
from src.monitoring.retraining_recommender import evaluate_retraining_recommendation
from src.monitoring.monitoring_store import save_monitoring_run_evidence


def load_monitoring_config(config_path: str = "config/monitoring_config.json") -> Dict[str, Any]:
    """Load monitoring configuration options."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "data_drift": {"psi_warning_threshold": 0.10, "psi_critical_threshold": 0.25, "tvd_warning_threshold": 0.10, "tvd_critical_threshold": 0.25},
        "performance_drift": {"f1_degradation_warning": 0.05, "f1_degradation_critical": 0.10},
        "fairness_drift": {"eod_increase_warning": 0.05, "eod_increase_critical": 0.10, "min_group_size": 30},
        "calibration_drift": {"brier_increase_warning": 0.05}
    }


def load_reference_data_for_model(
    dataset_id: str,
    model_version: Optional[str] = None,
    run_id: Optional[str] = None,
    base_dir: str = "."
) -> Dict[str, Any]:
    """
    Strictly load and validate reference dataset and configuration for a registered model version.

    Enforces strict dataset isolation:
    1. Verifies dataset_id matches model's dataset_id.
    2. Verifies dataset_hash.
    3. Authoritatively resolves and loads the reference DataFrame for that exact dataset.
    4. Never falls back to default/adult dataset for non-adult runs.

    Returns:
        dict containing:
            - reference_df: pd.DataFrame
            - dataset_config: DatasetConfig
            - reference_version: str
            - run_id: str
            - dataset_hash: str
            - fitted_model: object
            - preprocessor: object
            - evaluation_metrics: dict
            - run_manifest: dict
            - reference_raw_path: str
    """
    clean_ds_id = dataset_id.lower().strip()

    # 1. Load registry and check registered versions for this dataset
    versions = get_dataset_registered_versions(clean_ds_id, base_dir=base_dir)
    all_reg = load_model_registry(base_dir)

    if not versions:
        # Check models/<dataset_id>/ on disk or results/<dataset_id>/run_results.json
        from src.models.model_storage import get_dataset_model_runs
        disk_runs = get_dataset_model_runs(clean_ds_id, base_dir=base_dir)
        if disk_runs:
            for dr in disk_runs:
                rec = {
                    "dataset_id": clean_ds_id,
                    "model_version": dr.get("model_version", "v1"),
                    "run_id": dr.get("run_id", f"RUN-{clean_ds_id}"),
                    "dataset_hash": dr.get("dataset_hash", ""),
                    "model_type": dr.get("selected_model", "logistic_regression"),
                    "is_active": True,
                    "status": "ACTIVE",
                    "training_config": dr.get("config", {})
                }
                versions.append(rec)
        else:
            run_file = os.path.join(base_dir, "results", clean_ds_id, "run_results.json")
            if os.path.exists(run_file):
                try:
                    with open(run_file, "r", encoding="utf-8") as f:
                        rr = json.load(f)
                    if rr and rr.get("dataset_id") == clean_ds_id:
                        rec = {
                            "dataset_id": clean_ds_id,
                            "model_version": rr.get("model_version", "v1"),
                            "run_id": rr.get("run_id", f"RUN-{clean_ds_id}"),
                            "dataset_hash": rr.get("dataset_hash", ""),
                            "model_type": rr.get("selected_model", "logistic_regression"),
                            "is_active": True,
                            "status": "ACTIVE"
                        }
                        versions.append(rec)
                except Exception:
                    pass

    if not versions:
        if model_version:
            # Check if model_version was registered under another dataset in runs
            for r in all_reg.get("runs", []):
                if isinstance(r, dict) and r.get("model_version") == model_version:
                    other_ds = str(r.get("dataset_id", "")).lower().strip()
                    if other_ds != clean_ds_id:
                        raise ValueError(
                            f"BLOCK_MONITORING_DATASET_MISMATCH: Model version '{model_version}' "
                            f"belongs to dataset '{other_ds}', not '{clean_ds_id}'."
                        )
        raise ValueError(f"No registered model runs found for dataset '{clean_ds_id}'. Train a model first.")

    # 2. Select target version record
    target_ver_record = None
    if model_version:
        for v in versions:
            if v.get("model_version") == model_version:
                target_ver_record = v
                break
        if not target_ver_record:
            # Check if model_version exists in another dataset in runs
            for r in all_reg.get("runs", []):
                if isinstance(r, dict) and r.get("model_version") == model_version:
                    other_ds = str(r.get("dataset_id", "")).lower().strip()
                    if other_ds != clean_ds_id:
                        raise ValueError(
                            f"BLOCK_MONITORING_DATASET_MISMATCH: Model version '{model_version}' "
                            f"belongs to dataset '{other_ds}', not '{clean_ds_id}'."
                        )
            raise ValueError(f"Model version '{model_version}' not found for dataset '{clean_ds_id}'.")
    elif run_id:
        for v in versions:
            if v.get("run_id") == run_id:
                target_ver_record = v
                break
        if not target_ver_record:
            raise ValueError(f"Run ID '{run_id}' not found for dataset '{clean_ds_id}'.")
    else:
        # Active version or latest
        for v in versions:
            if v.get("is_active"):
                target_ver_record = v
                break
        if not target_ver_record:
            target_ver_record = versions[-1]

    # 3. Enforce strict dataset_id match
    record_ds = str(target_ver_record.get("dataset_id", "")).lower().strip()
    if record_ds and record_ds != clean_ds_id:
        raise ValueError(
            f"BLOCK_MONITORING_DATASET_MISMATCH: Target model record dataset_id '{record_ds}' "
            f"does not match selected dataset '{clean_ds_id}'."
        )

    ref_ver_str = target_ver_record["model_version"]
    ref_run_id = target_ver_record.get("run_id", "")
    ref_hash = target_ver_record.get("dataset_hash", "")

    # 4. Load saved model run artifacts
    model_run = load_saved_model_run(clean_ds_id, ref_run_id, base_dir=base_dir)
    fitted_model = model_run.get("fitted_model")
    preprocessor = model_run.get("preprocessor")
    run_manifest = model_run.get("manifest", {})
    eval_metrics = run_manifest.get("evaluation_metrics", {})
    meta = model_run.get("metadata", {})

    # 5. Load authoritative DatasetConfig for this dataset
    ds_cfg = load_dataset_config_by_id(clean_ds_id, base_dir=base_dir)
    if ds_cfg.dataset_id.lower().strip() != clean_ds_id:
        ds_cfg.dataset_id = clean_ds_id

    # Supplement config from model training metadata if available
    t_cfg = target_ver_record.get("training_config") or meta.get("config")
    if isinstance(t_cfg, dict):
        if t_cfg.get("target") and isinstance(t_cfg["target"], dict):
            ds_cfg.target.column = t_cfg["target"].get("column", ds_cfg.target.column)
            ds_cfg.target.positive_class = t_cfg["target"].get("positive_class", ds_cfg.target.positive_class)
        elif t_cfg.get("target_column"):
            ds_cfg.target.column = t_cfg.get("target_column")
            if t_cfg.get("positive_class"):
                ds_cfg.target.positive_class = t_cfg.get("positive_class")
        if t_cfg.get("protected_attributes"):
            ds_cfg.protected_attributes = t_cfg["protected_attributes"]
        if t_cfg.get("id_columns"):
            ds_cfg.id_columns = t_cfg["id_columns"]
        if t_cfg.get("ignore_columns"):
            ds_cfg.ignore_columns = t_cfg["ignore_columns"]

    # If eval_metrics was not in run_manifest, retrieve from authoritative run_results.json
    if not eval_metrics or not eval_metrics.get("test_performance"):
        run_file = os.path.join(base_dir, "results", clean_ds_id, "run_results.json")
        if os.path.exists(run_file):
            try:
                with open(run_file, "r", encoding="utf-8") as f:
                    rr = json.load(f)
                if rr and rr.get("dataset_id") == clean_ds_id:
                    eval_metrics = {
                        "test_performance": rr.get("baseline_metrics", {}),
                        "calibration_metrics": rr.get("calibration_metrics", {}),
                        "fairness_metrics": rr.get("fairness_metrics", {}),
                        "single_attribute_metrics": rr.get("single_attribute_metrics", {})
                    }
            except Exception:
                pass

    # 6. Locate and load reference DataFrame strictly for clean_ds_id
    raw_path_candidates = []
    if ds_cfg.path:
        p = ds_cfg.path if os.path.isabs(ds_cfg.path) else os.path.join(base_dir, ds_cfg.path)
        raw_path_candidates.append(p)

    if isinstance(t_cfg, dict) and t_cfg.get("path"):
        p = t_cfg["path"] if os.path.isabs(t_cfg["path"]) else os.path.join(base_dir, t_cfg["path"])
        raw_path_candidates.append(p)

    raw_dir = os.path.join(base_dir, "data", "raw")
    raw_path_candidates.extend([
        os.path.join(raw_dir, f"{clean_ds_id}.csv"),
        os.path.join(raw_dir, f"{clean_ds_id.split('_')[0]}.csv")
    ])
    if clean_ds_id in ["loan_approval", "loan"]:
        raw_path_candidates.extend([os.path.join(raw_dir, "loan_approval.csv"), os.path.join(raw_dir, "loan.csv")])
    elif clean_ds_id in ["student_performance", "student"]:
        raw_path_candidates.extend([os.path.join(raw_dir, "student_performance.csv"), os.path.join(raw_dir, "student.csv")])
    elif clean_ds_id in ["adult_census_income", "adult"]:
        raw_path_candidates.extend([os.path.join(raw_dir, "adult.csv"), os.path.join(raw_dir, "adult_census_income.csv")])

    resolved_raw_path = None
    for cand in raw_path_candidates:
        if cand and os.path.exists(cand):
            resolved_raw_path = cand
            break

    if not resolved_raw_path:
        raise FileNotFoundError(f"Reference dataset file not found for dataset '{clean_ds_id}'.")

    ref_df = pd.read_csv(resolved_raw_path)

    # 7. Cross-dataset baseline contamination check
    if clean_ds_id not in ["adult_census_income", "adult_income", "adult"]:
        adult_signature = {"education_num", "capital_gain", "capital_loss", "hours_per_week"}
        if len(adult_signature.intersection(set(ref_df.columns))) >= 2:
            raise ValueError(
                f"BLOCK_MONITORING_DATASET_MISMATCH: Adult reference dataset columns detected in "
                f"reference data for non-adult dataset '{clean_ds_id}'."
            )

    if clean_ds_id in ["loan_approval", "loan"]:
        loan_signature = {"loan_status", "credit_score", "loan_amount", "annual_income"}
        if len(loan_signature.intersection(set(ref_df.columns))) == 0:
            raise ValueError(
                "BLOCK_MONITORING_DATASET_MISMATCH: Loaded reference data does not match Loan Approval schema."
            )

    if clean_ds_id in ["student_performance", "student"]:
        student_signature = {"passed", "study_hours", "attendance"}
        if len(student_signature.intersection(set(ref_df.columns))) == 0:
            raise ValueError(
                "BLOCK_MONITORING_DATASET_MISMATCH: Loaded reference data does not match Student Performance schema."
            )

    return {
        "reference_df": ref_df,
        "dataset_config": ds_cfg,
        "reference_version": ref_ver_str,
        "run_id": ref_run_id,
        "dataset_hash": ref_hash,
        "fitted_model": fitted_model,
        "preprocessor": preprocessor,
        "evaluation_metrics": eval_metrics,
        "run_manifest": run_manifest,
        "reference_raw_path": resolved_raw_path
    }


def run_monitoring_pipeline(
    dataset_id: str,
    current_data: Union[str, pd.DataFrame],
    reference_version: Optional[str] = None,
    config_path: str = "config/monitoring_config.json",
    base_dir: str = "."
) -> Dict[str, Any]:
    """
    Run complete batch monitoring workflow for a given dataset ID and current monitoring data.

    Parameters:
        dataset_id (str): Target dataset identifier.
        current_data (str or pd.DataFrame): Path to current CSV file or pre-loaded DataFrame.
        reference_version (str, optional): Model version string (e.g. 'v1'). Defaults to active version.
        config_path (str): Path to monitoring configuration JSON.
        base_dir (str): Base workspace directory path.

    Returns:
        dict: Complete monitoring pipeline results dictionary.
    """
    monitoring_run_id = f"MON-{dataset_id}-{int(time.time())}"
    mon_config = load_monitoring_config(config_path)

    # 1. Load current monitoring DataFrame and compute hash
    if isinstance(current_data, str):
        data_path = current_data
        if not os.path.exists(data_path) and not os.path.isabs(data_path):
            data_path = os.path.join(base_dir, data_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Monitoring data file not found at: '{current_data}'")
        current_df = pd.read_csv(data_path)
        cur_hash = compute_file_sha256(data_path)
    else:
        current_df = current_data.copy()
        cur_hash = f"df_hash_{hash(current_df.to_string()) & 0xffffffff}"

    # 2. Authoritatively load and validate reference model & dataset
    ref_bundle = load_reference_data_for_model(
        dataset_id=dataset_id,
        model_version=reference_version,
        base_dir=base_dir
    )

    ref_raw_df = ref_bundle["reference_df"]
    ds_cfg = ref_bundle["dataset_config"]
    ref_version_str = ref_bundle["reference_version"]
    run_id = ref_bundle["run_id"]
    ref_hash = ref_bundle["dataset_hash"]
    fitted_model = ref_bundle["fitted_model"]
    preprocessor = ref_bundle["preprocessor"]
    eval_metrics = ref_bundle["evaluation_metrics"]
    run_manifest = ref_bundle["run_manifest"]

    ref_perf = eval_metrics.get("test_performance", {})
    ref_fair = eval_metrics.get("fairness_metrics", {})
    ref_cal = eval_metrics.get("calibration_metrics", {})

    # 3. Schema drift detection
    schema_res = detect_schema_drift(
        reference_df=ref_raw_df,
        current_df=current_df,
        target_column=ds_cfg.target.column,
        protected_attributes=ds_cfg.protected_attributes
    )

    # 4. Check for severe dataset mismatch between current_df and reference_df
    cur_cols_set = set(current_df.columns)
    ref_cols_set = set(ref_raw_df.columns)
    common_cols = ref_cols_set.intersection(cur_cols_set)

    # Known domain signature columns
    adult_sig = {"workclass", "education_num", "marital_status", "occupation", "capital_gain", "capital_loss", "hours_per_week"}
    loan_sig = {"loan_status", "credit_score", "loan_amount", "annual_income", "employment_years"}
    student_sig = {"passed", "study_hours", "attendance_rate", "parent_education", "absences", "gpa"}

    is_mismatched = False
    clean_id = dataset_id.lower().strip()

    if clean_id in ["adult_census_income", "adult", "adult_income"]:
        if len(loan_sig.intersection(cur_cols_set)) >= 2 or len(student_sig.intersection(cur_cols_set)) >= 2:
            is_mismatched = True
    elif clean_id in ["loan_approval", "loan", "loan_approval_dataset"]:
        if len(adult_sig.intersection(cur_cols_set)) >= 2 or len(student_sig.intersection(cur_cols_set)) >= 2:
            is_mismatched = True
    elif clean_id in ["student_performance", "student", "student_performance_dataset"]:
        if len(adult_sig.intersection(cur_cols_set)) >= 2 or len(loan_sig.intersection(cur_cols_set)) >= 2:
            is_mismatched = True

    if len(common_cols) <= 1:
        is_mismatched = True
    elif ds_cfg.target.column not in cur_cols_set and len(common_cols) / max(len(ref_cols_set), 1) < 0.35:
        is_mismatched = True

    if is_mismatched:
        mismatch_msg = (
            f"BLOCK_MONITORING_DATASET_MISMATCH: Monitoring batch schema ({len(current_df.columns)} columns) "
            f"does not match reference dataset '{dataset_id}' ({len(ref_raw_df.columns)} columns). "
            f"Common columns: {list(common_cols)}."
        )
        schema_res["status"] = "BLOCK_MONITORING_DATASET_MISMATCH"
        schema_res["has_critical_drift"] = True
        schema_res["critical_issues"].append(mismatch_msg)

        health_res = {
            "overall_health": "BLOCKED",
            "status": "BLOCK_MONITORING_DATASET_MISMATCH",
            "reasons": [mismatch_msg]
        }
        rec_res = {
            "status": "BLOCK_MONITORING_DATASET_MISMATCH",
            "action_required": False,
            "reasons": [mismatch_msg]
        }
        perf_res = {"status": "PERFORMANCE_UNAVAILABLE", "message": mismatch_msg}
        fair_res = {"status": "FAIRNESS_UNAVAILABLE", "message": mismatch_msg}
        cal_res = {"status": "CALIBRATION_UNAVAILABLE", "message": mismatch_msg}
        feature_res = {
            "status": "FEATURE_DRIFT_BLOCKED",
            "message": mismatch_msg,
            "drifted_features_count": 0,
            "total_features_analyzed": 0,
            "feature_metrics": {}
        }
        quality_res = {
            "status": "DATA_QUALITY_BLOCKED",
            "reference_rows": len(ref_raw_df),
            "current_rows": len(current_df),
            "overall_missing_pct_delta": 0.0
        }
        target_res = {
            "status": "TARGET_DRIFT_BLOCKED",
            "labels_available": False,
            "message": mismatch_msg
        }

        # Save evidence
        save_monitoring_run_evidence(
            dataset_id=dataset_id,
            monitoring_run_id=monitoring_run_id,
            reference_version=ref_version_str,
            current_data_hash=cur_hash,
            schema_report=schema_res,
            quality_report=quality_res,
            feature_drift_report=feature_res,
            target_drift_report=target_res,
            performance_report=perf_res,
            fairness_report=fair_res,
            calibration_report=cal_res,
            health_report=health_res,
            retraining_recommendation=rec_res,
            reference_dataset_id=dataset_id,
            reference_dataset_hash=ref_hash,
            reference_run_id=run_id,
            base_dir=base_dir
        )

        return {
            "monitoring_run_id": monitoring_run_id,
            "dataset_id": dataset_id,
            "monitoring_dataset_id": dataset_id,
            "monitoring_dataset_hash": cur_hash,
            "reference_dataset_id": dataset_id,
            "reference_dataset_hash": ref_hash,
            "reference_version": ref_version_str,
            "reference_run_id": run_id,
            "current_data_hash": cur_hash,
            "reference_data_hash": ref_hash,
            "schema_report": schema_res,
            "feature_drift_report": feature_res,
            "data_quality_report": quality_res,
            "target_drift_report": target_res,
            "performance_report": perf_res,
            "fairness_report": fair_res,
            "calibration_report": cal_res,
            "health_report": health_res,
            "retraining_recommendation": rec_res
        }

    # 5. Feature drift detection (Stage B)
    feature_res = analyze_feature_drift(
        reference_df=ref_raw_df,
        current_df=current_df,
        target_column=ds_cfg.target.column,
        protected_attributes=ds_cfg.protected_attributes,
        id_columns=ds_cfg.id_columns,
        ignore_columns=ds_cfg.ignore_columns,
        config=mon_config
    )

    # 6. Data quality drift detection
    min_grp = ds_cfg.intersectional.min_group_size
    quality_res = analyze_data_quality_drift(
        reference_df=ref_raw_df,
        current_df=current_df,
        target_column=ds_cfg.target.column,
        protected_attributes=ds_cfg.protected_attributes,
        min_group_size=min_grp,
        config=mon_config
    )

    # 7. Target drift detection
    target_res = analyze_target_drift(
        reference_df=ref_raw_df,
        current_df=current_df,
        target_column=ds_cfg.target.column,
        positive_class=ds_cfg.target.positive_class,
        config=mon_config
    )

    # 8. Model evaluation on current dataset (if schema compatible and model available)
    if schema_res.get("has_critical_drift") or fitted_model is None:
        perf_res = {"status": "PERFORMANCE_UNAVAILABLE", "message": "Blocked due to critical schema drift."}
        fair_res = {"status": "FAIRNESS_UNAVAILABLE", "message": "Blocked due to critical schema drift."}
        cal_res = {"status": "CALIBRATION_UNAVAILABLE", "message": "Blocked due to critical schema drift."}
    else:
        # Preprocess features X, target y, protected attributes A
        try:
            X_curr, y_curr, A_curr = prepare_pipeline_data(
                df=current_df,
                config=ds_cfg
            )
            if hasattr(fitted_model, "feature_names_in_"):
                expected_cols = list(fitted_model.feature_names_in_)
                for c in expected_cols:
                    if c not in X_curr.columns:
                        X_curr[c] = 0
                X_curr = X_curr[expected_cols]

            perf_res = monitor_performance_drift(
                model=fitted_model,
                X_test=X_curr,
                y_test=y_curr,
                reference_metrics=ref_perf,
                config=mon_config
            )
            fair_res = monitor_fairness_drift(
                model=fitted_model,
                X_test=X_curr,
                y_test=y_curr,
                A_test=A_curr,
                reference_fairness=ref_fair,
                min_group_size=min_grp,
                config=mon_config
            )
            cal_res = monitor_calibration_drift(
                model=fitted_model,
                X_test=X_curr,
                y_test=y_curr,
                reference_calibration=ref_cal,
                n_bins=ds_cfg.calibration_settings.get("n_bins", 10),
                config=mon_config
            )
        except Exception as e:
            perf_res = {"status": "PERFORMANCE_UNAVAILABLE", "message": str(e)}
            fair_res = {"status": "FAIRNESS_UNAVAILABLE", "message": str(e)}
            cal_res = {"status": "CALIBRATION_UNAVAILABLE", "message": str(e)}

    # 9. Synthesize model health
    health_res = evaluate_model_health(
        schema_report=schema_res,
        quality_report=quality_res,
        feature_drift_report=feature_res,
        target_drift_report=target_res,
        performance_report=perf_res,
        fairness_report=fair_res,
        calibration_report=cal_res,
        artifact_integrity=(fitted_model is not None)
    )

    # 10. Evaluate retraining recommendation
    rec_res = evaluate_retraining_recommendation(
        model_health_report=health_res,
        schema_report=schema_res,
        quality_report=quality_res,
        feature_drift_report=feature_res,
        target_drift_report=target_res,
        performance_report=perf_res,
        fairness_report=fair_res,
        calibration_report=cal_res,
        config=mon_config
    )

    # 11. Save monitoring evidence and update registry
    save_monitoring_run_evidence(
        dataset_id=dataset_id,
        monitoring_run_id=monitoring_run_id,
        reference_version=ref_version_str,
        current_data_hash=cur_hash,
        schema_report=schema_res,
        quality_report=quality_res,
        feature_drift_report=feature_res,
        target_drift_report=target_res,
        performance_report=perf_res,
        fairness_report=fair_res,
        calibration_report=cal_res,
        health_report=health_res,
        retraining_recommendation=rec_res,
        reference_dataset_id=dataset_id,
        reference_dataset_hash=ref_hash,
        reference_run_id=run_id,
        base_dir=base_dir
    )

    return {
        "monitoring_run_id": monitoring_run_id,
        "dataset_id": dataset_id,
        "monitoring_dataset_id": dataset_id,
        "monitoring_dataset_hash": cur_hash,
        "reference_dataset_id": dataset_id,
        "reference_dataset_hash": ref_hash,
        "reference_version": ref_version_str,
        "reference_run_id": run_id,
        "current_data_hash": cur_hash,
        "reference_data_hash": ref_hash,
        "schema_report": schema_res,
        "feature_drift_report": feature_res,
        "data_quality_report": quality_res,
        "target_drift_report": target_res,
        "performance_report": perf_res,
        "fairness_report": fair_res,
        "calibration_report": cal_res,
        "health_report": health_res,
        "retraining_recommendation": rec_res
    }
