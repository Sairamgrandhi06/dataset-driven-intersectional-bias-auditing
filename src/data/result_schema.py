"""
Result Schema Module.

Defines a standardized, dataset-independent result schema for execution outputs across
all phases of the Intersectional Bias Auditing framework.
"""

from typing import Dict, Any, List, Optional
import time
from datetime import datetime, timezone


def create_pipeline_run_result(
    dataset_id: str,
    dataset_hash: str,
    config_version: str,
    target_column: str,
    positive_class: Any,
    protected_attributes: List[str],
    feature_count: int,
    train_size: int,
    test_size: int,
    baseline_metrics: Dict[str, Any],
    fairness_metrics: Dict[str, Any],
    intersectional_metrics: Dict[str, Any],
    calibration_metrics: Dict[str, Any],
    single_attribute_metrics: Optional[Dict[str, Any]] = None,
    mitigation_metrics: Optional[Dict[str, Any]] = None,
    tradeoff_metrics: Optional[Dict[str, Any]] = None,
    selected_model: Optional[str] = None,
    model_type: Optional[str] = None,
    selection_status: Optional[str] = None,
    selection_reason: Optional[str] = None,
    model_version: Optional[str] = None,
    warnings: Optional[List[str]] = None,
    errors: Optional[List[str]] = None,
    execution_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Construct a dataset-independent pipeline run result dictionary matching the standard schema.
    """
    run_id = f"RUN-{dataset_id}-{int(time.time())}"
    timestamp = datetime.now(timezone.utc).isoformat()
    sel_m = selected_model or model_type

    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "config_version": config_version,
        "target_column": target_column,
        "positive_class": str(positive_class),
        "protected_attributes": protected_attributes,
        "feature_count": feature_count,
        "train_size": train_size,
        "test_size": test_size,
        "selected_model": sel_m,
        "model_type": sel_m,
        "selection_status": selection_status or ("COMPLETED" if sel_m else None),
        "selection_reason": selection_reason,
        "model_version": model_version or "v1",
        "baseline_model": {
            "algorithm": sel_m,
            "version": model_version or "v1",
            "selection_status": selection_status
        } if sel_m else {},
        "baseline_metrics": baseline_metrics,
        "single_attribute_metrics": single_attribute_metrics or {},
        "fairness_metrics": fairness_metrics,
        "intersectional_metrics": intersectional_metrics,
        "calibration_metrics": calibration_metrics,
        "mitigation_metrics": mitigation_metrics or {},
        "tradeoff_metrics": tradeoff_metrics or {},
        "warnings": warnings or [],
        "errors": errors or [],
        "execution_metadata": execution_metadata or {
            "status": "COMPLETED",
            "framework_version": "1.0.0"
        }
    }
