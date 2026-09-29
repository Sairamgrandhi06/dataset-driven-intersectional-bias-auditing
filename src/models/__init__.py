"""Model training and classification modules."""

from src.models.classifier import (
    train_baseline_model,
    predict_model,
    evaluate_performance_metrics
)
from src.models.model_registry_store import (
    load_model_registry,
    save_model_registry,
    register_model_run,
    get_dataset_registered_versions,
    set_active_model_version,
    generate_run_id
)
from src.models.retraining import retrain_dataset

__all__ = [
    "train_baseline_model",
    "predict_model",
    "evaluate_performance_metrics",
    "load_model_registry",
    "save_model_registry",
    "register_model_run",
    "get_dataset_registered_versions",
    "set_active_model_version",
    "generate_run_id",
    "retrain_dataset"
]
