"""
Model Registry Module.

Provides factory functions to register, instantiate, and configure candidate binary
classification models for dataset-specific training.
"""

from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


SUPPORTED_MODEL_KEYS = ["logistic_regression", "random_forest", "gradient_boosting"]


def get_candidate_model(model_key: str, random_seed: int = 42) -> Any:
    """
    Instantiate a scikit-learn estimator for the specified candidate model key.

    Parameters:
        model_key (str): Candidate identifier ('logistic_regression', 'random_forest', 'gradient_boosting').
        random_seed (int): Seed for reproducibility.

    Returns:
        BaseEstimator: Configured scikit-learn classifier model instance.
    """
    key_clean = str(model_key).lower().strip().replace(" ", "_").replace("-", "_")
    if key_clean in ["logistic_regression", "logisticregression", "logistic"]:
        return LogisticRegression(max_iter=1000, random_state=random_seed)
    elif key_clean in ["random_forest", "randomforest", "rf", "randomforestclassifier"]:
        return RandomForestClassifier(n_estimators=100, random_state=random_seed)
    elif key_clean in ["gradient_boosting", "gradientboosting", "gb", "gradientboostingclassifier"]:
        return GradientBoostingClassifier(n_estimators=100, random_state=random_seed)
    else:
        raise ValueError(
            f"Unsupported model key '{model_key}'. Supported keys are: {SUPPORTED_MODEL_KEYS}"
        )


def list_supported_models() -> List[str]:
    """Return list of supported candidate model key names."""
    return list(SUPPORTED_MODEL_KEYS)
