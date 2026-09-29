"""
Fairness Criteria Definitions & Metadata Module.

Provides formal metadata, mathematical definitions, direction of improvement, valid ranges,
and academic interpretations for supported fairness criteria.
"""

from typing import Any, Dict

CRITERIA_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "demographic_parity": {
        "name": "demographic_parity",
        "display_name": "Demographic Parity Difference",
        "better_direction": "lower",
        "range": "[0.0, 1.0]",
        "threshold_key": "demographic_parity_difference_max",
        "default_threshold": 0.10,
        "required_inputs": ["y_pred", "protected_attributes"],
        "output_field": "demographic_parity_difference",
        "mathematical_definition": "max_g P(y_hat = 1 | g) - min_g P(y_hat = 1 | g)",
        "interpretation": "Measures absolute difference in positive selection rates across groups. Project threshold max = 0.10."
    },
    "disparate_impact": {
        "name": "disparate_impact",
        "display_name": "Disparate Impact Ratio",
        "better_direction": "higher",
        "range": "[0.0, 1.0]",
        "threshold_key": "disparate_impact_min",
        "default_threshold": 0.80,
        "required_inputs": ["y_pred", "protected_attributes"],
        "output_field": "disparate_impact_ratio",
        "mathematical_definition": "min_g P(y_hat = 1 | g) / max_g P(y_hat = 1 | g)",
        "interpretation": "Measures ratio of lowest selection rate to highest selection rate. Project threshold min = 0.80 (80% rule)."
    },
    "equal_opportunity": {
        "name": "equal_opportunity",
        "display_name": "Equal Opportunity Difference (TPR)",
        "better_direction": "lower",
        "range": "[0.0, 1.0]",
        "threshold_key": "equal_opportunity_difference_max",
        "default_threshold": 0.10,
        "required_inputs": ["y_true", "y_pred", "protected_attributes"],
        "output_field": "equal_opportunity_difference",
        "mathematical_definition": "max_g P(y_hat = 1 | y = 1, g) - min_g P(y_hat = 1 | y = 1, g)",
        "interpretation": "Measures disparity in True Positive Rates (Recall) across groups. Project threshold max = 0.10."
    },
    "equalized_odds": {
        "name": "equalized_odds",
        "display_name": "Equalized Odds Difference",
        "better_direction": "lower",
        "range": "[0.0, 1.0]",
        "threshold_key": "equalized_odds_difference_max",
        "default_threshold": 0.10,
        "required_inputs": ["y_true", "y_pred", "protected_attributes"],
        "output_field": "equalized_odds_difference",
        "mathematical_definition": "max( delta TPR, delta FPR ) across groups",
        "interpretation": "Measures maximum difference across True Positive Rates and False Positive Rates. Project threshold max = 0.10."
    },
    "fpr_difference": {
        "name": "fpr_difference",
        "display_name": "False Positive Rate Difference",
        "better_direction": "lower",
        "range": "[0.0, 1.0]",
        "threshold_key": "fpr_difference_max",
        "default_threshold": 0.10,
        "required_inputs": ["y_true", "y_pred", "protected_attributes"],
        "output_field": "false_positive_rate_difference",
        "mathematical_definition": "max_g P(y_hat = 1 | y = 0, g) - min_g P(y_hat = 1 | y = 0, g)",
        "interpretation": "Measures disparity in False Positive Rates across groups. Project threshold max = 0.10."
    }
}


def get_criterion_definition(criterion_name: str) -> Dict[str, Any]:
    """
    Retrieve formal metadata dictionary for a specified criterion.

    Parameters:
        criterion_name (str): Name of the criterion.

    Returns:
        dict: Criterion definition metadata.
    """
    if criterion_name not in CRITERIA_DEFINITIONS:
        raise KeyError(f"Criterion '{criterion_name}' is not supported.")
    return CRITERIA_DEFINITIONS[criterion_name]


def list_supported_criteria() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of all supported criteria definitions."""
    return CRITERIA_DEFINITIONS
