"""Contracts subpackage."""
from src.contracts.error_contract import create_structured_error, CriteriaError
from src.contracts.fairness_criteria_contract import validate_criteria_input_contract
from src.contracts.criteria_output_contract import build_criteria_output_contract

__all__ = [
    "create_structured_error",
    "CriteriaError",
    "validate_criteria_input_contract",
    "build_criteria_output_contract"
]
