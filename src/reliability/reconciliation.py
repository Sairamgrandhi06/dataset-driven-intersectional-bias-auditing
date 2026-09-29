"""
Recovery Reconciliation Engine Module.

Compares pre-failure expected outputs against post-recovery metrics. Asserting match within 1e-4 tolerance.
Returns PASS or MISMATCH.
"""

from typing import Any, Dict, List, Optional
import numpy as np


def reconcile_recovered_output(
    expected_metrics: Dict[str, Any],
    recovered_metrics: Dict[str, Any],
    tolerance: float = 1e-4
) -> Dict[str, Any]:
    """
    Reconcile recovered output dictionary against pre-failure expected values.

    Returns:
        dict: Reconciliation result containing status ('PASS' or 'MISMATCH'), matched keys, and mismatch details.
    """
    mismatches = []
    matches = []

    for key, exp_val in expected_metrics.items():
        if key not in recovered_metrics:
            mismatches.append(f"Metric '{key}' missing from recovered output")
            continue

        rec_val = recovered_metrics[key]

        if isinstance(exp_val, (int, float)) and isinstance(rec_val, (int, float)):
            diff = abs(float(exp_val) - float(rec_val))
            if diff <= tolerance:
                matches.append(key)
            else:
                mismatches.append(f"Metric '{key}' mismatch: Expected={exp_val:.6f}, Recovered={rec_val:.6f}, Diff={diff:.6f}")
        elif exp_val == rec_val:
            matches.append(key)
        else:
            mismatches.append(f"Field '{key}' mismatch: Expected='{exp_val}', Recovered='{rec_val}'")

    status = "PASS" if len(mismatches) == 0 else "MISMATCH"

    return {
        "status": status,
        "total_compared": len(expected_metrics),
        "matched_count": len(matches),
        "mismatched_count": len(mismatches),
        "matches": matches,
        "mismatches": mismatches
    }
