"""
Data & Feature Drift Detection Module.

Calculates Population Stability Index (PSI), Wasserstein Distance, Total Variation Distance (TVD),
and categorical distribution shifts between reference and current monitoring datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.stats import wasserstein_distance


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Calculate Population Stability Index (PSI) between reference and current numerical arrays.

    Parameters:
        reference (np.ndarray): Reference feature array.
        current (np.ndarray): Current feature array.
        num_bins (int): Bin count.
        epsilon (float): Small constant to avoid zero-division and log(0).

    Returns:
        float: Calculated PSI value.
    """
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    # If feature is constant in reference
    if np.min(ref_clean) == np.max(ref_clean):
        if np.array_equal(ref_clean, cur_clean):
            return 0.0
        return 1.0

    # Determine quantile bin edges from reference distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, percentiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        bin_edges = np.linspace(np.min(ref_clean), np.max(ref_clean), num_bins + 1)

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

    ref_pct = ref_counts / len(ref_clean)
    cur_pct = cur_counts / len(cur_clean)

    # Safe zero smoothing
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(np.maximum(0.0, psi_val))


def calculate_tvd(
    ref_counts: Dict[Any, int],
    cur_counts: Dict[Any, int]
) -> float:
    """
    Calculate Total Variation Distance (TVD) between reference and current categorical distributions.

    Parameters:
        ref_counts (dict): Reference category frequencies.
        cur_counts (dict): Current category frequencies.

    Returns:
        float: TVD metric (0.0 to 1.0).
    """
    total_ref = sum(ref_counts.values())
    total_cur = sum(cur_counts.values())

    if total_ref == 0 or total_cur == 0:
        return 0.0

    all_cats = set(ref_counts.keys()).union(set(cur_counts.keys()))

    tvd = 0.5 * sum(
        abs((cur_counts.get(c, 0) / total_cur) - (ref_counts.get(c, 0) / total_ref))
        for c in all_cats
    )
    return float(tvd)


def detect_numerical_feature_drift(
    ref_series: pd.Series,
    cur_series: pd.Series,
    psi_warning: float = 0.10,
    psi_critical: float = 0.25
) -> Dict[str, Any]:
    """Calculate numerical feature drift metrics (PSI, Wasserstein distance, mean/std shift)."""
    ref_clean = ref_series.dropna().to_numpy()
    cur_clean = cur_series.dropna().to_numpy()

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return {
            "drift_score": 0.0,
            "status": "NO_DRIFT",
            "wasserstein_distance": 0.0,
            "ref_mean": 0.0,
            "cur_mean": 0.0
        }

    psi = calculate_psi(ref_clean, cur_clean)
    w_dist = float(wasserstein_distance(ref_clean, cur_clean))

    ref_mean = float(np.mean(ref_clean))
    cur_mean = float(np.mean(cur_clean))
    ref_std = float(np.std(ref_clean))
    cur_std = float(np.std(cur_clean))

    if psi >= psi_critical:
        status = "DRIFT_DETECTED"
    elif psi >= psi_warning:
        status = "WARNING"
    else:
        status = "NO_DRIFT"

    return {
        "drift_score": psi,
        "status": status,
        "wasserstein_distance": w_dist,
        "ref_mean": ref_mean,
        "cur_mean": cur_mean,
        "ref_std": ref_std,
        "cur_std": cur_std,
        "mean_shift": abs(cur_mean - ref_mean)
    }


def detect_categorical_feature_drift(
    ref_series: pd.Series,
    cur_series: pd.Series,
    tvd_warning: float = 0.10,
    tvd_critical: float = 0.25
) -> Dict[str, Any]:
    """Calculate categorical feature drift metrics (TVD, distribution shift, new/missing categories)."""
    ref_counts = ref_series.dropna().value_counts().to_dict()
    cur_counts = cur_series.dropna().value_counts().to_dict()

    tvd = calculate_tvd(ref_counts, cur_counts)

    ref_cats = set(ref_counts.keys())
    cur_cats = set(cur_counts.keys())

    new_cats = sorted([str(c) for c in (cur_cats - ref_cats)])
    missing_cats = sorted([str(c) for c in (ref_cats - cur_cats)])

    if tvd >= tvd_critical or len(new_cats) > 0:
        status = "DRIFT_DETECTED"
    elif tvd >= tvd_warning:
        status = "WARNING"
    else:
        status = "NO_DRIFT"

    return {
        "drift_score": tvd,
        "status": status,
        "tvd": tvd,
        "new_categories": new_cats,
        "missing_categories": missing_cats,
        "ref_category_count": len(ref_cats),
        "cur_category_count": len(cur_cats)
    }
