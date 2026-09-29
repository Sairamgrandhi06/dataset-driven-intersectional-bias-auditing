"""
Independent Intersectional Fairness Reference Implementation.

INDEPENDENCE DECLARATION:
This module is a completely independent, self-contained reference implementation created
to verify the production audit implementation ('src/fairness/intersectional.py').
It does NOT import or delegate to any functions in 'src/'.
"""

import numpy as np
import pandas as pd


def create_reference_intersectional_labels(A_df, attributes=["sex", "race"]):
    """
    Independently construct combined subgroup label strings (e.g. 'Female + Black').

    Parameters:
        A_df (pd.DataFrame): Protected attributes DataFrame containing requested columns.
        attributes (list[str]): List of attribute names to combine.

    Returns:
        pd.Series: Subgroup string label series.
    """
    clean_series = []
    for col in attributes:
        s = A_df[col].astype(str).str.strip()
        clean_series.append(s)

    combined = clean_series[0]
    for s in clean_series[1:]:
        combined = combined + " + " + s

    return combined


def calculate_reference_group_metrics(y_true, y_pred, subgroup_series):
    """
    Calculate independent group-level confusion matrix metrics for each subgroup.

    Parameters:
        y_true (np.ndarray): Binary target ground truth (0 or 1).
        y_pred (np.ndarray): Binary target predictions (0 or 1).
        subgroup_series (pd.Series or np.ndarray): Subgroup identity labels.

    Returns:
        dict: Dictionary mapping subgroup names to group metrics.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    subs = np.asarray(subgroup_series)

    unique_subgroups = sorted(list(set(subs)))
    group_metrics = {}

    for group in unique_subgroups:
        mask = (subs == group)
        n_samples = int(np.sum(mask))

        if n_samples == 0:
            continue

        gt = y_t[mask]
        pred = y_p[mask]

        tp = int(np.sum((gt == 1) & (pred == 1)))
        fp = int(np.sum((gt == 0) & (pred == 1)))
        tn = int(np.sum((gt == 0) & (pred == 0)))
        fn = int(np.sum((gt == 1) & (pred == 0)))

        positives = tp + fp
        actual_positives = tp + fn
        actual_negatives = tn + fp

        selection_rate = positives / n_samples
        tpr = tp / actual_positives if actual_positives > 0 else 0.0
        fpr = fp / actual_negatives if actual_negatives > 0 else 0.0
        fnr = fn / actual_positives if actual_positives > 0 else 0.0
        tnr = tn / actual_negatives if actual_negatives > 0 else 0.0

        group_metrics[group] = {
            "n_samples": n_samples,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "selection_rate": round(selection_rate, 6),
            "tpr": round(tpr, 6),
            "fpr": round(fpr, 6),
            "fnr": round(fnr, 6),
            "tnr": round(tnr, 6)
        }

    return group_metrics


def calculate_reference_disparities(group_metrics, min_group_size=30):
    """
    Calculate independent disparity bounds across primary groups (N >= min_group_size).

    Parameters:
        group_metrics (dict): Dictionary of subgroup metrics.
        min_group_size (int): Minimum sample count threshold for primary groups.

    Returns:
        dict: Disparity metrics dictionary.
    """
    primary_metrics = {g: m for g, m in group_metrics.items() if m["n_samples"] >= min_group_size}

    if not primary_metrics:
        primary_metrics = group_metrics

    sr_vals = [m["selection_rate"] for m in primary_metrics.values()]
    tpr_vals = [m["tpr"] for m in primary_metrics.values()]
    fpr_vals = [m["fpr"] for m in primary_metrics.values()]

    max_sr, min_sr = max(sr_vals), min(sr_vals)
    max_tpr, min_tpr = max(tpr_vals), min(tpr_vals)
    max_fpr, min_fpr = max(fpr_vals), min(fpr_vals)

    dp_diff = float(max_sr - min_sr)
    di_ratio = float(min_sr / max_sr) if max_sr > 0 else 1.0
    tpr_diff = float(max_tpr - min_tpr)
    fpr_diff = float(max_fpr - min_fpr)
    eo_diff = max(tpr_diff, fpr_diff)

    return {
        "demographic_parity_difference": round(dp_diff, 6),
        "disparate_impact_ratio": round(di_ratio, 6),
        "equal_opportunity_difference": round(tpr_diff, 6),
        "equalized_odds_difference": round(eo_diff, 6),
        "false_positive_rate_difference": round(fpr_diff, 6)
    }


def audit_intersectional_reference(y_true, y_pred, A_df, min_group_size=30):
    """
    Independent high-level entry point for intersectional reference audit.

    Parameters:
        y_true (np.ndarray): Binary ground truth.
        y_pred (np.ndarray): Binary predictions.
        A_df (pd.DataFrame): Protected attributes DataFrame.
        min_group_size (int): Safety threshold.

    Returns:
        dict: Complete audit result.
    """
    labels = create_reference_intersectional_labels(A_df)
    group_metrics = calculate_reference_group_metrics(y_true, y_pred, labels)
    disparities = calculate_reference_disparities(group_metrics, min_group_size=min_group_size)

    primary_groups = {g: m for g, m in group_metrics.items() if m["n_samples"] >= min_group_size}
    low_sample_groups = {g: m for g, m in group_metrics.items() if m["n_samples"] < min_group_size}

    return {
        "total_groups_discovered": len(group_metrics),
        "primary_group_count": len(primary_groups),
        "low_sample_group_count": len(low_sample_groups),
        "all_group_metrics": group_metrics,
        "primary_groups": primary_groups,
        "low_sample_groups": low_sample_groups,
        "disparities": disparities
    }
