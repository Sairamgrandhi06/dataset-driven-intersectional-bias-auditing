"""
Model Calibration Evaluation Module.

This module evaluates probability calibration for classification models.

Calibration Concepts & Metrics:
-------------------------------
1. Calibration Meaning:
   A model is well-calibrated if predicted probabilities reflect empirical class frequencies.
   For instance, among predictions assigned a probability of 0.80, 80% should be true positives.

2. Brier Score:
   The Brier Score measures the mean squared error between predicted probabilities and ground truth:
   BS = (1/N) * sum((y_prob_i - y_true_i)^2)
   Range: [0.0, 1.0]. LOWER Brier Score indicates BETTER calibration.

3. Expected Calibration Error (ECE):
   ECE computes the weighted average absolute difference between empirical accuracy and predicted confidence
   across probability bins:
   ECE = sum(|B_b| / N * |acc(B_b) - conf(B_b)|)
   Range: [0.0, 1.0]. LOWER ECE indicates BETTER calibration.

4. Impact of Bias Mitigation on Calibration:
   In-processing bias mitigation (sample reweighting or threshold optimization) alters decision boundaries
   to satisfy fairness constraints. This optimization can shift predicted probabilities away from pure
   empirical likelihoods, potentially affecting calibration.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


def calculate_expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE) across uniform probability bins.

    Parameters:
        y_true (np.ndarray or pd.Series): Binary ground truth labels.
        y_prob (np.ndarray or pd.Series): Predicted positive class probabilities.
        n_bins (int): Number of uniform probability bins.

    Returns:
        float: Expected Calibration Error value in range [0.0, 1.0].
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)

    n_samples = len(y_t)
    if n_samples == 0:
        return 0.0

    # Tolerance-aware probability validation (bounds checking with numerical epsilon)
    tol = 1e-4
    if np.any(y_p < -tol) or np.any(y_p > 1.0 + tol):
        raise ValueError(
            f"y_prob contains non-probability values outside [0, 1] tolerance (min={float(np.min(y_p)):.4f}, max={float(np.max(y_p)):.4f})."
        )
    y_p = np.clip(y_p, 0.0, 1.0)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        if i == n_bins - 1:
            in_bin = (y_p >= bin_lower) & (y_p <= bin_upper)
        else:
            in_bin = (y_p >= bin_lower) & (y_p < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_t[in_bin])
            bin_conf = np.mean(y_p[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_calibration(y_true, y_prob, n_bins=10):
    """
    Evaluate Brier Score, ECE, and reliability curve coordinates for prediction probabilities.

    Parameters:
        y_true (pd.Series or np.ndarray): Ground truth binary labels.
        y_prob (pd.Series or np.ndarray): Predicted positive class probabilities.
        n_bins (int): Number of bins for reliability curve and ECE.

    Returns:
        dict: Calibration metrics containing brier_score, expected_calibration_error,
              prob_true, prob_pred, n_bins, and explanations.
    """
    if y_prob is None:
        return {
            "status": "UNAVAILABLE",
            "brier_score": None,
            "ece": None,
            "expected_calibration_error": None,
            "reliability_curve": {"prob_true": [], "prob_pred": []},
            "n_bins": n_bins
        }

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)

    # Tolerance-aware probability validation (allow tiny IEEE-754 rounding noise around bounds)
    tol = 1e-4
    if np.any(y_p < -tol) or np.any(y_p > 1.0 + tol):
        raise ValueError(
            f"y_prob contains non-probability values outside [0, 1] tolerance (min={float(np.min(y_p)):.4f}, max={float(np.max(y_p)):.4f})."
        )
    y_p = np.clip(y_p, 0.0, 1.0)

    bs = float(brier_score_loss(y_t, y_p))
    ece = float(calculate_expected_calibration_error(y_t, y_p, n_bins=n_bins))

    prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=n_bins, strategy="uniform")

    return {
        "status": "AVAILABLE",
        "brier_score": bs,
        "ece": ece,
        "expected_calibration_error": ece,
        "reliability_curve": {
            "prob_true": [float(v) for v in prob_true],
            "prob_pred": [float(v) for v in prob_pred]
        },
        "n_bins": n_bins,
        "explanation": (
            "Brier Score measures mean squared error of predicted probabilities. "
            "Lower values indicate superior probability calibration."
        )
    }


def compare_calibration(baseline_calib, mitigated_calib):
    """
    Compare probability calibration metrics between baseline and mitigated models.

    Parameters:
        baseline_calib (dict): Output of evaluate_calibration for baseline model.
        mitigated_calib (dict): Output of evaluate_calibration for mitigated model.

    Returns:
        dict: Comparison metrics containing baseline, mitigated, delta, pct_change, and impact_status.
    """
    b_bs = baseline_calib["brier_score"]
    m_bs = mitigated_calib["brier_score"]
    delta_bs = m_bs - b_bs
    pct_bs = (delta_bs / abs(b_bs) * 100.0) if abs(b_bs) > 1e-6 else 0.0
    bs_status = "Improved" if delta_bs < 0 else ("Worsened" if delta_bs > 0 else "Neutral")

    b_ece = baseline_calib["expected_calibration_error"]
    m_ece = mitigated_calib["expected_calibration_error"]
    delta_ece = m_ece - b_ece
    pct_ece = (delta_ece / abs(b_ece) * 100.0) if abs(b_ece) > 1e-6 else 0.0
    ece_status = "Improved" if delta_ece < 0 else ("Worsened" if delta_ece > 0 else "Neutral")

    return {
        "brier_score": {
            "baseline": b_bs,
            "mitigated": m_bs,
            "absolute_change": delta_bs,
            "percentage_change": pct_bs,
            "impact_status": bs_status
        },
        "expected_calibration_error": {
            "baseline": b_ece,
            "mitigated": m_ece,
            "absolute_change": delta_ece,
            "percentage_change": pct_ece,
            "impact_status": ece_status
        },
        "interpretation": (
            f"Brier Score changed from {b_bs:.4f} to {m_bs:.4f} (Abs Change: {delta_bs:+.4f}, "
            f"{pct_bs:+.2f}%). Status: {bs_status}."
        )
    }
