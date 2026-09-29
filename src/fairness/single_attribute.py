"""
Single-Attribute Fairness Auditing Module.

This module evaluates classification predictions against protected attributes
(e.g., sex, race) independently to calculate group-level performance metrics
and fairness disparity measures.

Mathematical Definitions of Metrics:
-------------------------------------
1. Group-Level Metrics (for group g):
   - Sample Count (N): Number of samples belonging to group g.
   - Selection Rate (SR): P(y_pred = 1 | A = g) = count(y_pred=1 in g) / N_g
   - True Positive Rate (TPR / Recall): P(y_pred = 1 | y_true = 1, A = g) = TP_g / (TP_g + FN_g)
   - False Positive Rate (FPR): P(y_pred = 1 | y_true = 0, A = g) = FP_g / (FP_g + TN_g)
   - False Negative Rate (FNR): P(y_pred = 0 | y_true = 1, A = g) = FN_g / (TP_g + FN_g)
   - True Negative Rate (TNR / Specificity): P(y_pred = 0 | y_true = 0, A = g) = TN_g / (TN_g + FP_g)

2. Disparity / Fairness Measures (across groups):
   - Demographic Parity Difference (DPD): max_g(SR_g) - min_g(SR_g)
     Measures the absolute difference in positive selection rates across groups.
   - Disparate Impact Ratio (DIR): min_g(SR_g) / max_g(SR_g)
     Measures the ratio of the lowest group selection rate to the highest group selection rate.
     (Values below 0.80 indicate potential adverse impact under the 80% rule).
   - Equalized Odds Difference (EOD): max( (max_g(TPR_g) - min_g(TPR_g)), (max_g(FPR_g) - min_g(FPR_g)) )
     Measures the maximum disparity across both True Positive Rates and False Positive Rates.
   - Equal Opportunity Difference (EOppD): max_g(TPR_g) - min_g(TPR_g)
     Measures the difference in True Positive Rates (opportunity for qualified candidates).
   - False Positive Rate Difference (FPRD): max_g(FPR_g) - min_g(FPR_g)
     Measures the difference in False Positive Rates across groups.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def calculate_group_metrics(y_true, y_pred, group_mask):
    """
    Calculate performance and selection metrics for a specific subgroup mask.

    Parameters:
        y_true (np.ndarray or pd.Series): Ground truth binary labels.
        y_pred (np.ndarray or pd.Series): Predicted binary labels.
        group_mask (np.ndarray or pd.Series): Boolean mask selecting samples in the group.

    Returns:
        dict: Group metric dictionary containing sample_count, selection_rate,
              TPR, FPR, FNR, TNR, TP, FP, TN, FN.
    """
    y_t = np.asarray(y_true)[group_mask]
    y_p = np.asarray(y_pred)[group_mask]

    n_samples = len(y_t)
    if n_samples == 0:
        return {
            "sample_count": 0,
            "selection_rate": 0.0,
            "true_positive_rate": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "true_negative_rate": 0.0,
            "confusion_matrix": {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        }

    # Selection rate: fraction of positive predictions
    pos_preds = int(np.sum(y_p == 1))
    selection_rate = float(pos_preds / n_samples)

    # Confusion matrix
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Rate calculations with zero-division handling
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "sample_count": int(n_samples),
        "selection_rate": selection_rate,
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "true_negative_rate": tnr,
        "confusion_matrix": {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn)
        }
    }


def audit_single_attribute(y_true, y_pred, protected_series, attribute_name="attribute"):
    """
    Perform a single-attribute fairness audit across all unique groups of a protected attribute.

    Parameters:
        y_true (pd.Series or np.ndarray): Ground truth binary labels.
        y_pred (pd.Series or np.ndarray): Predicted binary labels.
        protected_series (pd.Series or np.ndarray): Values of the protected attribute.
        attribute_name (str): Name of the protected attribute (e.g. 'sex', 'race').

    Returns:
        dict: Complete audit report containing group_metrics, disparities, and metric definitions.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    prot_arr = np.asarray(protected_series)

    unique_groups = np.unique(prot_arr)
    group_results = {}

    for group in unique_groups:
        mask = (prot_arr == group)
        group_results[str(group)] = calculate_group_metrics(y_true_arr, y_pred_arr, mask)

    if len(unique_groups) < 2:
        disparities = {
            "demographic_parity_difference": None,
            "disparate_impact_ratio": None,
            "equalized_odds_difference": None,
            "equal_opportunity_difference": None,
            "false_positive_rate_difference": None
        }
        status = "INSUFFICIENT_GROUPS"
    else:
        # Extract rate lists for disparity calculations
        selection_rates = [g["selection_rate"] for g in group_results.values()]
        tprs = [g["true_positive_rate"] for g in group_results.values()]
        fprs = [g["false_positive_rate"] for g in group_results.values()]

        max_sr = max(selection_rates) if selection_rates else 0.0
        min_sr = min(selection_rates) if selection_rates else 0.0

        max_tpr = max(tprs) if tprs else 0.0
        min_tpr = min(tprs) if tprs else 0.0

        max_fpr = max(fprs) if fprs else 0.0
        min_fpr = min(fprs) if fprs else 0.0

        # Calculate Disparities
        dpd = float(max_sr - min_sr)
        dir_ratio = float(min_sr / max_sr) if max_sr > 0 else 1.0
        equal_opp_diff = float(max_tpr - min_tpr)
        fpr_diff = float(max_fpr - min_fpr)
        equalized_odds_diff = max(equal_opp_diff, fpr_diff)

        disparities = {
            "demographic_parity_difference": dpd,
            "disparate_impact_ratio": dir_ratio,
            "equalized_odds_difference": equalized_odds_diff,
            "equal_opportunity_difference": equal_opp_diff,
            "false_positive_rate_difference": fpr_diff
        }
        status = "AVAILABLE"

    definitions = {
        "demographic_parity_difference": "max(Selection Rate) - min(Selection Rate) across groups.",
        "disparate_impact_ratio": "min(Selection Rate) / max(Selection Rate) across groups.",
        "equalized_odds_difference": "max(Equal Opportunity Difference, False Positive Rate Difference).",
        "equal_opportunity_difference": "max(True Positive Rate) - min(True Positive Rate) across groups.",
        "false_positive_rate_difference": "max(False Positive Rate) - min(False Positive Rate) across groups."
    }

    return {
        "attribute_name": attribute_name,
        "status": status,
        "groups": group_results,
        "disparities": disparities,
        "metric_definitions": definitions
    }


def run_single_attribute_audits(y_true, y_pred, A_df):
    """
    Run single-attribute fairness audits for all protected columns in DataFrame A.

    Parameters:
        y_true (pd.Series or np.ndarray): Ground truth binary labels.
        y_pred (pd.Series or np.ndarray): Predicted binary labels.
        A_df (pd.DataFrame): DataFrame containing protected attribute columns.

    Returns:
        dict: Audit results dictionary keyed by column name in A_df.
    """
    audit_results = {}
    for col in A_df.columns:
        audit_results[col] = audit_single_attribute(y_true, y_pred, A_df[col], attribute_name=col)
    return audit_results
