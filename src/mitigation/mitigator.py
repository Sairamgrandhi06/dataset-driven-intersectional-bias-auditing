"""
Bias Mitigation and Trade-off Evaluation Module.

This module provides functions to:
1. Train a mitigated classification model using Fairlearn in-processing reductions
   (ExponentiatedGradient) under fairness constraints (EqualizedOdds, DemographicParity).
2. Support real-time execution logging and optimizer iteration tracking.
3. Predict binary labels and class probabilities from the mitigated model.
4. Extract structural and parametric changes made by the mitigation algorithm.
5. Quantify explicit pre vs. post accuracy-fairness trade-offs and group opportunity gains.
6. Evaluate a grid of constraint strictness parameters to construct Pareto trade-off curves.
"""

import time
import warnings
from typing import cast, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from fairlearn.reductions import (
    ExponentiatedGradient,
    EqualizedOdds,
    DemographicParity,
    TruePositiveRateParity
)

from src.fairness.intersectional import audit_intersectional_attributes


def get_fairness_constraint(constraint_type="EqualizedOdds"):
    """
    Instantiate a Fairlearn Moment object for the specified fairness constraint.

    Parameters:
        constraint_type (str): Constraint name ('EqualizedOdds', 'DemographicParity', 'TruePositiveRateParity', etc.).

    Returns:
        fairlearn.reductions.Moment: Fairlearn constraint object.
    """
    c_map = {
        "EqualizedOdds": EqualizedOdds,
        "DemographicParity": DemographicParity,
        "TruePositiveRateParity": TruePositiveRateParity,
        "TruePositiveRateDifference": TruePositiveRateParity
    }

    if constraint_type not in c_map:
        raise ValueError(f"Unsupported constraint type '{constraint_type}'. Must be one of {list(c_map.keys())}")

    return c_map[constraint_type]()


def get_base_estimator(estimator_type="logistic_regression", random_state=42, max_iter=1000):
    """
    Instantiate the base ML classifier for Fairlearn reduction reusing the authoritative
    Step 2 candidate model factory and hyperparameters.
    """
    if hasattr(estimator_type, "fit") and not isinstance(estimator_type, str):
        from sklearn.base import clone
        cand = clone(estimator_type)
        if isinstance(cand, GradientBoostingClassifier):
            cand.set_params(max_features="sqrt")
        return cand

    if estimator_type is None or not str(estimator_type).strip():
        estimator_type = "logistic_regression"

    from src.models.model_registry import get_candidate_model
    cand = get_candidate_model(str(estimator_type), random_seed=random_state)
    if isinstance(cand, LogisticRegression):
        cand.set_params(solver="liblinear")
    elif isinstance(cand, GradientBoostingClassifier):
        cand.set_params(max_features="sqrt")
    return cand


def train_mitigated_model(
    X_train,
    y_train,
    sensitive_features,
    constraint_type="EqualizedOdds",
    eps=0.01,
    random_state=42,
    max_iter=50,
    base_estimator_type="logistic_regression",
    logger_callback=None
):
    """
    Train a mitigated classifier using Fairlearn ExponentiatedGradient reduction.

    Parameters:
        X_train (pd.DataFrame or np.ndarray): Training feature matrix.
        y_train (pd.Series or np.ndarray): Training binary target vector.
        sensitive_features (pd.Series, pd.DataFrame, or np.ndarray): Sensitive attribute(s).
        constraint_type (str): Fairness constraint ('EqualizedOdds', 'DemographicParity', 'TruePositiveRateParity').
        eps (float): Maximum allowed constraint violation bound.
        random_state (int): Seed for reproducibility.
        max_iter (int): Maximum iterations for reduction optimization.
        base_estimator_type (str): Base estimator type.
        logger_callback (callable, optional): Callback function f(msg) for real-time progress logging.

    Returns:
        ExponentiatedGradient: Trained Fairlearn mitigated model object.
    """
    constraint = get_fairness_constraint(constraint_type)
    base_estimator = get_base_estimator(base_estimator_type, random_state=random_state)

    if logger_callback:
        logger_callback(f"[INIT] Initializing Fairlearn ExponentiatedGradient reduction...")
        logger_callback(f"[CONFIG] Constraint: {constraint_type} | Tolerance (eps): {eps} | Max Iterations: {max_iter}")
        logger_callback(f"[CONFIG] Base Estimator: {base_estimator.__class__.__name__} | Random Seed: {random_state}")

    t0 = time.time()

    mitigated_model = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        eps=eps,
        max_iter=max_iter
    )

    X_train_arr = np.ascontiguousarray(X_train, dtype=np.float32) if isinstance(X_train, (pd.DataFrame, np.ndarray)) else X_train

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        mitigated_model.fit(X_train_arr, y_train, sensitive_features=sensitive_features)

    elapsed = time.time() - t0

    if logger_callback:
        n_preds = len(getattr(mitigated_model, "predictors_", []))
        weights = getattr(mitigated_model, "weights_", [])
        best_gap = getattr(mitigated_model, "best_gap_", None)
        gap_str = f"{best_gap:.5f}" if best_gap is not None else "N/A"

        logger_callback(f"[SOLVER] Duality optimization finished in {elapsed:.3f}s.")
        logger_callback(f"[ENSEMBLE] Generated {n_preds} candidate sub-predictors.")
        if len(weights) > 0:
            active_weights = [round(float(w), 4) for w in weights if w > 0.001]
            logger_callback(f"[WEIGHTS] Active predictor weights: {active_weights} (sum={sum(weights):.4f})")
        logger_callback(f"[CONVERGENCE] Final Best Duality Gap: {gap_str} (Tolerance Bound: {eps})")
        logger_callback(f"[SUCCESS] Fairlearn mitigation model successfully trained and ready for evaluation.")

    return mitigated_model


def train_mitigated_model_with_logging(
    X_train,
    y_train,
    sensitive_features,
    constraint_type="EqualizedOdds",
    eps=0.01,
    random_state=42,
    max_iter=50,
    base_estimator_type="logistic_regression",
    logger_callback=None
):
    """
    Train a mitigated model and collect structured execution logs for UI rendering.

    Returns:
        tuple: (mitigated_model, log_messages)
    """
    log_messages = []

    def internal_logger(msg):
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {msg}"
        log_messages.append(formatted)
        if logger_callback:
            logger_callback(formatted)

    internal_logger(f"Starting In-Processing Bias Mitigation Pipeline on {len(X_train)} training records.")
    
    # Sensitive attributes analysis
    if isinstance(sensitive_features, (pd.Series, pd.DataFrame)):
        if isinstance(sensitive_features, pd.DataFrame):
            unique_groups = sensitive_features.drop_duplicates().to_dict(orient="records")
            n_groups = len(unique_groups)
        else:
            n_groups = sensitive_features.nunique()
        internal_logger(f"Detected {n_groups} distinct sensitive group intersections in training set.")

    model = train_mitigated_model(
        X_train=X_train,
        y_train=y_train,
        sensitive_features=sensitive_features,
        constraint_type=constraint_type,
        eps=eps,
        random_state=random_state,
        max_iter=max_iter,
        base_estimator_type=base_estimator_type,
        logger_callback=internal_logger
    )

    return model, log_messages


def predict_mitigated_model(mitigated_model, X_test, random_state=0):
    """
    Generate binary predictions and probabilities from a trained mitigated model.

    Parameters:
        mitigated_model: Trained Fairlearn ExponentiatedGradient or fitted predictor.
        X_test (pd.DataFrame or np.ndarray): Test feature matrix.
        random_state (int or None): Random seed for deterministic prediction sampling.

    Returns:
        tuple: (y_pred, y_prob) where:
            - y_pred (np.ndarray): Binary predicted class labels (0 or 1).
            - y_prob (np.ndarray): Predicted positive class probabilities.
    """
    if random_state is not None:
        np.random.seed(random_state)

    y_pred = mitigated_model.predict(X_test)

    y_prob = None
    if hasattr(mitigated_model, "_pmf_predict"):
        try:
            pmf = mitigated_model._pmf_predict(X_test)
            y_prob = pmf[:, 1] if (isinstance(pmf, np.ndarray) and pmf.ndim > 1) else pmf
        except Exception:
            y_prob = None

    if y_prob is None and hasattr(mitigated_model, "predict_proba"):
        try:
            prob = mitigated_model.predict_proba(X_test)
            y_prob = prob[:, 1] if (isinstance(prob, np.ndarray) and prob.ndim > 1) else prob
        except Exception:
            y_prob = None

    if y_prob is None and hasattr(mitigated_model, "_pmf"):
        try:
            pmf = mitigated_model._pmf(X_test)
            y_prob = pmf[:, 1] if (isinstance(pmf, np.ndarray) and pmf.ndim > 1) else pmf
        except Exception:
            y_prob = None

    if y_prob is None:
        if hasattr(mitigated_model, "predictors_") and hasattr(mitigated_model, "weights_"):
            try:
                y_prob = np.zeros(len(X_test), dtype=float)
                for w, p in zip(mitigated_model.weights_, mitigated_model.predictors_):
                    if hasattr(p, "predict_proba"):
                        y_prob += w * p.predict_proba(X_test)[:, 1]
                    else:
                        y_prob += w * p.predict(X_test)
            except Exception:
                y_prob = y_pred.astype(float)
        else:
            y_prob = y_pred.astype(float)

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        tol = 1e-4
        if not (np.any(y_prob < -tol) or np.any(y_prob > 1.0 + tol)):
            y_prob = np.clip(y_prob, 0.0, 1.0)

    return y_pred, y_prob


def train_threshold_optimizer(
    base_estimator,
    X_train,
    y_train,
    sensitive_features,
    constraint="equalized_odds",
    predict_method="auto",
    logger_callback=None
):
    """
    Train a post-processing ThresholdOptimizer on fitted or unfitted base estimator.
    """
    from fairlearn.postprocessing import ThresholdOptimizer

    if logger_callback:
        logger_callback(f"[INIT] Initializing Fairlearn ThresholdOptimizer post-processing...")
        logger_callback(f"[CONFIG] Constraint: {constraint} | Predict Method: {predict_method}")

    t0 = time.time()
    optimizer = ThresholdOptimizer(
        estimator=base_estimator,
        constraints=cast(Any, constraint),
        predict_method=cast(Any, predict_method),
        prefit=hasattr(base_estimator, "classes_")
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        optimizer.fit(X_train, y_train, sensitive_features=sensitive_features)

    elapsed = time.time() - t0
    if logger_callback:
        logger_callback(f"[OPTIMIZER] Per-group decision threshold optimization completed in {elapsed:.3f}s.")
        logger_callback(f"[SUCCESS] ThresholdOptimizer successfully fitted and calibrated.")

    return optimizer


def predict_threshold_optimizer(optimizer, X_test, sensitive_features, random_state=0):
    """
    Generate predictions from a fitted ThresholdOptimizer.
    """
    if random_state is not None:
        np.random.seed(random_state)
    y_pred = optimizer.predict(X_test, sensitive_features=sensitive_features, random_state=random_state)
    
    # Estimate probabilities if underlying thresholder supports it
    y_prob = y_pred.astype(float)
    if hasattr(optimizer, "_pmf_predict"):
        try:
            pmf = optimizer._pmf_predict(X_test, sensitive_features=sensitive_features)
            y_prob = pmf[:, 1] if (isinstance(pmf, np.ndarray) and pmf.ndim > 1) else pmf
        except Exception:
            pass

    return y_pred, y_prob


def extract_threshold_optimizer_changes(optimizer) -> dict:
    """
    Extract per-group threshold adjustments from a fitted ThresholdOptimizer.
    """
    threshold_dict = {}
    if hasattr(optimizer, "interpolated_thresholder_") and hasattr(optimizer.interpolated_thresholder_, "interpolation_dict"):
        interp = optimizer.interpolated_thresholder_.interpolation_dict
        for grp, rules in interp.items():
            threshold_dict[str(grp)] = {
                "p0": float(rules.get("p0", 0.0)),
                "p1": float(rules.get("p1", 1.0)),
                "operation0": str(rules.get("operation0", "")),
                "operation1": str(rules.get("operation1", ""))
            }

    return {
        "algorithm": "Fairlearn ThresholdOptimizer (Post-Processing)",
        "group_threshold_adjustments": threshold_dict,
        "summary": (
            "The ThresholdOptimizer post-processes classifier output probabilities by learning group-specific "
            "classification cutoffs and randomized interpolation boundaries to achieve demographic parity/equalized odds."
        )
    }



def extract_mitigation_model_changes(
    mitigated_model,
    X_train=None,
    y_train=None,
    feature_names=None
) -> dict:
    """
    Extract transparent architectural and parameter changes made during bias mitigation.

    Parameters:
        mitigated_model: Trained ExponentiatedGradient model.
        X_train, y_train: Training data (optional).
        feature_names: List of feature names (optional).

    Returns:
        dict: Detailed breakdown of ensemble sub-predictors, weights, duality gaps, and decision shifts.
    """
    predictors = getattr(mitigated_model, "predictors_", [])
    weights = getattr(mitigated_model, "weights_", [])
    best_gap = getattr(mitigated_model, "best_gap_", None)
    last_iter = getattr(mitigated_model, "last_iter_", None)

    predictor_details = []
    for idx, (w, p) in enumerate(zip(weights, predictors)):
        p_name = p.__class__.__name__
        weight_val = float(w)
        is_active = weight_val > 0.001

        info = {
            "predictor_index": idx + 1,
            "estimator_type": p_name,
            "ensemble_weight": round(weight_val, 4),
            "weight_percentage": f"{weight_val * 100:.2f}%",
            "is_active": is_active
        }

        # If linear model, capture intercept & top weighted features
        if hasattr(p, "coef_") and hasattr(p, "intercept_"):
            info["intercept"] = round(float(p.intercept_[0]) if hasattr(p.intercept_, "__len__") else float(p.intercept_), 4)
            if feature_names is not None and len(feature_names) == p.coef_.shape[-1]:
                coefs = p.coef_[0] if p.coef_.ndim > 1 else p.coef_
                top_indices = np.argsort(np.abs(coefs))[-5:][::-1]
                info["top_features"] = [
                    {"feature": feature_names[i], "coefficient": round(float(coefs[i]), 4)}
                    for i in top_indices
                ]
        elif hasattr(p, "feature_importances_"):
            if feature_names is not None and len(feature_names) == len(p.feature_importances_):
                top_indices = np.argsort(p.feature_importances_)[-5:][::-1]
                info["top_features"] = [
                    {"feature": feature_names[i], "importance": round(float(p.feature_importances_[i]), 4)}
                    for i in top_indices
                ]

        predictor_details.append(info)

    return {
        "algorithm": "Fairlearn ExponentiatedGradient",
        "total_predictors_trained": len(predictors),
        "active_predictors_count": sum(1 for w in weights if w > 0.001),
        "best_duality_gap": round(float(best_gap), 5) if best_gap is not None else None,
        "optimization_iterations": last_iter,
        "ensemble_composition": predictor_details,
        "summary": (
            f"The ExponentiatedGradient solver constructed a convex randomized ensemble of "
            f"{len(predictors)} sub-classifiers ({sum(1 for w in weights if w > 0.001)} active) "
            f"to re-weight decision boundaries and satisfy demographic fairness constraints."
        )
    }


def calculate_mitigation_tradeoff(
    baseline_metrics,
    mitigated_metrics,
    baseline_intersectional_audit,
    mitigated_intersectional_audit
):
    """
    Calculate explicit predictive accuracy trade-offs, fairness gains, and group opportunity deltas.

    Parameters:
        baseline_metrics (dict): ML performance metrics of baseline model.
        mitigated_metrics (dict): ML performance metrics of mitigated model.
        baseline_intersectional_audit (dict): Intersectional audit results of baseline model.
        mitigated_intersectional_audit (dict): Intersectional audit results of mitigated model.

    Returns:
        dict: Trade-off dictionary containing Delta Accuracy, Delta F1, Delta Disparities, and Tradeoff Efficiency.
    """
    base_acc = float(baseline_metrics.get("accuracy", 0.0))
    mit_acc = float(mitigated_metrics.get("accuracy", 0.0))
    delta_acc = mit_acc - base_acc

    base_f1 = float(baseline_metrics.get("f1_score", 0.0))
    mit_f1 = float(mitigated_metrics.get("f1_score", 0.0))
    delta_f1 = mit_f1 - base_f1

    base_auc = baseline_metrics.get("roc_auc")
    mit_auc = mitigated_metrics.get("roc_auc")
    delta_auc = (float(mit_auc) - float(base_auc)) if (base_auc is not None and mit_auc is not None) else None

    base_prec = float(baseline_metrics.get("precision", 0.0))
    mit_prec = float(mitigated_metrics.get("precision", 0.0))
    delta_prec = mit_prec - base_prec

    base_rec = float(baseline_metrics.get("recall", 0.0))
    mit_rec = float(mitigated_metrics.get("recall", 0.0))
    delta_rec = mit_rec - base_rec

    # Disparities Before vs After
    b_disp = baseline_intersectional_audit.get("disparities", {})
    m_disp = mitigated_intersectional_audit.get("disparities", {})

    b_dpd = b_disp.get("demographic_parity_difference")
    m_dpd = m_disp.get("demographic_parity_difference")
    gain_dpd = (b_dpd - m_dpd) if (b_dpd is not None and m_dpd is not None) else None
    gain_dpd_pct = ((gain_dpd / b_dpd) * 100) if (gain_dpd is not None and b_dpd and b_dpd > 1e-6) else None

    b_eod = b_disp.get("equalized_odds_difference")
    m_eod = m_disp.get("equalized_odds_difference")
    gain_eod = (b_eod - m_eod) if (b_eod is not None and m_eod is not None) else None
    gain_eod_pct = ((gain_eod / b_eod) * 100) if (gain_eod is not None and b_eod and b_eod > 1e-6) else None

    b_eopp = b_disp.get("equal_opportunity_difference")
    m_eopp = m_disp.get("equal_opportunity_difference")
    gain_eopp = (b_eopp - m_eopp) if (b_eopp is not None and m_eopp is not None) else None

    b_dir = b_disp.get("disparate_impact_ratio")
    m_dir = m_disp.get("disparate_impact_ratio")
    gain_dir = (m_dir - b_dir) if (b_dir is not None and m_dir is not None) else None

    # Group-Level Opportunity Breakdown (Before vs After)
    b_groups = baseline_intersectional_audit.get("all_group_metrics", {})
    m_groups = mitigated_intersectional_audit.get("all_group_metrics", {})
    group_comparison = []

    for grp_key in sorted(b_groups.keys()):
        if grp_key in m_groups:
            bg = b_groups[grp_key]
            mg = m_groups[grp_key]
            b_tpr = bg.get("true_positive_rate")
            m_tpr = mg.get("true_positive_rate")
            b_fpr = bg.get("false_positive_rate")
            m_fpr = mg.get("false_positive_rate")
            b_sr = bg.get("selection_rate")
            m_sr = mg.get("selection_rate")

            group_comparison.append({
                "group": grp_key,
                "sample_count": bg.get("sample_count", 0),
                "baseline_selection_rate": b_sr,
                "mitigated_selection_rate": m_sr,
                "selection_rate_delta": (m_sr - b_sr) if (b_sr is not None and m_sr is not None) else None,
                "baseline_tpr": b_tpr,
                "mitigated_tpr": m_tpr,
                "tpr_delta": (m_tpr - b_tpr) if (b_tpr is not None and m_tpr is not None) else None,
                "baseline_fpr": b_fpr,
                "mitigated_fpr": m_fpr,
                "fpr_delta": (m_fpr - b_fpr) if (b_fpr is not None and m_fpr is not None) else None
            })

    # Tradeoff Efficiency: Fairness gain per unit accuracy loss
    acc_loss = abs(delta_acc)
    if gain_eod is not None:
        efficiency_ratio = (gain_eod / acc_loss) if acc_loss > 1e-6 else (float("inf") if gain_eod >= 0 else 0.0)
    else:
        efficiency_ratio = None

    eod_gain_str = f"{gain_eod:.4f}" if gain_eod is not None else "N/A"
    eod_pct_str = f"({gain_eod_pct:.1f}% reduction)" if gain_eod_pct is not None else ""

    return {
        "performance_comparison": {
            "accuracy": {
                "baseline": base_acc, "mitigated": mit_acc, "delta": delta_acc,
                "pct_change": (delta_acc / base_acc * 100) if base_acc > 0 else 0.0,
                "status": "Improved" if delta_acc > 0.005 else ("Worsened" if delta_acc < -0.005 else "Neutral")
            },
            "precision": {
                "baseline": base_prec, "mitigated": mit_prec, "delta": delta_prec,
                "pct_change": (delta_prec / base_prec * 100) if base_prec > 0 else 0.0,
                "status": "Improved" if delta_prec > 0.005 else ("Worsened" if delta_prec < -0.005 else "Neutral")
            },
            "recall": {
                "baseline": base_rec, "mitigated": mit_rec, "delta": delta_rec,
                "pct_change": (delta_rec / base_rec * 100) if base_rec > 0 else 0.0,
                "status": "Improved" if delta_rec > 0.005 else ("Worsened" if delta_rec < -0.005 else "Neutral")
            },
            "f1_score": {
                "baseline": base_f1, "mitigated": mit_f1, "delta": delta_f1,
                "pct_change": (delta_f1 / base_f1 * 100) if base_f1 > 0 else 0.0,
                "status": "Improved" if delta_f1 > 0.005 else ("Worsened" if delta_f1 < -0.005 else "Neutral")
            },
            "roc_auc": {
                "baseline": base_auc, "mitigated": mit_auc, "delta": delta_auc,
                "status": "Improved" if (delta_auc and delta_auc > 0.005) else ("Worsened" if (delta_auc and delta_auc < -0.005) else "Neutral")
            }
        },
        "fairness_comparison": {
            "demographic_parity_difference": {
                "baseline": b_dpd, "mitigated": m_dpd, "fairness_gain": gain_dpd, "gain_pct": gain_dpd_pct,
                "status": "Improved" if (gain_dpd and gain_dpd > 0.005) else ("Worsened" if (gain_dpd and gain_dpd < -0.005) else "Neutral")
            },
            "equalized_odds_difference": {
                "baseline": b_eod, "mitigated": m_eod, "fairness_gain": gain_eod, "gain_pct": gain_eod_pct,
                "status": "Improved" if (gain_eod and gain_eod > 0.005) else ("Worsened" if (gain_eod and gain_eod < -0.005) else "Neutral")
            },
            "equal_opportunity_difference": {
                "baseline": b_eopp, "mitigated": m_eopp, "fairness_gain": gain_eopp,
                "status": "Improved" if (gain_eopp and gain_eopp > 0.005) else ("Worsened" if (gain_eopp and gain_eopp < -0.005) else "Neutral")
            },
            "disparate_impact_ratio": {
                "baseline": b_dir, "mitigated": m_dir, "fairness_improvement": gain_dir,
                "status": "Improved" if (gain_dir and gain_dir > 0.02) else ("Worsened" if (gain_dir and gain_dir < -0.02) else "Neutral")
            }
        },
        "group_opportunity_breakdown": group_comparison,
        "tradeoff_efficiency": {
            "equalized_odds_gain_per_acc_loss": efficiency_ratio,
            "gain_eod": gain_eod,
            "gain_eod_pct": gain_eod_pct,
            "delta_acc": delta_acc,
            "interpretation": (
                f"Mitigation successfully reduced Equalized Odds Disparity gap by {eod_gain_str} {eod_pct_str} "
                f"with an accuracy impact of {delta_acc:+.4f}."
            )
        }
    }


def generate_pareto_front_grid(
    X_train,
    y_train,
    sensitive_train,
    X_test,
    y_test,
    sensitive_test_df,
    constraint_type="EqualizedOdds",
    eps_grid=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
    random_state=42,
    base_estimator_type="logistic_regression"
):
    """
    Evaluate a grid of constraint strictness parameters (eps) to build a Pareto trade-off front.

    Parameters:
        X_train, y_train, sensitive_train: Training data and sensitive features.
        X_test, y_test, sensitive_test_df: Test data and sensitive attributes DataFrame.
        constraint_type (str): Fairness constraint to optimize.
        eps_grid (list): List of eps values (constraint tolerance bounds).
        random_state (int): Seed for reproducibility.
        base_estimator_type (str): Base estimator key or model instance.

    Returns:
        list: List of dict records containing eps, accuracy, f1_score, equalized_odds_diff, demographic_parity_diff.
    """
    grid_results = []
    from src.models.classifier import evaluate_performance_metrics

    for eps in eps_grid:
        mit_model = train_mitigated_model(
            X_train,
            y_train,
            sensitive_train,
            constraint_type=constraint_type,
            eps=eps,
            random_state=random_state,
            base_estimator_type=base_estimator_type
        )

        y_pred, y_prob = predict_mitigated_model(mit_model, X_test, random_state=random_state)
        perf = evaluate_performance_metrics(y_test, y_pred, y_prob)
        audit = audit_intersectional_attributes(y_test, y_pred, sensitive_test_df, min_group_size=30)

        grid_results.append({
            "eps": float(eps),
            "accuracy": perf["accuracy"],
            "f1_score": perf["f1_score"],
            "roc_auc": perf.get("roc_auc"),
            "equalized_odds_difference": audit["disparities"]["equalized_odds_difference"],
            "demographic_parity_difference": audit["disparities"]["demographic_parity_difference"],
            "equal_opportunity_difference": audit["disparities"]["equal_opportunity_difference"]
        })

    return grid_results
