"""
Retraining Recommender Engine.

Analyzes monitoring outputs across schema, data quality, feature drift, target drift, performance,
fairness, and calibration to generate structured retraining recommendations: NO_RETRAINING_NEEDED,
RETRAINING_RECOMMENDED, or RETRAINING_REQUIRED with explicit human-readable reasons.
"""

from typing import Dict, Any, List, Optional


def evaluate_retraining_recommendation(
    model_health_report: Dict[str, Any],
    schema_report: Dict[str, Any],
    quality_report: Dict[str, Any],
    feature_drift_report: Dict[str, Any],
    target_drift_report: Dict[str, Any],
    performance_report: Dict[str, Any],
    fairness_report: Dict[str, Any],
    calibration_report: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate structured retraining recommendation with explicit failure/drift reasons.

    Returns:
        dict: Recommendation report containing status, action_required, and explicit reasons list.
    """
    reasons: List[str] = []
    requires_critical_action = False
    recommends_action = False

    # 1. Check schema drift
    if schema_report.get("has_critical_drift"):
        issues = schema_report.get("critical_issues", [])
        reasons.append(f"Critical schema drift detected: {'; '.join(issues)}")
        requires_critical_action = True

    # 2. Check data quality drift
    collapsed_groups = quality_report.get("collapsed_protected_groups", [])
    if collapsed_groups:
        reasons.append(f"Protected subgroup sample collapse detected: {', '.join(collapsed_groups)}")
        recommends_action = True

    overall_miss_delta = quality_report.get("overall_missing_pct_delta", 0.0)
    if overall_miss_delta >= 0.15:
        reasons.append(f"Overall missing value rate increased by {overall_miss_delta * 100:.1f}%")
        recommends_action = True

    # 3. Check feature drift
    drifted_features = feature_drift_report.get("drifted_features", [])
    if len(drifted_features) > 0:
        feature_metrics = feature_drift_report.get("feature_metrics", {})
        for feat in drifted_features[:5]:
            score = feature_metrics.get(feat, {}).get("drift_score", 0.0)
            reasons.append(f"Feature '{feat}' experienced significant drift (drift score: {score:.3f})")
        recommends_action = True

    # 4. Check target drift
    if target_drift_report.get("status") == "DRIFT_DETECTED":
        delta = target_drift_report.get("positive_rate_delta", 0.0)
        reasons.append(f"Target positive class rate shifted by {delta * 100:+.1f}%")
        recommends_action = True

    # 5. Check performance degradation
    if performance_report.get("status") == "DEGRADED":
        f1_delta = performance_report.get("deltas", {}).get("f1_score_delta", 0.0)
        pct_change = performance_report.get("deltas", {}).get("f1_score_pct_change", 0.0)
        reasons.append(f"Model F1-score degraded by {abs(f1_delta):.4f} ({abs(pct_change):.1f}%)")
        recommends_action = True
    elif performance_report.get("status") == "WARNING":
        f1_delta = performance_report.get("deltas", {}).get("f1_score_delta", 0.0)
        reasons.append(f"Minor performance degradation observed (F1 delta: {f1_delta:+.4f})")

    # 6. Check fairness degradation
    if fairness_report.get("status") in ["FAIRNESS_DEGRADED", "DEGRADED"]:
        eod_delta = fairness_report.get("eod_delta")
        if eod_delta is not None:
            reasons.append(f"Intersectional Equalized Odds Difference increased by +{eod_delta:.4f}")
        else:
            reasons.append("Intersectional fairness disparities worsened beyond configured threshold.")
        recommends_action = True

    # 7. Check calibration degradation
    if calibration_report.get("status") == "DEGRADED":
        b_delta = calibration_report.get("brier_delta")
        if b_delta is not None:
            reasons.append(f"Model Brier calibration score degraded by +{b_delta:.4f}")
        recommends_action = True

    # Assign recommendation status
    if requires_critical_action:
        status = "RETRAINING_REQUIRED"
    elif recommends_action:
        status = "RETRAINING_RECOMMENDED"
    else:
        status = "NO_RETRAINING_NEEDED"
        reasons.append("All monitored metrics across schema, data quality, feature drift, performance, fairness, and calibration remain within stable bounds.")

    return {
        "status": status,
        "action_required": (status != "NO_RETRAINING_NEEDED"),
        "requires_user_approval": True,
        "reasons_count": len(reasons),
        "reasons": reasons
    }
