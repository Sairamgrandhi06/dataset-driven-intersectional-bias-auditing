"""
Model Health Synthesizer Module.

Evaluates 7 health dimensions (schema, data quality, feature drift, target drift, performance,
fairness, calibration) and synthesizes an overall model health state: HEALTHY, WARNING, DEGRADED, or BLOCKED.
"""

from typing import Dict, Any, List, Optional


def evaluate_model_health(
    schema_report: Dict[str, Any],
    quality_report: Dict[str, Any],
    feature_drift_report: Dict[str, Any],
    target_drift_report: Dict[str, Any],
    performance_report: Dict[str, Any],
    fairness_report: Dict[str, Any],
    calibration_report: Dict[str, Any],
    artifact_integrity: bool = True
) -> Dict[str, Any]:
    """
    Synthesize individual monitoring reports into a comprehensive Model Health Assessment.

    Returns:
        dict: Consolidated Model Health Report containing component states and overall health state.
    """
    # 1. Schema health
    if schema_report.get("status") == "SCHEMA_DRIFT_DETECTED" or schema_report.get("has_critical_drift"):
        schema_health = "BLOCKED"
    else:
        schema_health = "HEALTHY"

    # 2. Data quality health
    dq_status = quality_report.get("status", "QUALITY_STABLE")
    if dq_status == "QUALITY_DEGRADED":
        quality_health = "DEGRADED"
    elif dq_status == "WARNING":
        quality_health = "WARNING"
    else:
        quality_health = "HEALTHY"

    # 3. Feature drift health
    fd_status = feature_drift_report.get("status", "NO_DRIFT")
    if fd_status == "DRIFT_DETECTED":
        feature_drift_health = "DEGRADED"
    elif fd_status == "WARNING":
        feature_drift_health = "WARNING"
    else:
        feature_drift_health = "HEALTHY"

    # 4. Target drift health
    td_status = target_drift_report.get("status", "TARGET_STABLE")
    if td_status == "TARGET_LABELS_UNAVAILABLE":
        target_drift_health = "UNAVAILABLE"
    elif td_status == "DRIFT_DETECTED":
        target_drift_health = "DEGRADED"
    elif td_status == "WARNING":
        target_drift_health = "WARNING"
    else:
        target_drift_health = "HEALTHY"

    # 5. Performance health
    p_status = performance_report.get("status", "STABLE")
    if p_status == "PERFORMANCE_UNAVAILABLE":
        performance_health = "UNAVAILABLE"
    elif p_status == "DEGRADED":
        performance_health = "DEGRADED"
    elif p_status == "WARNING":
        performance_health = "WARNING"
    else:
        performance_health = "HEALTHY"

    # 6. Fairness health
    f_status = fairness_report.get("status", "FAIRNESS_STABLE")
    if f_status == "FAIRNESS_UNAVAILABLE":
        fairness_health = "UNAVAILABLE"
    elif f_status == "INSUFFICIENT_GROUP_COVERAGE":
        fairness_health = "INSUFFICIENT_COVERAGE"
    elif f_status in ["FAIRNESS_DEGRADED", "DEGRADED"]:
        fairness_health = "DEGRADED"
    elif f_status == "FAIRNESS_WARNING":
        fairness_health = "WARNING"
    else:
        fairness_health = "HEALTHY"

    # 7. Calibration health
    c_status = calibration_report.get("status", "STABLE")
    if c_status == "CALIBRATION_UNAVAILABLE":
        calibration_health = "UNAVAILABLE"
    elif c_status == "DEGRADED":
        calibration_health = "DEGRADED"
    elif c_status == "WARNING":
        calibration_health = "WARNING"
    else:
        calibration_health = "HEALTHY"

    # Artifact integrity
    art_health = "HEALTHY" if artifact_integrity else "CORRUPTED"

    component_statuses = {
        "schema_health": schema_health,
        "quality_health": quality_health,
        "feature_drift_health": feature_drift_health,
        "target_drift_health": target_drift_health,
        "performance_health": performance_health,
        "fairness_health": fairness_health,
        "calibration_health": calibration_health,
        "artifact_integrity": art_health
    }

    # Synthesize overall health state
    if schema_health == "BLOCKED" or art_health == "CORRUPTED":
        overall_health = "BLOCKED"
    elif "DEGRADED" in component_statuses.values():
        overall_health = "DEGRADED"
    elif "WARNING" in component_statuses.values() or "INSUFFICIENT_COVERAGE" in component_statuses.values():
        overall_health = "WARNING"
    else:
        overall_health = "HEALTHY"

    return {
        "overall_health": overall_health,
        "component_statuses": component_statuses,
        "is_healthy": (overall_health == "HEALTHY"),
        "requires_attention": (overall_health in ["WARNING", "DEGRADED", "BLOCKED"])
    }
