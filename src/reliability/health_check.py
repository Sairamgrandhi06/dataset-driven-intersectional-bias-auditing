"""
Operating System Health Check Module.

Verifies raw dataset existence, file hashes, configuration files, expected oracle availability,
writable output directories, and telemetry readiness prior to pipeline execution.
"""

import json
import os
from typing import Any, Dict

from src.data.quality_gates import compute_file_sha256


def perform_operating_health_check() -> Dict[str, Any]:
    """
    Execute comprehensive system operating health check.

    Returns:
        dict: Health check result containing overall status ('PASS' or 'FAIL') and check list.
    """
    checks = []
    failures = []

    # 1. Dataset Exists & Hash Check
    raw_path = "data/raw/adult.csv"
    expected_hash = "9007ff524acf0776b9598a0ed883058df91f59c003e78cb2cddd88f87535fc86"
    file_exists = os.path.exists(raw_path)
    actual_hash = compute_file_sha256(raw_path) if file_exists else "MISSING"

    hash_pass = file_exists and (actual_hash == expected_hash)
    checks.append({
        "component": "dataset_integrity",
        "status": "PASS" if hash_pass else "FAIL",
        "observed": f"File exists={file_exists}, Hash={actual_hash[:16]}...",
        "expected": f"File exists=True, Hash={expected_hash[:16]}..."
    })
    if not hash_pass:
        failures.append("Dataset file missing or SHA-256 hash mismatch")

    # 2. Configuration Files Exist Check
    req_configs = [
        "config/default_config.json",
        "config/data_quality_config.json",
        "config/fairness_criteria_config.json"
    ]
    missing_configs = [cfg for cfg in req_configs if not os.path.exists(cfg)]
    cfg_pass = len(missing_configs) == 0
    checks.append({
        "component": "configuration_files",
        "status": "PASS" if cfg_pass else "FAIL",
        "observed": f"Missing: {missing_configs}" if not cfg_pass else "All required config files present",
        "expected": f"All config files required: {req_configs}"
    })
    if not cfg_pass:
        failures.append(f"Missing config files: {missing_configs}")

    # 3. Expected Oracle Exists Check
    oracle_path = "evidence/reference/intersectional_expected_results.json"
    oracle_exists = os.path.exists(oracle_path)
    checks.append({
        "component": "expected_result_oracle",
        "status": "PASS" if oracle_exists else "FAIL",
        "observed": f"Oracle file exists at '{oracle_path}'" if oracle_exists else f"Oracle missing at '{oracle_path}'",
        "expected": "Expected-result oracle JSON file must exist"
    })
    if not oracle_exists:
        failures.append("Expected-result oracle file missing")

    # 4. Output Directories Writable Check
    output_dirs = ["data/processed", "evidence/reliability", "evidence/telemetry", "results/figures"]
    writable_pass = True
    for d in output_dirs:
        try:
            os.makedirs(d, exist_ok=True)
            test_file = os.path.join(d, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception:
            writable_pass = False

    checks.append({
        "component": "output_directories_writable",
        "status": "PASS" if writable_pass else "FAIL",
        "observed": "All output directories writable" if writable_pass else "Output directory permission error",
        "expected": "All processed and evidence output directories must be writable"
    })
    if not writable_pass:
        failures.append("Output directory permission error")

    overall_status = "PASS" if len(failures) == 0 else "FAIL"

    return {
        "overall_status": overall_status,
        "checks_passed": sum(1 for c in checks if c["status"] == "PASS"),
        "total_checks": len(checks),
        "checks": checks,
        "failures": failures
    }
