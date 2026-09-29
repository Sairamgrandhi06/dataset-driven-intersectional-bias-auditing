"""
Unit tests for Replay and Idempotency Verification.
"""

import pytest
import os
import json
from src.reliability.replay import run_pipeline_replay_verification


def test_pipeline_replay_verification():
    """Verify 3-run replay produces numerical equality and creates replay_report.json."""
    report = run_pipeline_replay_verification(n_runs=3, tolerance=1e-4)

    assert report["reproducibility_status"] == "PASS"
    assert report["all_numerical_match"] is True

    report_path = "evidence/reliability/replay_report.json"
    assert os.path.exists(report_path)
