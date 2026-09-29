"""
Regression and Consistency Tests for Step 8 Model Registry & History Lifecycle.

Validates:
1. Exactly one ACTIVE model per dataset.
2. Deduplication of model version records (no duplicate v1 records).
3. Stale ACTIVE records normalization to SUPERSEDED / COMPLETED.
4. Consistency between Step 7 reference model resolution and Step 8 active model.
5. Preservation of historical completed/superseded model lineage.
6. Strict dataset isolation across all model history and active context calls.
7. No version fabrication for unversioned disk runs.
"""

import os
import pytest
from dashboard.utils import (
    get_dataset_model_history,
    get_dataset_active_model_context,
    get_project_root
)
from src.monitoring.pipeline import load_reference_data_for_model


def test_1_exactly_one_active_model_per_dataset():
    """
    Verify that every dataset has EXACTLY ONE active model version in its history.
    """
    datasets = [
        "final_project_loan_demo__1",
        "adult_census_income",
        "loan_approval",
        "student_performance"
    ]
    for ds in datasets:
        history = get_dataset_model_history(ds)
        assert len(history) > 0, f"Dataset {ds} should have registered model history"
        active_recs = [m for m in history if m.get("is_active")]
        assert len(active_recs) == 1, (
            f"Dataset {ds} must have exactly 1 ACTIVE model record, found {len(active_recs)}"
        )
        assert active_recs[0]["status"] == "ACTIVE"


def test_2_final_project_loan_demo_has_v3_active_and_v2_v1_superseded():
    """
    Verify final_project_loan_demo__1 authoritatively resolves v3 as ACTIVE, with v2 and v1 as SUPERSEDED.
    """
    history = get_dataset_model_history("final_project_loan_demo__1")
    assert len(history) == 3, f"Expected exactly 3 records for final_project_loan_demo__1, got {len(history)}"

    v3_rec = next((m for m in history if m["model_version"] == "v3"), None)
    v2_rec = next((m for m in history if m["model_version"] == "v2"), None)
    v1_rec = next((m for m in history if m["model_version"] == "v1"), None)

    assert v3_rec is not None, "v3 record must be present in history"
    assert v2_rec is not None, "v2 record must be present in history"
    assert v1_rec is not None, "v1 record must be present in history"

    assert v3_rec["is_active"] is True
    assert v3_rec["status"] == "ACTIVE"
    assert "run_20260905_030003_final_project_loan_demo__1" in v3_rec["run_id"]

    assert v2_rec["is_active"] is False
    assert v2_rec["status"] == "SUPERSEDED"
    assert "run_20260831_093310_final_project_loan_demo__1" in v2_rec["run_id"]

    assert v1_rec["is_active"] is False
    assert v1_rec["status"] == "SUPERSEDED"
    assert "run_20260831_093203_final_project_loan_demo__1" in v1_rec["run_id"]


def test_3_no_duplicate_version_records():
    """
    Verify that no dataset history contains duplicate model versions.
    """
    datasets = [
        "final_project_loan_demo__1",
        "adult_census_income",
        "loan_approval",
        "student_performance"
    ]
    for ds in datasets:
        history = get_dataset_model_history(ds)
        versions = [m["model_version"] for m in history if m.get("model_version")]
        assert len(versions) == len(set(versions)), (
            f"Duplicate versions detected in dataset {ds}: {versions}"
        )


def test_4_stale_active_records_normalized_to_superseded():
    """
    Verify that if raw records contain multiple 'ACTIVE' flags, get_dataset_model_history
    strictly normalizes all older/duplicate records to SUPERSEDED so only one remains ACTIVE.
    """
    history = get_dataset_model_history("final_project_loan_demo__1")
    active_recs = [m for m in history if m.get("is_active")]
    assert len(active_recs) == 1
    assert active_recs[0]["model_version"] == "v3"

    superseded_recs = [m for m in history if not m.get("is_active")]
    for s in superseded_recs:
        assert s["status"] in ["SUPERSEDED", "COMPLETED", "ARCHIVED"]


def test_5_consistency_between_step7_and_step8_active_models():
    """
    Verify that Step 7 reference model selection and Step 8 active model context
    resolve the identical authoritative active model (v3 for final_project_loan_demo__1).
    """
    # Step 8 context resolution
    ctx = get_dataset_active_model_context("final_project_loan_demo__1")
    assert ctx["model_version"] == "v3"
    assert ctx["status"] == "ACTIVE"
    assert "run_20260905_030003_final_project_loan_demo__1" in ctx["run_id"]

    # Step 7 reference model default loading (no version specified -> defaults to active)
    ref_bundle = load_reference_data_for_model("final_project_loan_demo__1", base_dir=".")
    assert ref_bundle["reference_version"] == "v3"
    assert ref_bundle["run_id"] == ctx["run_id"]

    # Step 7 UI labels check
    history = get_dataset_model_history("final_project_loan_demo__1")
    valid_records = [m for m in history if isinstance(m, dict) and m.get("model_version")]
    active_indices = [idx for idx, m in enumerate(valid_records) if m.get("is_active")]
    default_idx = active_indices[0] if active_indices else 0
    assert valid_records[default_idx]["model_version"] == "v3"


def test_6_preservation_of_historical_lineage_and_auditability():
    """
    Verify that v1 and v2 are preserved with exact run IDs and artifact path for auditability.
    """
    history = get_dataset_model_history("final_project_loan_demo__1")
    v1_rec = next(m for m in history if m["model_version"] == "v1")
    assert v1_rec["dataset_id"] == "final_project_loan_demo__1"
    assert v1_rec["run_id"] == "run_20260831_093203_final_project_loan_demo__1"
    assert v1_rec["model_type"] == "logistic_regression"
    assert v1_rec["status"] == "SUPERSEDED"
    assert v1_rec["is_active"] is False

    root = get_project_root()
    art_path = v1_rec.get("model_artifact_path", "")
    assert os.path.exists(art_path) or os.path.exists(
        os.path.join(root, "models", "final_project_loan_demo__1", v1_rec["run_id"], "selected_model.joblib")
    )


def test_7_strict_dataset_isolation_in_model_history():
    """
    Verify that querying history for a dataset contains only models from that dataset.
    """
    loan_demo_history = get_dataset_model_history("final_project_loan_demo__1")
    adult_history = get_dataset_model_history("adult_census_income")

    loan_ids = set(m["dataset_id"] for m in loan_demo_history)
    adult_ids = set(m["dataset_id"] for m in adult_history)

    assert loan_ids == {"final_project_loan_demo__1"}
    assert adult_ids == {"adult_census_income"}

    # No cross contamination
    loan_run_ids = set(m["run_id"] for m in loan_demo_history)
    adult_run_ids = set(m["run_id"] for m in adult_history)
    assert loan_run_ids.isdisjoint(adult_run_ids)


def test_8_no_synthetic_versions_from_disk_folders():
    """
    Verify that unversioned disk run folders under models/final_project_loan_demo__1/
    do NOT fabricate synthetic versions (v4..v34).
    """
    history = get_dataset_model_history("final_project_loan_demo__1")
    versions = [m["model_version"] for m in history]
    assert "v4" not in versions
    assert "v5" not in versions
    assert "v34" not in versions
    assert len(history) == 3


def test_9_unregistered_run_results_never_become_active():
    """
    Verify that unregistered runs (such as RUN-final_project_loan_demo__1-1788575708)
    never appear as an active model in the history.
    """
    history = get_dataset_model_history("final_project_loan_demo__1")
    run_ids = [m["run_id"] for m in history]
    for rid in run_ids:
        assert not rid.startswith("RUN-final_project_loan_demo__1")

    active_rec = next(m for m in history if m.get("is_active"))
    assert active_rec["model_version"] == "v3"
    assert active_rec["run_id"] == "run_20260905_030003_final_project_loan_demo__1"


def test_10_resolve_registered_model_history_direct_call():
    """
    Verify that resolve_registered_model_history directly resolves the exact expected records.
    """
    from dashboard.utils import resolve_registered_model_history
    resolved = resolve_registered_model_history("final_project_loan_demo__1")
    assert len(resolved) == 3
    assert resolved[0]["model_version"] == "v3"
    assert resolved[0]["is_active"] is True
    assert resolved[0]["status"] == "ACTIVE"
    assert resolved[1]["model_version"] == "v2"
    assert resolved[1]["is_active"] is False
    assert resolved[1]["status"] == "SUPERSEDED"
    assert resolved[2]["model_version"] == "v1"
    assert resolved[2]["is_active"] is False
    assert resolved[2]["status"] == "SUPERSEDED"

