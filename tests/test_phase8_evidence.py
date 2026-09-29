"""
Unit tests for Phase 8 Evidence Artifacts and Execution Script.
"""

import os
import json
import pytest

from src.run_phase8 import (
    generate_dataset_manifest,
    generate_schema_snapshot,
    generate_source_to_claim_map,
    run_phase8_pipeline
)
from src.data.quality_gates import load_quality_config


def test_evidence_files_exist_after_phase8():
    """Verify run_phase8_pipeline creates all 4 required evidence JSON artifacts."""
    run_phase8_pipeline()

    assert os.path.exists("evidence/dataset/dataset_manifest.json")
    assert os.path.exists("evidence/dataset/schema_snapshot.json")
    assert os.path.exists("evidence/dataset/quality_report.json")
    assert os.path.exists("evidence/dataset/source_to_claim_map.json")


def test_dataset_manifest_contents():
    """Verify dataset_manifest.json contains valid hash, row count, and provenance status."""
    with open("evidence/dataset/dataset_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["dataset_name"] == "UCI Adult Census Income Dataset"
    assert manifest["raw_row_count"] == 32561
    assert manifest["raw_column_count"] == 15
    assert manifest["target_column"] == "income"
    assert manifest["protected_attributes"] == ["sex", "race"]
    assert len(manifest["file_sha256"]) == 64
    assert manifest["provenance_status"] == "ACQUIRED_UCI_REPOSITORY"
    assert manifest["config_version"] == "1.0"


def test_schema_snapshot_contents():
    """Verify schema_snapshot.json contains entries for all 15 raw dataset columns."""
    with open("evidence/dataset/schema_snapshot.json", "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert schema["column_count"] == 15
    assert len(schema["columns"]) == 15
    col_names = [c["name"] for c in schema["columns"]]
    assert "income" in col_names
    assert "sex" in col_names
    assert "race" in col_names


def test_source_to_claim_map_contents():
    """Verify source_to_claim_map.json contains structured claims."""
    with open("evidence/dataset/source_to_claim_map.json", "r", encoding="utf-8") as f:
        claim_map = json.load(f)

    assert "claims" in claim_map
    assert len(claim_map["claims"]) >= 5
    claim_ids = [c["claim_id"] for c in claim_map["claims"]]
    assert "DATA-001" in claim_ids
    assert "AUDIT-001" in claim_ids
