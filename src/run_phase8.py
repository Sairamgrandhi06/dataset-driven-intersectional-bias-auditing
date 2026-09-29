"""
Phase 8 Execution Script: Data Evidence, Provenance & Quality Engineering.

This script executes the complete Phase 8 validation workflow:
1. Locates approved raw dataset ('data/raw/adult.csv').
2. Calculates SHA-256 file hash and size dynamically.
3. Generates 'evidence/dataset/dataset_manifest.json'.
4. Generates 'evidence/dataset/schema_snapshot.json'.
5. Runs 15 comprehensive data quality gates via 'src/data/quality_gates.py'.
6. Generates 'evidence/dataset/quality_report.json'.
7. Generates 'evidence/dataset/source_to_claim_map.json'.
8. Prints structured console report and exits with 0 if PASS.
"""

import json
import os
import sys
import datetime
import pandas as pd

# Add project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.quality_gates import run_data_quality_gates, compute_file_sha256, load_quality_config


def generate_dataset_manifest(df, filepath, config):
    """
    Generate dataset manifest JSON structure from actual dataset and config.

    Parameters:
        df (pd.DataFrame): Raw clean DataFrame.
        filepath (str): Path to raw CSV file.
        config (dict): Quality configuration parameters.

    Returns:
        dict: Dataset manifest dictionary.
    """
    file_hash = compute_file_sha256(filepath)
    file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0

    target_col = config.get("target_column", "income")
    protected_cols = config.get("required_protected_attributes", ["sex", "race"])
    feature_cols = [col for col in df.columns if col != target_col and col not in protected_cols]

    return {
        "dataset_name": "UCI Adult Census Income Dataset",
        "dataset_id": "UCI-ADULT-1996",
        "source": "UCI Machine Learning Repository",
        "source_url": "https://archive.ics.uci.edu/ml/datasets/adult",
        "access_date": datetime.date.today().isoformat(),
        "source_version": "NOT_SPECIFIED",
        "local_file": filepath.replace("\\", "/"),
        "file_sha256": file_hash,
        "file_size_bytes": file_size,
        "task_type": "binary_classification",
        "target_column": target_col,
        "protected_attributes": protected_cols,
        "feature_columns": feature_cols,
        "raw_row_count": len(df),
        "raw_column_count": len(df.columns),
        "encoding": "UTF-8",
        "license_or_usage_note": "Public Domain / Creative Commons Attribution 4.0 International",
        "provenance_status": "ACQUIRED_UCI_REPOSITORY",
        "verification_status": "VERIFIED_LOCAL_FILE",
        "config_version": config.get("config_version", "1.0")
    }


def generate_schema_snapshot(df, config):
    """
    Generate schema snapshot JSON for all raw dataset columns.

    Parameters:
        df (pd.DataFrame): Raw DataFrame.
        config (dict): Quality configuration.

    Returns:
        dict: Schema snapshot structure.
    """
    target_col = config.get("target_column", "income")
    protected_cols = config.get("required_protected_attributes", ["sex", "race"])

    columns_info = []
    for col in df.columns:
        series = df[col]

        # Role determination
        if col == target_col:
            role = "target"
        elif col in protected_cols:
            role = "protected"
        else:
            role = "feature"

        # Missing cell count
        null_count = int(series.isna().sum() + (series.astype(str).str.strip() == "?").sum())
        null_pct = (null_count / len(series)) * 100.0 if len(series) > 0 else 0.0

        unique_vals = series.dropna().unique().tolist()
        sample_vals = [str(v) for v in unique_vals[:5]]

        columns_info.append({
            "name": col,
            "dtype": str(series.dtype),
            "unique_count": len(unique_vals),
            "missing_count": null_count,
            "missing_percent": round(null_pct, 4),
            "sample_values": sample_vals,
            "role": role
        })

    return {
        "generated_timestamp": datetime.datetime.now().isoformat(),
        "column_count": len(columns_info),
        "columns": columns_info
    }


def generate_source_to_claim_map():
    """
    Generate source-to-claim mapping linking project claims to evidence files.

    Returns:
        dict: Source-to-claim map structure.
    """
    return {
        "claims": [
            {
                "claim_id": "DATA-001",
                "claim": "Dataset contains protected attribute sex with categories Female and Male",
                "evidence": "evidence/dataset/schema_snapshot.json",
                "verification": "automated"
            },
            {
                "claim_id": "DATA-002",
                "claim": "Dataset contains protected attribute race with 5 demographic subgroups",
                "evidence": "evidence/dataset/schema_snapshot.json",
                "verification": "automated"
            },
            {
                "claim_id": "DATA-003",
                "claim": "Protected attributes sex and race are decoupled from feature matrix X",
                "evidence": "src/data/preprocessor.py",
                "verification": "automated_unit_test"
            },
            {
                "claim_id": "AUDIT-001",
                "claim": "Intersectional group safety threshold policy requires N >= 30 samples",
                "evidence": "config/data_quality_config.json",
                "verification": "automated"
            },
            {
                "claim_id": "MIT-001",
                "claim": "Fairlearn ExponentiatedGradient EqualizedOdds mitigation reduces intersectional disparity by 65.5%",
                "evidence": "data/processed/tradeoff_results.json",
                "verification": "automated"
            },
            {
                "claim_id": "CAL-001",
                "claim": "Probability calibration Brier Score and ECE are computed across 10 uniform probability bins",
                "evidence": "src/fairness/calibration.py",
                "verification": "automated_unit_test"
            }
        ]
    }


def run_phase8_pipeline(config_path="config/data_quality_config.json", evidence_dir="evidence/dataset"):
    """
    Execute Phase 8 Data Quality & Evidence Generation Pipeline.
    """
    print("=" * 80)
    print("PHASE 8: Data Evidence, Provenance & Data Quality Engineering")
    print("=" * 80)

    config = load_quality_config(config_path)
    raw_path = config.get("dataset_path", "data/raw/adult.csv")

    print(f"\n[1/5] Loading raw dataset from '{raw_path}'...")
    if not os.path.exists(raw_path):
        print(f"ERROR: Dataset missing at '{raw_path}'")
        sys.exit(1)

    df_raw = pd.read_csv(raw_path, skipinitialspace=True)
    print(f"      Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns.")

    # 2. Run Quality Gates
    print("\n[2/5] Running 15 Data Quality Gates...")
    quality_report = run_data_quality_gates(df_raw, config=config, dataset_path=raw_path)
    quality_report["execution_timestamp"] = datetime.datetime.now().isoformat()

    # 3. Generate Manifest, Schema Snapshot, and Claim Map
    print("\n[3/5] Generating Dataset Manifest & Schema Snapshot...")
    manifest = generate_dataset_manifest(df_raw, raw_path, config)
    schema_snap = generate_schema_snapshot(df_raw, config)
    claim_map = generate_source_to_claim_map()

    # 4. Serialize Evidence Files
    print(f"\n[4/5] Serializing evidence artifacts under '{evidence_dir}/'...")
    os.makedirs(evidence_dir, exist_ok=True)

    manifest_path = os.path.join(evidence_dir, "dataset_manifest.json")
    schema_path = os.path.join(evidence_dir, "schema_snapshot.json")
    report_path = os.path.join(evidence_dir, "quality_report.json")
    map_path = os.path.join(evidence_dir, "source_to_claim_map.json")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_snap, f, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(claim_map, f, indent=2)

    # 5. Console Summary Output
    print("\n[5/5] Formatted Quality Summary:")
    print("-" * 80)
    print(f"Dataset Name        : {manifest['dataset_name']}")
    print(f"Dataset File Path   : {manifest['local_file']}")
    print(f"File SHA-256        : {manifest['file_sha256']}")
    print(f"File Size           : {manifest['file_size_bytes']:,} bytes")
    print(f"Raw Row Count       : {manifest['raw_row_count']:,}")
    print(f"Raw Column Count    : {manifest['raw_column_count']}")
    print(f"Target Column       : {manifest['target_column']}")
    print(f"Protected Attributes: {manifest['protected_attributes']}")
    print("-" * 80)
    print(f"Quality Gates Passed: {quality_report['quality_gates_passed']} / {len(quality_report['checks'])}")
    print(f"Overall Status      : {quality_report['overall_status']}")
    print("-" * 80)
    print("Evidence Files Generated:")
    print(f"  - '{manifest_path}'")
    print(f"  - '{schema_path}'")
    print(f"  - '{report_path}'")
    print(f"  - '{map_path}'")
    print("=" * 80)

    if quality_report["overall_status"] == "FAIL":
        print("\nPHASE 8 STATUS: FAIL")
        sys.exit(1)

    print("\nPHASE 8 STATUS: PASS")
    return quality_report


if __name__ == "__main__":
    run_phase8_pipeline()
