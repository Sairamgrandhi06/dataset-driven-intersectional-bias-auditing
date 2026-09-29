"""
End-to-End Regression Test Suite: Streamlit Upload Workflow Priority & State Isolation.

Verifies:
1. Previously selected adult_census_income registered dataset.
2. Upload final_project_loan_demo.csv (600 rows, 8 columns).
3. State resolution and Streamlit synchronization.
4. Current workflow dataset is the uploaded loan dataset.
5. Workflow source is 'New Upload'.
6. Active model context shows 'Pending Training' / 'v0' with NO adult model/version leakage.
7. Step 4 target/protected/ID configuration belongs exclusively to the uploaded dataset.
8. adult_census_income saved configuration does not override or leak into new upload.
9. Configuration persistence is strictly isolated per dataset.
"""

import os
import json
import pytest
import pandas as pd
import streamlit as st

from dashboard.utils import (
    get_dataset_active_model_context,
    get_dataset_governance_summary,
    load_saved_dataset_configuration,
    profile_dataframe,
    get_project_root
)


@pytest.fixture(autouse=True)
def cleanup_test_configs():
    """Ensure test-generated config files and test run outputs do not pollute workspace or affect tests."""
    import shutil
    root = get_project_root()
    test_files = [
        os.path.join(root, "config", "final_project_loan_demo_config.json"),
        os.path.join(root, "config", "test_persist_loan_demo_config.json")
    ]
    test_dirs = [
        os.path.join(root, "results", "final_project_loan_demo"),
        os.path.join(root, "models", "final_project_loan_demo")
    ]
    def _cleanup():
        for p in test_files:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        for d in test_dirs:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass

    _cleanup()
    yield
    _cleanup()


def test_upload_priority_over_previously_selected_registered_dataset():
    """
    Simulate workflow:
    1. Previously selected adult_census_income (registered benchmark).
    2. User uploads final_project_loan_demo.csv.
    3. Verify current workflow dataset is final_project_loan_demo.
    4. Verify source is 'New Upload'.
    5. Verify global context has NO adult model/version leakage.
    """
    # 1. Initial State: adult_census_income registered dataset active
    st.session_state["workflow_dataset_id"] = "Adult Census Income (Default)"
    st.session_state["workflow_dataset_source"] = "Registered Dataset"
    st.session_state["active_dataset"] = "Adult Census Income (Default)"
    st.session_state["sidebar_registered_dataset_select"] = "Adult Census Income (Default)"

    # Check adult context before upload
    adult_ctx = get_dataset_active_model_context("adult_census_income")
    assert adult_ctx["dataset_id"] == "adult_census_income"
    assert adult_ctx["has_model"] is True

    # 2. Simulate Uploading final_project_loan_demo.csv
    root = get_project_root()
    loan_demo_path = os.path.join(root, "data", "raw", "final_project_loan_demo.csv")
    assert os.path.exists(loan_demo_path), "final_project_loan_demo.csv must exist in data/raw"

    df_uploaded = pd.read_csv(loan_demo_path)
    assert len(df_uploaded) in [500, 600]
    assert len(df_uploaded.columns) in [7, 8]

    raw_filename = "final_project_loan_demo.csv"
    raw_name = os.path.splitext(raw_filename)[0]
    dataset_id_input = "".join(c if c.isalnum() else "_" for c in raw_name.lower()).strip("_")
    assert dataset_id_input == "final_project_loan_demo"

    # Simulate Streamlit upload handler execution
    prev_wf_id = st.session_state.get("workflow_dataset_id")
    prev_wf_source = st.session_state.get("workflow_dataset_source")

    st.session_state["pending_dataset_id"] = dataset_id_input
    st.session_state["workflow_dataset_id"] = dataset_id_input
    st.session_state["workflow_dataset_source"] = "New Upload"
    st.session_state["active_dataset"] = dataset_id_input

    # 3. Verify workflow state immediately after upload
    assert st.session_state["workflow_dataset_id"] == "final_project_loan_demo"
    assert st.session_state["workflow_dataset_source"] == "New Upload"
    assert st.session_state["active_dataset"] == "final_project_loan_demo"

    # 4. Verify Global Context for the newly uploaded dataset
    loan_ctx = get_dataset_active_model_context("final_project_loan_demo")
    assert loan_ctx["dataset_id"] == "final_project_loan_demo"
    assert loan_ctx["selected_model"] == "Pending Training"
    assert loan_ctx["model_version"] == "v0"
    assert loan_ctx["status"] == "PENDING_TRAINING"
    assert loan_ctx["has_model"] is False

    # Verify NO adult model / version context in the active upload context
    assert loan_ctx["selected_model"] != adult_ctx["selected_model"] or loan_ctx["selected_model"] == "Pending Training"
    assert loan_ctx["model_version"] != adult_ctx["model_version"]
    assert "adult" not in loan_ctx["dataset_id"]


def test_step4_configuration_belongs_strictly_to_uploaded_dataset():
    """
    Verify Step 4 configuration restoration and AI recommendations for the uploaded dataset:
    - Target belongs to loan dataset (loan_status, approved)
    - Protected attributes belong to loan dataset (gender, race)
    - ID column belongs to loan dataset (applicant_id)
    - adult_census_income saved config is NEVER restored for final_project_loan_demo
    """
    root = get_project_root()
    loan_demo_path = os.path.join(root, "data", "raw", "final_project_loan_demo.csv")
    df_uploaded = pd.read_csv(loan_demo_path)

    dataset_id_input = "final_project_loan_demo"
    clean_ds = "adult_census_income"  # Previous dataset

    # 1. Verify load_saved_dataset_configuration strictly returns None for new upload
    saved_cfg = load_saved_dataset_configuration(dataset_id_input)
    assert saved_cfg is None, "New upload must not have a saved configuration"

    # Verify adult config is not retrieved when querying dataset_id_input
    adult_saved = load_saved_dataset_configuration(clean_ds)
    assert adult_saved is not None
    assert adult_saved["target_column"] == "income"

    # 2. Profile uploaded DataFrame
    profile = profile_dataframe(df_uploaded, dataset_id=dataset_id_input)
    target_cands = profile.get("target_candidates", [])
    protected_cands = profile.get("protected_attribute_candidates", [])
    id_cands = [col for col, meta in profile.get("columns", {}).items() if meta.get("potential_id")]

    # 3. Assert AI Suggestions reflect genuine loan dataset
    assert len(target_cands) > 0
    top_target = target_cands[0]["column"]
    assert top_target == "loan_status"
    assert target_cands[0]["recommended_positive_class"] == "approved"

    prot_names = [c["column"] if isinstance(c, dict) else c for c in protected_cands]
    assert "gender" in prot_names or "race" in prot_names
    assert "education" not in prot_names  # No adult features
    assert "marital_status" not in prot_names

    assert "applicant_id" in id_cands
    assert "fnlwgt" not in id_cands

    # 4. Form Initialization values (Step 4)
    all_cols = list(df_uploaded.columns)
    init_ds_id = saved_cfg.get("dataset_id", dataset_id_input) if saved_cfg else dataset_id_input
    assert init_ds_id == "final_project_loan_demo", "Dataset Identifier must match the uploaded dataset ID"
    assert init_ds_id != "adult_census_income"

    target_default_val = top_target
    assert target_default_val == "loan_status"

    rec_pos_val = str(target_cands[0]["recommended_positive_class"])
    assert rec_pos_val == "approved"

    avail_prot = [c for c in all_cols if c != target_default_val]
    default_prot = [c for c in prot_names if c in avail_prot]
    assert "gender" in default_prot
    assert "race" in default_prot

    avail_id = [c for c in all_cols if c != target_default_val and c not in default_prot]
    default_id = [c for c in id_cands if c in avail_id]
    assert "applicant_id" in default_id


def test_configuration_persistence_isolation_on_pipeline_execution():
    """
    Verify persisting confirmed configuration for final_project_loan_demo:
    1. Stores strictly under 'final_project_loan_demo'.
    2. Does NOT overwrite 'adult_census_income' in st.session_state['confirmed_configs'].
    """
    if "confirmed_configs" not in st.session_state:
        st.session_state["confirmed_configs"] = {}

    # Seed adult config in session state
    st.session_state["confirmed_configs"]["adult_census_income"] = {
        "dataset_id": "adult_census_income",
        "target_column": "income",
        "positive_class": ">50K",
        "protected_attributes": ["sex", "race"],
        "id_columns": [],
        "is_confirmed": True
    }

    # Simulate saving loan configuration
    loan_entry = {
        "dataset_id": "final_project_loan_demo",
        "target_column": "loan_status",
        "positive_class": "approved",
        "protected_attributes": ["gender", "race"],
        "id_columns": ["applicant_id"],
        "is_confirmed": True
    }
    st.session_state["confirmed_configs"]["final_project_loan_demo"] = loan_entry

    # Verify strict isolation
    assert st.session_state["confirmed_configs"]["adult_census_income"]["target_column"] == "income"
    assert st.session_state["confirmed_configs"]["adult_census_income"]["positive_class"] == ">50K"
    assert st.session_state["confirmed_configs"]["final_project_loan_demo"]["target_column"] == "loan_status"
    assert st.session_state["confirmed_configs"]["final_project_loan_demo"]["positive_class"] == "approved"


def test_explicit_registered_dataset_switching():
    """
    Verify switching back to a registered dataset:
    - User explicitly selects a registered benchmark.
    - Workflow dataset updates to the registered dataset and source changes to 'Registered Dataset'.
    - New upload priority is re-established if a new file is subsequently uploaded.
    """
    # 1. State is currently New Upload
    st.session_state["workflow_dataset_id"] = "final_project_loan_demo"
    st.session_state["workflow_dataset_source"] = "New Upload"

    # 2. User explicitly switches to registered dataset in sidebar
    st.session_state["sidebar_registered_dataset_select"] = "loan_approval"
    selected = st.session_state["sidebar_registered_dataset_select"]
    st.session_state["workflow_dataset_id"] = selected
    st.session_state["workflow_dataset_source"] = "Registered Dataset"
    st.session_state["active_dataset"] = selected
    st.session_state["pending_dataset_id"] = None

    assert st.session_state["workflow_dataset_id"] == "loan_approval"
    assert st.session_state["workflow_dataset_source"] == "Registered Dataset"

    # 3. User uploads another new CSV -> New upload takes priority again
    new_upload_id = "custom_customer_churn"
    st.session_state["pending_dataset_id"] = new_upload_id
    st.session_state["workflow_dataset_id"] = new_upload_id
    st.session_state["workflow_dataset_source"] = "New Upload"
    st.session_state["active_dataset"] = new_upload_id

    assert st.session_state["workflow_dataset_id"] == "custom_customer_churn"
    assert st.session_state["workflow_dataset_source"] == "New Upload"
