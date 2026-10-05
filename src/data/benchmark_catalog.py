"""
Registered Benchmark Catalog Module.

Maintains the explicit catalog of officially supported, deployment-safe benchmark datasets.
Decouples benchmark choices from historical model registry entries, experiment runs,
and test datasets.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import streamlit as st


def get_catalog_config_path(base_dir: str = ".") -> str:
    """Return absolute path to config/registered_benchmarks.json."""
    return os.path.abspath(os.path.join(base_dir, "config", "registered_benchmarks.json"))


@st.cache_data(show_spinner=False)
def _load_registered_benchmarks_cached(cat_path: str, mtime: float, size: int) -> Dict[str, Dict[str, Any]]:
    """Cached loader for registered benchmark catalog JSON."""
    try:
        with open(cat_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "registered_benchmarks" in data and isinstance(data["registered_benchmarks"], dict):
            return data["registered_benchmarks"]
    except Exception:
        pass
    return {}


def load_registered_benchmarks(base_dir: str = ".") -> Dict[str, Dict[str, Any]]:
    """
    Load explicit registered benchmark catalog from config/registered_benchmarks.json.
    Returns dictionary mapping dataset_id to benchmark metadata.
    """
    cat_path = get_catalog_config_path(base_dir)
    if os.path.exists(cat_path):
        try:
            mtime = os.path.getmtime(cat_path)
            size = os.path.getsize(cat_path)
            loaded = _load_registered_benchmarks_cached(cat_path, mtime, size)
            if loaded:
                return loaded
        except Exception:
            pass

    # Fallback to hardcoded standard catalog if file is temporarily unavailable
    return {
        "adult_census_income": {
            "dataset_id": "adult_census_income",
            "display_name": "Adult Census Income",
            "type": "benchmark",
            "enabled": True,
            "source_type": "bundled",
            "source": "data/benchmarks/adult_census_income.csv",
            "fallback_sources": [
                "data/benchmarks/adult_census_income.csv",
                "data/raw/adult.csv",
                "data/raw/adult_census_income.csv"
            ],
            "config_path": "config/adult_census_income_config.json",
            "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "columns": [
                "age", "workclass", "fnlwgt", "education", "education_num",
                "marital_status", "occupation", "relationship", "race", "sex",
                "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
            ],
            "description": "US Census Bureau income classification benchmark with gender and race protected attributes."
        },
        "students": {
            "dataset_id": "students",
            "display_name": "Student Academic Performance",
            "type": "benchmark",
            "enabled": True,
            "source_type": "bundled",
            "source": "data/benchmarks/students.csv",
            "fallback_sources": [
                "data/benchmarks/students.csv",
                "data/raw/students.csv",
                "data/raw/student_performance.csv"
            ],
            "config_path": "config/students_config.json",
            "download_url": None,
            "description": "Student academic performance benchmark with gender and ethnicity protected attributes."
        }
    }


def is_benchmark_source_available(bm_meta: Dict[str, Any], base_dir: str = ".") -> bool:
    """
    Validate whether a benchmark's data source can be resolved locally or downloaded.
    """
    if not bm_meta.get("enabled", True):
        return False

    # Check primary source path
    src = bm_meta.get("source")
    if src:
        abs_src = src if os.path.isabs(src) else os.path.join(base_dir, src)
        if os.path.exists(abs_src):
            return True

    # Check fallback sources
    for fb in bm_meta.get("fallback_sources", []):
        abs_fb = fb if os.path.isabs(fb) else os.path.join(base_dir, fb)
        if os.path.exists(abs_fb):
            return True

    # Check if download URL is configured
    if bm_meta.get("download_url"):
        return True

    return False


def get_available_benchmarks(base_dir: str = ".", validate_sources: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Return all enabled, selectable registered benchmarks.
    Excludes internal historical runs, test datasets, or datasets without valid sources.
    """
    catalog = load_registered_benchmarks(base_dir)
    available = {}

    for b_id, bm in catalog.items():
        if not bm.get("enabled", True):
            continue
        if validate_sources and not is_benchmark_source_available(bm, base_dir):
            continue
        available[b_id] = bm

    return available


def get_benchmark_display_names(base_dir: str = ".") -> List[str]:
    """Return list of display names for all selectable benchmarks."""
    available = get_available_benchmarks(base_dir, validate_sources=True)
    return [bm["display_name"] for bm in available.values()]


def get_benchmark_by_id_or_name(key: str, base_dir: str = ".") -> Optional[Dict[str, Any]]:
    """Look up benchmark metadata by dataset_id or display_name (case-insensitive)."""
    if not key:
        return None

    clean_key = str(key).lower().strip().replace(" (default)", "")
    catalog = load_registered_benchmarks(base_dir)

    for b_id, bm in catalog.items():
        if b_id.lower() == clean_key:
            return bm
        if bm.get("display_name", "").lower() == clean_key:
            return bm
        if clean_key.replace(" ", "_") == b_id.lower():
            return bm

    return None


def is_registered_benchmark(dataset_id: str, base_dir: str = ".") -> bool:
    """Return True if dataset_id matches an explicit registered benchmark entry."""
    return get_benchmark_by_id_or_name(dataset_id, base_dir) is not None


@st.cache_data(show_spinner=False)
def _read_benchmark_artifacts_cached(
    csv_path: Optional[str],
    csv_mtime: float,
    csv_size: int,
    cfg_path: Optional[str],
    cfg_mtime: float,
    cfg_size: int
) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Cached loader for benchmark CSV DataFrame and config JSON."""
    df = None
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = None

    cfg_dict = None
    if cfg_path and os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
        except Exception:
            cfg_dict = None

    return df, cfg_dict


def resolve_benchmark_dataframe(
    key: str,
    base_dir: str = "."
) -> Tuple[Optional[pd.DataFrame], Optional[dict], str]:
    """
    Resolve and load benchmark DataFrame and associated configuration.
    If source file is missing but download_url is configured, downloads and caches the file.

    Returns:
        (DataFrame or None, config_dict or None, dataset_id)
    """
    bm = get_benchmark_by_id_or_name(key, base_dir)
    if not bm:
        return None, None, key

    dataset_id = bm.get("dataset_id", key)
    src_candidates = []
    
    if bm.get("source"):
        src_candidates.append(bm["source"])
    src_candidates.extend(bm.get("fallback_sources", []))

    resolved_csv_path = None
    for cand in src_candidates:
        abs_cand = cand if os.path.isabs(cand) else os.path.join(base_dir, cand)
        if os.path.exists(abs_cand):
            resolved_csv_path = abs_cand
            break

    # If missing locally but downloadable, download to primary source destination
    if not resolved_csv_path and bm.get("download_url") and bm.get("source"):
        try:
            from src.data.loader import download_dataset_if_needed
            dest = bm["source"] if os.path.isabs(bm["source"]) else os.path.join(base_dir, bm["source"])
            download_dataset_if_needed(dest, bm["download_url"], columns=bm.get("columns"))
            if os.path.exists(dest):
                resolved_csv_path = dest
        except Exception:
            pass

    resolved_cfg_path = None
    cfg_path = bm.get("config_path")
    if cfg_path:
        abs_cfg = cfg_path if os.path.isabs(cfg_path) else os.path.join(base_dir, cfg_path)
        if os.path.exists(abs_cfg):
            resolved_cfg_path = abs_cfg

    csv_mtime = os.path.getmtime(resolved_csv_path) if resolved_csv_path and os.path.exists(resolved_csv_path) else 0.0
    csv_size = os.path.getsize(resolved_csv_path) if resolved_csv_path and os.path.exists(resolved_csv_path) else 0

    cfg_mtime = os.path.getmtime(resolved_cfg_path) if resolved_cfg_path and os.path.exists(resolved_cfg_path) else 0.0
    cfg_size = os.path.getsize(resolved_cfg_path) if resolved_cfg_path and os.path.exists(resolved_cfg_path) else 0

    df, cfg_dict = _read_benchmark_artifacts_cached(
        resolved_csv_path, csv_mtime, csv_size,
        resolved_cfg_path, cfg_mtime, cfg_size
    )

    return (df.copy() if df is not None else None), cfg_dict, dataset_id
