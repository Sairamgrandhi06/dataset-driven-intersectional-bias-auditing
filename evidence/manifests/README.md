# System & Execution Manifests

This directory stores execution manifests documenting software dependencies, runtime environments, hardware/OS configurations, and pipeline random seeds.

## System Configuration Manifest

- **Environment**: Python 3.14+ virtual environment (`.venv`)
- **Core Dependencies**: `pandas`, `numpy`, `scikit-learn`, `fairlearn`, `matplotlib`, `seaborn`, `streamlit`, `pytest`
- **Random Seeds**: Fixed seed `random_state=42` and `np.random.seed(6)` for 100% deterministic reproducibility.
