# Source-to-Result Lineage Evidence

This directory documents the end-to-end data lineage from raw input datasets (`data/raw/adult.csv`) through feature preprocessing, baseline classification, intersectional auditing, bias mitigation, calibration evaluation, and dashboard visualization.

## Lineage Nodes

1. **Raw Dataset**: `data/raw/adult.csv` (SHA-256 verified)
2. **Preprocessed Feature Matrix $X$, Target $y$, Protected Attributes $A$**: Derived by `src/data/preprocessor.py`
3. **Baseline Model Predictions**: `data/processed/baseline_results.json`
4. **Intersectional Audit Metrics**: `data/processed/intersectional_results.json`
5. **Mitigated Model Predictions**: `data/processed/mitigated_results.json`
6. **Three-Way Trade-off & Calibration Report**: `data/processed/tradeoff_results.json` & `final_results.json`
7. **Interactive Dashboard**: `dashboard/app.py`
