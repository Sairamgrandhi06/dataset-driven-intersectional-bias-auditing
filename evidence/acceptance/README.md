# Acceptance & Verification Evidence

This directory contains acceptance criteria verification logs and test execution summaries confirming project compliance with functional and quality specifications.

## Key Acceptance Criteria

- **AC-1**: Feature/Attribute Decoupling ($X, y, A$ separation) verified by automated tests in `tests/test_data.py`.
- **AC-2**: Intersectional Subgroup Thresholding ($N \ge 30$) enforced by `src/fairness/intersectional.py`.
- **AC-3**: Bias Mitigation & Pareto Front Reporting verified by `src/mitigation/mitigator.py`.
- **AC-4**: Probability Calibration (Brier Score & ECE) verified by `src/fairness/calibration.py`.
- **AC-5**: Interactive Dashboard Verification verified by `tests/test_dashboard.py`.
- **AC-6**: Automated Regression Test Suite verified by `pytest tests/`.
