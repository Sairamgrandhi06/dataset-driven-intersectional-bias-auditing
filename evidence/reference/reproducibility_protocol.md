# Intersectional Audit Reproducibility Protocol

This document defines the official step-by-step protocol for reproducing the intersectional fairness audit (`Sex` $\times$ `Race`) and validating results against the independent expected-result oracle.

---

## 1. Environment Setup & Prerequisites

1. **Python Environment**: Python 3.14+ virtual environment (`.venv`).
2. **Dependencies**: `pandas`, `numpy`, `scikit-learn`, `fairlearn`, `pytest`.
3. **Dataset File**: `data/raw/adult.csv` (SHA-256: `9007ff524acf0776b9598a0ed883058df91f59c003e78cb2cddd88f87535fc86`).

---

## 2. Dataset Hash & Identity Verification

Verify the raw dataset integrity prior to running the audit:
```bash
python -c "import hashlib; print(hashlib.sha256(open('data/raw/adult.csv','rb').read()).hexdigest())"
```
*Expected SHA-256*: `9007ff524acf0776b9598a0ed883058df91f59c003e78cb2cddd88f87535fc86`

---

## 3. Configuration & Parameter Standards

- **Random Seed**: `random_state = 42` for train/test split and classifier fitting.
- **Train / Test Split Ratio**: 80% Train ($N = 24,129$), 20% Test ($N = 6,033$).
- **Statistical Safety Threshold**: Minimum subgroup sample size $N \ge 30$.
- **Disparity Metric Tolerance**: Absolute tolerance $\epsilon = 10^{-4}$ ($0.0001$).

---

## 4. Execution Commands

### A. Run One-Command Reproducibility Pipeline (Phase 9):
```bash
python src/run_phase9.py
```

### B. Run Standalone Reference Audit:
```bash
python reference/run_reference_audit.py
```

### C. Run Production vs. Reference Comparison:
```bash
python reference/compare_with_production.py
```

### D. Run Automated Test Suite:
```bash
python -m pytest -v
```

---

## 5. Expected Target Benchmark Values

| Metric | Expected Value | Tolerance | Acceptance Status |
| :--- | :--- | :--- | :--- |
| **Demographic Parity Difference** | `0.3171` | $\pm 0.0001$ | **PASS** |
| **Disparate Impact Ratio** | `0.1333` | $\pm 0.0001$ | **PASS** |
| **Equal Opportunity Difference (TPR)** | `0.4825` | $\pm 0.0001$ | **PASS** |
| **Equalized Odds Difference** | `0.4825` | $\pm 0.0001$ | **PASS** |
| **False Positive Rate Difference** | `0.1455` | $\pm 0.0001$ | **PASS** |

---

## 6. Failure Handling Protocol

If any metric comparison exceeds the $\pm 10^{-4}$ tolerance threshold or dataset hash mismatch occurs:
1. Pipeline halts immediately with exit code `1`.
2. Mismatch details are output to console and saved in `evidence/reference/reproducibility_report.json`.
3. Check random seed configuration and raw dataset integrity.
