# Independent Intersectional Reference Audit Module

This directory contains a standalone, independent reference implementation for intersectional fairness auditing (`Sex` $\times$ `Race`).

## Independence Declaration
The reference implementation in `intersectional_reference.py` is written completely independently from the production pipeline (`src/fairness/intersectional.py`). It does NOT import or delegate to production code. Its purpose is to act as an independent mathematical oracle to verify production audit logic and guarantee 100% result reconciliation.

## Key Components

- **`intersectional_reference.py`**: Standalone calculations for group labels, sample counts $N$, selection rates, TPR, FPR, FNR, and disparity bounds across primary groups ($N \ge 30$).
- **`expected_results.py`**: Helper functions for loading and verifying the frozen oracle stored in `evidence/reference/intersectional_expected_results.json`.
- **`run_reference_audit.py`**: Command script executing the independent reference audit and comparing output against the expected-result oracle.
- **`compare_with_production.py`**: Standalone reconciliation script asserting match between production and reference implementations within $10^{-4}$ tolerance.

## Command Execution

Run independent reference audit:
```bash
python reference/run_reference_audit.py
```

Run production vs. reference comparison:
```bash
python reference/compare_with_production.py
```
