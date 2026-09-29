# Codebase Audit: Dataset-Specific Assumptions & Hardcoded Artifacts

## Overview
This document records every dataset-specific assumption, hardcoded column name, fixed benchmark value, and structural constraint identified in the codebase prior to implementing dataset-driven generalization.

---

## 1. Data Loader & Schema Validation (`src/data/loader.py`)

- **Dataset Source URL**: Hardcoded UCI Adult Census Income dataset URL (`https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data`).
- **File Paths**: Default file path `data/raw/adult.csv`.
- **Column Header Schema**: Default column list explicitly enumerated for Adult (`age`, `workclass`, `fnlwgt`, `education`, `education_num`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `capital_gain`, `capital_loss`, `hours_per_week`, `native_country`, `income`).
- **Whitespace Cleaning**: Standard whitespace strip assumes string columns; missing marker `"?"` is specific to UCI Adult format.

---

## 2. Preprocessing & Encoding (`src/data/preprocessor.py`)

- **Missing Value Marker**: Hardcoded string replacement for `"?"` -> `NaN`.
- **Target Value**: Hardcoded positive class label `positive_value=">50K"` and trailing dot cleanup (`">50K."`).
- **Protected Attribute Exclusion**: Assumes `sex` and `race` as default protected columns.
- **Binary Target Constraint**: Assumes binary classification target.

---

## 3. Data Quality Gates (`src/data/quality_gates.py`)

- **Minimum Row Threshold**: Fixed threshold `MINIMUM_ROW_COUNT = 30000` (tailored to Adult's 32,561 training rows).
- **Target Classes**: Fixed expectation `["<=50K", ">50K"]`.
- **Protected Attributes**: Fixed required list `["sex", "race"]`.

---

## 4. Single-Attribute & Intersectional Auditing (`src/fairness/`)

- **Default Attributes**: Parameters default to `attributes=['sex', 'race']`.
- **Intersectional Group Formatting**: Delimiter string `" + "` combined with assumptions of two categorical protected attributes.
- **Group Metric Definitions**: Minimum group sample threshold fixed at `min_group_size = 30` by default.

---

## 5. Model Training & Mitigation (`src/models/classifier.py`, `src/mitigation/mitigator.py`)

- **Baseline Classifier**: Hardcoded `LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)`.
- **Sensitive Features**: Expects `sex` and `race` series/dataframe.
- **Fairlearn In-Processing**: Enforces `EqualizedOdds` constraint on intersectional combination series.

---

## 6. Reference Oracle & Benchmark Assertions (`reference/`, `src/run_phase*.py`)

- **Authoritative Benchmark Values**:
  - Baseline Accuracy: `0.8399`
  - Mitigated Accuracy: `0.7578` (or `0.7603` within `0.005`)
  - Baseline Equalized Odds Difference: `0.4825`
  - Mitigated Equalized Odds Difference: `0.1667`
  - Baseline Brier Score: `0.1128` | Mitigated Brier Score: `0.2156`
  - Baseline ECE: `0.0151` | Mitigated ECE: `0.2166`
- **Sample Counts**: Test set size fixed at `N = 6,033` in oracle metadata (`expected_results.py`).
- **Oracle Assertion Scope**: Currently checked globally without checking if the dataset matches the frozen Adult reference dataset.

---

## 7. Configuration System (`config/`)

- `default_config.json`: Adult dataset URL, raw path, column definitions, `positive_target_value = ">50K"`, protected attributes `["sex", "race"]`.
- `data_quality_config.json`: `minimum_rows = 30000`, expected target classes `["<=50K", ">50K"]`, required protected attributes `["sex", "race"]`.

---

## 8. Dashboard Presentation (`dashboard/app.py`, `dashboard/utils.py`)

- **Hardcoded Labels & Tabs**: Radio choices explicitly reference `"sex"` and `"race"` for single-attribute audits and `"Sex × Race"` for intersectional audits.
- **Fixed Narrative & Texts**: Text descriptions reference Adult dataset sample sizes (`N = 6,033`), Adult group names (`Male + Amer-Indian-Eskimo`), and Adult benchmark metrics.

---

## Conclusion & Generalization Requirements

To make the codebase genuinely dataset-driven:
1. All column names, target column names, positive class labels, protected attributes, row thresholds, and model settings must be supplied via a dataset configuration contract.
2. Binary target handling must dynamically map configured positive class values to `1` and negative class values to `0`, rejecting multiclass targets (>2 classes) with explicit errors.
3. Protected attribute handling must dynamically process 1, 2, 3, or more attributes.
4. Reference oracle benchmark assertions must be scoped strictly to runs using the frozen Adult reference dataset configuration.
5. Streamlit dashboard must dynamically inspect dataset configurations and results, removing hardcoded Adult text when analyzing arbitrary CSV files.
