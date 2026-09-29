# Intersectional Bias Auditing with Explicit Accuracy Trade-off Reporting

## 1. Project Title
**Intersectional Bias Auditing with Explicit Accuracy Trade-off Reporting**
*An Academic AI/ML Framework for Demographic Fairness Evaluation, Bias Mitigation, and Multi-Objective Trade-off Analysis.*

---

## 2. Problem Statement
Machine learning classification models deployed in sensitive socio-economic domains (such as credit scoring, employment screening, and healthcare allocation) frequently exhibit demographic bias. Traditional fairness auditing approaches evaluate protected attributes—such as **Sex** or **Race**—in isolation. However, individuals simultaneously possess multiple overlapping social identities (e.g., Black Females, Elderly Immigrants). Evaluating attributes independently creates an "intersectionality blindspot," where compound disparities remain hidden beneath single-attribute averages. Furthermore, standard bias mitigation techniques modify model parameters or decision boundaries, causing unquantified degradation in predictive performance and probability calibration.

---

## 3. Objectives
1. **Decoupled Data Architecture**: Enforce strict separation between feature matrix $X$, target label $y$, and sensitive attributes $A$.
2. **Single & Intersectional Fairness Auditing**: Quantify group-level selection rates, True Positive Rates (TPR), False Positive Rates (FPR), False Negative Rates (FNR), and disparity bounds across single and combined subgroups ($\text{Sex} \times \text{Race}$).
3. **Statistical Safety Thresholding**: Establish a threshold policy ($N \ge 30$) to separate statistically valid primary groups from low-sample subgroups subject to sampling noise.
4. **Targeted Bias Mitigation**: Apply Fairlearn `ExponentiatedGradient` in-processing reductions under `EqualizedOdds` fairness constraints.
5. **Probability Calibration Assessment**: Evaluate Brier Score and Expected Calibration Error (ECE) across uniform probability bins ($n=10$) to assess probability reliability changes post-mitigation.
6. **Explicit Three-Way Trade-off Reporting**: Report exact Pareto trade-offs comparing fairness gains against predictive performance loss and calibration degradation.
7. **Interactive Web Dashboard**: Provide an interactive 8-page Streamlit web app for visual auditing and academic demonstration.
8. **Automated Testing Suite**: Maintain a 100% passing `pytest` suite for regression testing and end-to-end reproducibility.

---

## 4. Dataset
- **Name**: UCI Adult Census Income Dataset (`adult.csv`)
- **Total Raw Records**: 32,561 rows
- **Cleaned Dataset**: 30,162 rows (after dropping missing values)
- **Target Variable ($y$)**: Binary income indicator (`<=50K` $\to 0$, `>50K` $\to 1$)
- **Feature Matrix ($X$)**: 91 one-hot encoded features (excluding protected attributes `sex` and `race`)
- **Protected Attributes ($A$)**: `sex` (`Female`, `Male`) and `race` (`Amer-Indian-Eskimo`, `Asian-Pac-Islander`, `Black`, `Other`, `White`)
- **Train/Test Split**: 80% Train ($N = 24,129$), 20% Test ($N = 6,033$), stratified with fixed seed `random_state=42`.

---

## 5. Technology Stack
- **Language**: Python 3.14+
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`
- **Fairness & Mitigation**: `fairlearn`
- **Visualization**: `matplotlib`, `seaborn`
- **Interactive Web App**: `streamlit`
- **Unit Testing**: `pytest`

---

## 6. System Architecture
```
.
├── config/
│   └── default_config.json        # Pipeline configuration & random seeds
├── data/
│   ├── raw/                       # Raw input dataset (adult.csv)
│   └── processed/                 # Generated result JSON and CSV files
├── src/                           # Main Python package
│   ├── data/                      # Loader and preprocessor modules
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── models/                    # Baseline classifier module
│   │   └── classifier.py
│   ├── fairness/                  # Auditing, calibration & trade-off modules
│   │   ├── single_attribute.py
│   │   ├── intersectional.py
│   │   ├── calibration.py
│   │   └── trade_off.py
│   ├── mitigation/                # Fairlearn reduction wrapper
│   │   └── mitigator.py
│   ├── visualization/            # Static figure generation
│   │   └── plots.py
│   ├── run_phase2.py              # Baseline audit execution
│   ├── run_phase3.py              # Intersectional audit execution
│   ├── run_phase4.py              # Bias mitigation execution
│   └── run_phase5.py              # Three-way trade-off & calibration execution
├── dashboard/                     # Interactive Streamlit Web App
│   ├── app.py                     # Main dashboard application
│   └── utils.py                   # Dynamic data loading & formatting utilities
├── tests/                         # Automated test suite
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_fairness.py
│   ├── test_intersectional.py
│   ├── test_mitigation.py
│   ├── test_tradeoff.py
│   ├── test_dashboard.py
│   └── test_reproducibility.py
├── results/figures/               # Output figure charts (PNG)
├── requirements.txt               # Python package dependencies
└── README.md                      # Complete project documentation
```

---

## 7. ML Methodology & Data Leakage Guardrails
1. **Decoupled Architecture**: Protected attributes ($A$) are explicitly dropped from feature matrix $X$ prior to model training, preventing direct feature dependence while remaining accessible for fairness evaluation.
2. **Strict Training/Test Isolation**: Preprocessing scaling parameters and Fairlearn mitigation models are fitted strictly on training data ($X_{\text{train}}, y_{\text{train}}$). Test set ($X_{\text{test}}, y_{\text{test}}$) is strictly reserved for evaluation.
3. **Deterministic Seed Control**: Fixed seed (`random_state=42`) is applied across dataset splitting, classifier training, and Fairlearn prediction sampling for 100% deterministic reproducibility.

---

## 8. Baseline Model
- **Algorithm**: Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- **Solver**: `lbfgs` (`max_iter=1000`, `random_state=42`)
- **Performance ($N = 6,033$ Test Set)**:
  - **Accuracy**: `0.8399` (83.99%)
  - **Precision**: `0.7267` (72.67%)
  - **Recall**: `0.5719` (57.19%)
  - **F1-Score**: `0.6401`
  - **ROC-AUC**: `0.8841`
  - **Confusion Matrix**: `TN=4,208 | FP=323 | FN=643 | TP=859`

---

## 9. Single-Attribute Fairness Auditing
Evaluates selection rates and disparity bounds across `Sex` and `Race` independently:
- **Demographic Parity Difference**: $\max_g(\text{SR}_g) - \min_g(\text{SR}_g)$
- **Disparate Impact Ratio**: $\frac{\min_g(\text{SR}_g)}{\max_g(\text{SR}_g)}$
- **Equalized Odds Difference**: $\max( \Delta\text{TPR}, \Delta\text{FPR} )$
- **Results for Sex**: Demographic Parity Diff = `0.1618`, Disparate Impact Ratio = `0.3494`, Equalized Odds Diff = `0.0697`.
- **Results for Race**: Demographic Parity Diff = `0.2319`, Disparate Impact Ratio = `0.1683`, Equalized Odds Diff = `0.3551`.

---

## 10. Intersectional Fairness Auditing
Constructs combined `Sex` $\times$ `Race` subgroup identities (10 distinct subgroups discovered in test data):
- **Statistical Safety Threshold Policy**: Subgroups with sample size $N \ge 30$ (7 primary groups) evaluate primary disparity bounds. Low-sample groups ($N < 30$, 3 groups) are reported in full transparency but flagged to prevent small-sample noise from distorting conclusions.
- **Intersectional Disparities (Primary Groups)**:
  - **Demographic Parity Difference**: `0.3171`
  - **Disparate Impact Ratio**: `0.1333`
  - **Equalized Odds Difference**: `0.4825`
  - **Equal Opportunity Difference (TPR)**: `0.4825`
- **Most Disadvantaged Subgroup**: `Male + Amer-Indian-Eskimo` (Selection Rate = `0.0488`).
- **Highest Performing Subgroup**: `Male + Asian-Pac-Islander` (Selection Rate = `0.3659`).

---

## 11. Bias Mitigation
- **Algorithm**: Fairlearn `ExponentiatedGradient` in-processing reduction.
- **Constraint**: `EqualizedOdds` (enforces equalized true positive and false positive rates across intersectional groups).
- **Sensitive Features**: Intersectional `Sex × Race` combination series.
- **Probability Extraction**: Uses Fairlearn's `_pmf_predict` method to output continuous ensemble policy probabilities $P(\hat{y}=1 \mid X)$ in $[0, 1]$.

---

## 12. Calibration Analysis
Evaluates model probability reliability:
- **Brier Score**: Mean squared error between prediction probabilities and binary targets ($\text{BS} = \frac{1}{N} \sum (y_{\text{prob}} - y_{\text{true}})^2$).
- **Expected Calibration Error (ECE)**: Weighted average absolute difference between empirical accuracy and predicted confidence across 10 uniform probability bins ($[0.0, 0.1), \dots, [0.9, 1.0]$).

---

## 13. Accuracy / Fairness Trade-off
Mitigation alters decision boundaries to satisfy subgroup parity constraints, imposing quantifiable costs on predictive performance and calibration:

| Metric Category | Metric Name | Baseline Model | Mitigated Model | Absolute Change ($\Delta$) | Impact Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Performance** | Accuracy | `0.8399` | `0.7578` | `-0.0820` (-9.8%) | Worsened |
| | Precision | `0.7267` | `0.5936` | `-0.1331` (-18.3%) | Worsened |
| | Recall | `0.5719` | `0.0866` | `-0.4854` (-84.9%) | Worsened |
| | F1-Score | `0.6401` | `0.1511` | `-0.4890` (-76.4%) | Worsened |
| | ROC-AUC | `0.8841` | `0.5335` | `-0.3506` (-39.7%) | Worsened |
| **Calibration** | Brier Score | `0.1128` | `0.2156` | `+0.1028` (+91.1%) | Worsened |
| | ECE | `0.0151` | `0.2166` | `+0.2015` (+1337.2%)| Worsened |
| **Fairness** | Demographic Parity Diff | `0.3171` | `0.0274` | **`-0.2897` (-91.4%)** | **Improved** |
| | Disparate Impact Ratio | `0.1333` | `0.4393` | **`+0.3060` (+229.5%)**| **Improved** |
| | Equalized Odds Diff | `0.4825` | `0.1667` | **`-0.3158` (-65.5%)** | **Improved** |
| | Equal Opportunity Diff | `0.4825` | `0.1667` | **`-0.3158` (-65.5%)** | **Improved** |
| | FPR Difference | `0.1455` | `0.0146` | **`-0.1310` (-90.0%)** | **Improved** |

---

## 14. Authoritative Results Summary
- **Baseline Accuracy**: `0.8399` (83.99%)
- **Mitigated Accuracy**: `0.7578` (75.78%)
- **Baseline Equalized Odds Disparity**: `0.4825`
- **Mitigated Equalized Odds Disparity**: `0.1667` (**65.5% disparity reduction**)
- **Baseline Brier Score**: `0.1128` | **Mitigated Brier Score**: `0.2156`
- **Baseline ECE**: `0.0151` | **Mitigated ECE**: `0.2166`
- **Serialized Output Files**: `data/processed/final_results.json`, `data/processed/final_results.csv`, `data/processed/tradeoff_results.json`, `data/processed/tradeoff_summary.csv`.

---

## 15. Streamlit Interactive Dashboard
Run locally with:
```bash
streamlit run dashboard/app.py
```
### Dashboard Pages:
1. **Overview**: Project title, problem summary, key metric cards (Baseline vs Mitigated Accuracy, EOD, Brier Score).
2. **Baseline Model**: Logistic Regression parameters, performance metrics, test confusion matrix, metric definitions.
3. **Single-Attribute Audit**: Sex and Race audit dropdowns, group selection rates, TPR/FPR/FNR metrics, disparity measures.
4. **Intersectional Audit**: Sex × Race group metrics table, primary vs low-sample flags, interactive charts, statistical caution statement.
5. **Bias Mitigation**: Fairlearn ExponentiatedGradient details, before vs after fairness table, comparative figure displays.
6. **Trade-off Analysis**: Comprehensive Before vs After table across Performance, Fairness, and Calibration with directional badges (`Improved`/`Worsened`).
7. **Calibration**: Baseline vs Mitigated Brier Score & ECE cards, reliability curves, probability distribution histograms.
8. **Conclusion**: Scientifically cautious synthesis, fairness gain vs performance cost analysis, operational limitations.

---

## 16. Project Structure
See Section 6 above.

---

## 17. Installation
1. **Clone Repository & Navigate to Directory**:
   ```bash
   cd d:/personal_proj
   ```
2. **Create & Activate Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 18. Execution Commands
1. **Run Baseline Model & Single-Attribute Audit (Phase 2)**:
   ```bash
   python src/run_phase2.py
   ```
2. **Run Intersectional Fairness Audit (Phase 3)**:
   ```bash
   python src/run_phase3.py
   ```
3. **Run Bias Mitigation & Trade-off Reporting (Phase 4)**:
   ```bash
   python src/run_phase4.py
   ```
4. **Run Three-Way Trade-off & Calibration Analysis (Phase 5)**:
   ```bash
   python src/run_phase5.py
   ```
5. **Run Data Evidence & Quality Gates (Phase 8)**:
   ```bash
   python src/run_phase8.py
   ```
6. **Run Independent Reference & Oracle Validation (Phase 9)**:
   ```bash
   python src/run_phase9.py
   ```
7. **Run Fairness Criteria Compatibility & Conflict Analysis (Phase 10)**:
   ```bash
   python src/run_phase10.py
   ```
8. **Launch Interactive Streamlit Web App (Phase 6)**:
   ```bash
   streamlit run dashboard/app.py
   ```

---

## 19. Testing
Run the complete automated test suite:
```bash
python -m pytest -v
```
- **Total Test Count**: **90 tests passing** (100% pass rate).
- **Test Modules**: `test_data.py`, `test_models.py`, `test_fairness.py`, `test_intersectional.py`, `test_mitigation.py`, `test_tradeoff.py`, `test_dashboard.py`, `test_reproducibility.py`, `test_quality_gates.py`, `test_phase8_evidence.py`, `test_reference_oracle.py`, `test_criteria_contracts.py`, `test_criteria_compatibility.py`, `test_conflict_policy.py`, `test_criteria_telemetry.py`, `test_failure_contract.py`, `test_degraded_mode.py`, `test_recovery.py`, `test_reconciliation.py`, `test_replay.py`, `test_nt1.py`, `test_nt2.py`, `test_nt3.py`, `test_nt4.py`, `test_nt5.py`.

---

## 20. Limitations
1. **Statistical Volatility in Low-Sample Groups**: Subgroups with $N < 30$ produce noisy rate estimates and are excluded from primary disparity calculations.
2. **In-Processing Calibration Shift**: Enforcing equalized odds parity alters decision thresholds, shifting predicted probabilities away from baseline likelihoods.
3. **Dataset Scope**: Audit results are specific to the UCI Adult Census Income dataset and cannot be generalized to other domain populations without re-auditing.

---

## 21. Future Scope
1. **Post-Processing Calibration Repair**: Integrate post-processing probability recalibration (e.g., Platt Scaling or Isotonic Regression) post-mitigation.
2. **Multi-Attribute Intersectional Auditing**: Extend intersectional group definition to 3+ protected attributes (e.g., Sex × Race × Age).
3. **Causal Fairness Modeling**: Incorporate causal directed acyclic graphs (DAGs) to evaluate counterfactual fairness.

---

## 22. Phase 8 — Evidence, Provenance & Data Quality Engineering
Phase 8 establishes a formal evidence subsystem that records and validates raw dataset identity, SHA-256 file hashes, schema snapshots, 15 comprehensive quality gates, and source-to-claim mappings.

### Key Evidence Artifacts:
- **`evidence/dataset/dataset_manifest.json`**: Captures raw dataset metadata, local path (`data/raw/adult.csv`), SHA-256 hash (`9007ff524acf0776b9598a0ed883058df91f59c003e78cb2cddd88f87535fc86`), file size ($3,551,168$ bytes), raw row count ($32,561$), column count ($15$), target column (`income`), protected attributes (`sex`, `race`), and provenance status (`ACQUIRED_UCI_REPOSITORY`).
- **`evidence/dataset/schema_snapshot.json`**: Dynamic schema inspection recording column dtypes, unique value counts, null counts, null percentages, sample values, and functional roles (`target`, `protected`, `feature`).
- **`evidence/dataset/quality_report.json`**: Execution timestamp, SHA-256 hash, duplicate row measurements, missing cell percentages, and status breakdown across 15 automated quality gates (**15 / 15 PASS**).
- **`evidence/dataset/source_to_claim_map.json`**: Formal verification mapping linking project dataset and fairness claims (e.g., `DATA-001`, `AUDIT-001`, `MIT-001`, `CAL-001`) to evidence files.

### Execution Command:
```bash
python src/run_phase8.py
```

---

## 23. Phase 9 — Independent Reproducibility & Expected-Result Oracle
Phase 9 implements an independent, self-contained reference audit module (`reference/intersectional_reference.py`) that operates without importing or invoking production pipeline code. It evaluates the production audit implementation against a frozen expected-result oracle (`evidence/reference/intersectional_expected_results.json`).

### Key Reference & Oracle Artifacts:
- **`reference/intersectional_reference.py`**: Independent calculations of Sex × Race subgroup labels, group sample counts, selection rates, TPR, FPR, FNR, and disparity bounds.
- **`evidence/reference/intersectional_expected_results.json`**: Frozen expected-result oracle containing authoritative Phase 3 benchmark targets.
- **`evidence/reference/reproducibility_report.json` / `.csv`**: Machine-readable reconciliation report asserting 100% agreement between production and reference implementations within $10^{-4}$ tolerance.
- **`evidence/reference/resource_profile.json`**: Measured resource envelope detailing dataset hash, test sample size ($6,033$), Python version, OS platform, and execution duration.
- **`evidence/reference/reproducibility_protocol.md`**: Step-by-step reproduction guide.

### Execution Commands:
```bash
# Run one-command Phase 9 reproducibility validation:
python src/run_phase9.py

# Run standalone independent reference audit:
python reference/run_reference_audit.py

# Run production vs. reference comparison:
python reference/compare_with_production.py
```

---

## 24. Phase 10 — Fairness Criteria Compatibility & Explicit Conflict Reporting
Phase 10 implements FR-4 compliance by establishing formal input, output, error, configuration, and telemetry contracts for evaluating 5 supported fairness criteria (`demographic_parity`, `disparate_impact`, `equal_opportunity`, `equalized_odds`, `fpr_difference`).

### Key Principles & Features:
- **No-Silent-Priority Policy (`report_all_do_not_silently_prioritize`)**: Prevents the system from automatically prioritizing, optimizing, or hiding failing criteria. When criteria thresholds conflict, the compatibility engine returns `CONFLICT_DETECTED` and sets `operator_decision_required = True`.
- **Formal Contracts**: Input contract validation (`src/contracts/fairness_criteria_contract.py`), output schema construction (`src/contracts/criteria_output_contract.py`), and structured error codes (`src/contracts/error_contract.py`).
- **Telemetry System**: Writes persistent JSONL telemetry event logs under `evidence/telemetry/fairness_criteria.jsonl`.
- **Evidence Artifacts**: Generates `evidence/fairness/criteria_compatibility_report.json` and `.csv`.
- **Academic Terminology**: Uses precise terminology (*"criterion threshold conflict"*, *"simultaneous satisfaction not demonstrated"*) rather than claiming mathematical impossibility without formal proof.

### Execution Command:
```bash
python src/run_phase10.py
```

---

## 25. Phase 11 — Reliability, Failure Handling, Recovery & Negative-Test Campaign
Phase 11 implements FR-7 compliance by establishing a controlled reliability subsystem with five negative tests (NT-1 through NT-5), each with formal detection, safe/degraded state entry, recovery execution, output reconciliation, and persistent evidence retention.

### Negative Test Campaign:
| Test | Failure Condition | Safe Response | Recovery |
|------|------------------|---------------|----------|
| NT-1 | Missing intersectional group | `BLOCK_INTERSECTIONAL_RESULT` | Restore approved dataset & re-audit |
| NT-2 | Hidden mitigation cost | `BLOCK_MITIGATION_CLAIM` | Recompute complete trade-off metrics |
| NT-3 | Silent criteria selection | `ENTER_BLOCKED_STATE` | Require explicit operator policy |
| NT-4 | Sensitive evidence exposure | `ACCESS_DENIED_BLOCK_EXPORT` | Sanitize to permitted fields only |
| NT-5 | Schema evolution / malformed input | `REJECT_INPUT_BLOCK_TRAINING` | Restore approved schema fixture |

### Key Subsystems:
- **`src/reliability/failure_detector.py`**: Detects NT-1 through NT-5 failure conditions.
- **`src/reliability/degraded_mode.py`**: State machine (`NORMAL`, `DEGRADED`, `BLOCKED`, `RECOVERING`, `RECOVERED`, `FAILED`).
- **`src/reliability/recovery_manager.py`**: Executes complete detection→safe-response→recovery→reconciliation workflows.
- **`src/reliability/reconciliation.py`**: Reconciles recovered metrics within 1e-4 tolerance.
- **`src/reliability/replay.py`**: Verifies 3-run deterministic pipeline replay.
- **`src/reliability/health_check.py`**: Pre-pipeline operating health checker.

### Evidence Artifacts:
- `evidence/reliability/failure_events.jsonl`
- `evidence/reliability/recovery_events.jsonl`
- `evidence/reliability/reconciliation_events.jsonl`
- `evidence/reliability/replay_report.json`

### Execution Commands:
```bash
python src/run_negative_tests.py
python src/run_phase11.py
```

### Testing:
- **Total Test Count**: **90 tests passing** (100% pass rate, including 14 new Phase 11 tests).

---

## 26. Conclusion
The bias mitigation algorithm achieved a substantial reduction in observed intersectional fairness disparity (Equalized Odds difference reduced by **65.5%**, from `0.4825` to `0.1667`). However, this fairness improvement was accompanied by quantifiable costs in predictive accuracy (`83.99%` $\to$ `75.78%`), recall (`57.19%` $\to$ `8.66%`), and probability calibration (Brier Score `0.1128` $\to$ `0.2156`, ECE `0.0151` $\to$ `0.2166`). These findings highlight the fundamental necessity of explicit multi-objective trade-off reporting in algorithmic fairness interventions.

*Observed disparities represent empirical fairness measurements and do not by themselves establish legal or intentional discrimination.*

---

## License
Academic Demonstration Project - All Rights Reserved.

