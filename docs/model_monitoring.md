# Model & Dataset Monitoring, Drift Detection, Model Health & Retraining Governance

## 1. Overview & Objectives

The **Batch Monitoring & Governance Subsystem** provides continuous post-deployment surveillance for registered machine learning models across multiple datasets. It systematically monitors data drift, schema compatibility, target drift, performance degradation, fairness disparity shifts, and calibration loss without making silent model updates or performing unauthorized retraining.

```
       Monitoring Batch CSV Upload
                    │
                    ▼
     ┌─────────────────────────────┐
     │ 1. Schema Drift Validator   │ ──► [SCHEMA_DRIFT_DETECTED] (BLOCKED)
     └──────────────┬──────────────┘
                    │ (Compatible)
                    ▼
     ┌─────────────────────────────┐
     │ 2. Feature Drift (PSI/TVD)  │
     │ 3. Data Quality Monitor     │
     │ 4. Target Label Drift       │
     │ 5. Performance Monitor      │
     │ 6. Fairness Drift Monitor   │
     │ 7. Calibration Monitor      │
     └──────────────┬──────────────┘
                    │
                    ▼
     ┌─────────────────────────────┐
     │ 8. Model Health Synthesizer │ ──► [HEALTHY / WARNING / DEGRADED / BLOCKED]
     └──────────────┬──────────────┘
                    │
                    ▼
     ┌─────────────────────────────┐
     │ 9. Retraining Recommender   │ ──► [NO_RETRAINING_NEEDED / RETRAINING_RECOMMENDED]
     └──────────────┬──────────────┘
                    │
                    ▼
          Human Approval Gate ──► retrain_dataset() ──► New Active Version (vN ──► vN+1)
```

---

## 2. Distinguishing Data Concepts

| Data Type | Description & Purpose |
| :--- | :--- |
| **Training Data** | Historic dataset used for candidate estimator fitting and 5-fold cross-validation (`X_train`, `y_train`). |
| **Test Data** | Held-out evaluation split used during initial model selection and baseline audit (`X_test`, `y_test`). |
| **Monitoring Data** | New production/batch dataset uploaded post-deployment to assess real-world data shift and model health. |
| **Production Data** | Real-world feature streams in active inference environments. |

---

## 3. Drift Metrics & Algorithms

### Feature-Level Data Drift
- **Numerical Features:** Population Stability Index (PSI) and Wasserstein Distance.
  $$\text{PSI} = \sum \left( P_{\text{current}} - P_{\text{ref}} \right) \times \ln\left( \frac{P_{\text{current}}}{P_{\text{ref}}} \right)$$
  - Safe epsilon smoothing ($10^{-4}$) is applied to prevent zero division and log undefined states.
- **Categorical Features:** Total Variation Distance (TVD) and Category Distribution Shift.
  $$\text{TVD} = \frac{1}{2} \sum \left| P_{\text{current}}(c) - P_{\text{ref}}(c) \right|$$

### Schema Drift
Detects added columns, missing required features, data type shifts, and unobserved or missing categorical levels. Critical schema incompatibilities set the schema state to `SCHEMA_DRIFT_DETECTED` and block automated model inference.

### Target / Label Drift
Compares baseline positive class rate against batch positive class rate when ground-truth labels are present. If labels are absent, target drift returns `TARGET_LABELS_UNAVAILABLE`.

---

## 4. Performance, Fairness & Calibration Surveillance

- **Performance Monitoring:** Evaluates Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix deltas against baseline.
- **Fairness Monitoring:** Evaluates single-attribute (DPD, DIR, EOD, EOppD, FPRD) and intersectional Equalized Odds Difference. Respects statistical safety rules and reports `INSUFFICIENT_GROUP_COVERAGE` if subgroup sizes drop below configured thresholds.
- **Calibration Monitoring:** Measures Brier score and Expected Calibration Error (ECE) shifts.

---

## 5. Model Health & Retraining Governance

The **Model Health Synthesizer** combines all 7 monitoring dimensions into a consolidated health state:
- **`HEALTHY`**: All active metrics remain stable within configured thresholds.
- **`WARNING`**: Feature drift warnings or minor non-critical performance/fairness drops observed.
- **`DEGRADED`**: Severe performance drop ($\ge 10\%$), fairness disparity increase ($\ge 0.10$), or data quality collapse.
- **`BLOCKED`**: Critical schema drift or missing model estimator artifacts.

The **Retraining Recommender** outputs `NO_RETRAINING_NEEDED`, `RETRAINING_RECOMMENDED`, or `RETRAINING_REQUIRED` with an explicit list of failure reasons.

### Governance Approval Gate
The framework **never retrains automatically** unless explicitly approved. When retraining is recommended, the user must click **[ APPROVE RETRAINING ]** in the Streamlit UI or pass an explicit CLI approval flag, triggering `retrain_dataset()` and incrementing the active model version (`vN` → `vN+1`).

---

## 6. Evidence Registry & Artifacts

All monitoring executions are logged in central registry [registry/monitoring_registry.json](file:///d:/personal_proj/registry/monitoring_registry.json) and saved as immutable JSON evidence files under:
`evidence/monitoring/<dataset_id>/<monitoring_run_id>/`
- `monitoring_report.json`
- `drift_report.json`
- `performance_monitoring.json`
- `fairness_monitoring.json`
- `calibration_monitoring.json`
- `health_report.json`
- `retraining_recommendation.json`
