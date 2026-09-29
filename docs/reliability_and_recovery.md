# Reliability, Failure Handling, Recovery & Negative-Test Campaign

This document describes the Phase 11 reliability subsystem for the Intersectional Bias Auditing project.

## Overview

Phase 11 implements FR-7 compliance by establishing a controlled negative-test campaign (NT-1 through NT-5) that verifies the system's ability to:

1. **Detect** defined failure conditions using structured detectors.
2. **Enter a safe/degraded state** preventing invalid results from being published.
3. **Record** failure events in persistent JSONL evidence logs.
4. **Execute** documented recovery workflows.
5. **Reconcile** recovered outputs against expected benchmarks.
6. **Verify** deterministic pipeline replay and output idempotency.
7. **Retain** all failure, recovery, and reconciliation evidence.

---

## System Operational States

| State | Description | Can Publish |
|-------|-------------|-------------|
| `NORMAL` | Normal pipeline execution | Yes |
| `DEGRADED` | Non-critical component limited; safe subset only | No |
| `BLOCKED` | Result must not be published | No |
| `RECOVERING` | Recovery in progress | No |
| `RECOVERED` | Recovery succeeded; reconciliation passed | Yes |
| `FAILED` | Recovery or reconciliation failed | No |

> **Safety Rule**: The system must never report a final PASS while in `BLOCKED` or `FAILED` state.

---

## Negative Test Campaign

| Test | Failure Condition | Detection | Safe Response | Recovery | Reconciliation |
|------|------------------|-----------|---------------|----------|----------------|
| NT-1 | Missing intersectional group | `detect_nt1_missing_intersections()` | `BLOCK_INTERSECTIONAL_RESULT` | Restore approved dataset & re-audit | PASS |
| NT-2 | Hidden mitigation cost metrics | `detect_nt2_hidden_mitigation_cost()` | `BLOCK_MITIGATION_CLAIM` | Recompute complete trade-off metrics | PASS |
| NT-3 | Incompatible fairness criteria silently selected | `detect_nt3_criteria_conflict()` | `ENTER_BLOCKED_OR_DEFERRED_DECISION_STATE` | Require explicit operator policy decision | PASS |
| NT-4 | Sensitive evidence exposure attempt | `detect_nt4_sensitive_evidence()` | `ACCESS_DENIED_BLOCK_EXPORT` | Sanitize payload to permitted fields only | PASS |
| NT-5 | Malformed / schema-evolution input | `detect_nt5_schema_evolution()` | `REJECT_INPUT_BLOCK_TRAINING` | Restore approved schema fixture | PASS |

---

## NT-1 — Missing Intersectional Groups

**Condition**: An intersectional audit output is missing one or more expected Sex × Race group combinations.

**Detection**: `src/reliability/failure_detector.py → detect_nt1_missing_intersections()`

**Safe Response**: Block publication of intersectional fairness claims. Single-attribute audits remain valid.

**Recovery**:
1. Restore approved dataset fixture (`data/raw/adult.csv`, SHA-256 verified).
2. Re-run `audit_intersectional_attributes()`.
3. Verify all 10 expected Sex × Race group combinations are present.
4. Reconcile total group count against expected value (10 groups).

**Evidence**: `evidence/reliability/failure_events.jsonl`, `recovery_events.jsonl`

---

## NT-2 — Hidden Mitigation Cost

**Condition**: A mitigation report is missing required cost metrics (`recall`, `brier_score`, `expected_calibration_error`).

**Detection**: `detect_nt2_hidden_mitigation_cost()`

**Safe Response**: Block publication of mitigation claim. No result may be declared "mitigation successful" with incomplete cost evidence.

**Recovery**:
1. Reload authoritative `data/processed/tradeoff_results.json`.
2. Verify all mandatory metrics are present: `accuracy`, `recall`, `equalized_odds_difference`, `brier_score`, `expected_calibration_error`.
3. Reconcile recovered values against Phase 5 authoritative benchmarks within 1e-4 tolerance.

---

## NT-3 — Silent Fairness-Criteria Selection

**Condition**: Incompatible fairness criteria threshold conflict detected without explicit operator decision.

**Detection**: `detect_nt3_criteria_conflict()` (integrates with Phase 10 compatibility engine)

**Safe Response**: Enter `BLOCKED/DEFERRED_DECISION_STATE`. Do not silently choose a criterion. Preserve all criterion results.

**Recovery**:
1. Apply explicit `report_all_do_not_silently_prioritize` policy.
2. Confirm `silent_priority_detected = False`.
3. Reconcile policy enforcement result.

---

## NT-4 — Sensitive Evidence Exposure

**Condition**: Export payload contains prohibited PII fields (`raw_ssn`, `raw_name`, `medical_record_number`).

**Detection**: `detect_nt4_sensitive_evidence()`

**Safe Response**: `ACCESS_DENIED` — block export immediately.

**Recovery**:
1. Sanitize export payload, retaining only `permitted_public_fields`.
2. Verify sanitized payload contains zero prohibited PII fields.
3. Reconcile field count.

> **Academic Note**: NT-4 uses entirely **synthetic** data. No real personal information is used anywhere in this project.

---

## NT-5 — Schema Evolution / Malformed Input

**Condition**: Input dataset missing required schema columns (`income`, `sex`, `race`).

**Detection**: `detect_nt5_schema_evolution()`

**Safe Response**: `REJECT_INPUT_BLOCK_TRAINING` — do not train model or generate fairness claims.

**Recovery**:
1. Load approved dataset (`data/raw/adult.csv`).
2. Verify all required schema columns are present.
3. Reconcile schema restoration.

---

## Reconciliation Engine

`src/reliability/reconciliation.py` compares expected vs recovered metric values:

- **Numerical values**: within floating-point tolerance of `1e-4`.
- **String/categorical fields**: exact string equality.
- Returns `PASS` or `MISMATCH`.
- Never silently accepts mismatches — mismatch sets `recovery_status = FAILED`.

---

## Replay / Idempotency

`src/reliability/replay.py` executes 3 independent pipeline runs (Run A, Run B, Run C) and verifies:

- Same dataset SHA-256 hash across all runs.
- Same disparity metric values within `1e-4` tolerance (numerical equality).
- Distinguishes exact byte equality vs numerical equality explicitly.
- Outputs `evidence/reliability/replay_report.json`.

---

## Evidence Retention

All reliability events are persisted under `evidence/reliability/`:

| File | Contents |
|------|----------|
| `failure_events.jsonl` | Structured failure event records for each NT test |
| `recovery_events.jsonl` | Recovery workflow execution records |
| `reconciliation_events.jsonl` | Reconciliation comparison results |
| `replay_report.json` | Deterministic replay verification report |

---

## Limitations

1. **NT-4 authentication is academic only**: The access control check is a controlled fixture evaluation, not production authentication.
2. **NT-1 group list is fixed**: The 10-group Sex × Race expectation list is hardcoded in `failure_detector.py`. Adding new expected groups requires updating the constant.
3. **Replay tolerance**: Numerical equality within 1e-4 is verified; exact float bit equality is reported separately and may differ due to floating-point arithmetic ordering.
