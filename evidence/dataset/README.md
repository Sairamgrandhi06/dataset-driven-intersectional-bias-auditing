# Dataset Evidence & Quality Reports

This directory contains versioned evidence artifacts describing dataset identity, provenance, schema, quality gates, and source-to-claim mappings.

## Generated Artifacts

- **`dataset_manifest.json`**: Dataset identity, SHA-256 hash, file size, row/column counts, protected attributes, target variable, and provenance status.
- **`schema_snapshot.json`**: Inferred column data types, null counts, unique value counts, and variable roles (`target`, `protected`, `feature`).
- **`quality_report.json`**: Execution timestamp, quality gate pass/fail status across 15 checks, missing values, duplicates, and subgroup distributions.
- **`source_to_claim_map.json`**: Mapping of project dataset and audit claims to their verification evidence sources.
