# LogTriage Data Validation Report

Generated from `src/validate_data.py` and `src/compare_external_logs.py` on the full 5,000-example corpus.

**Last validated:** after `src/fix_validation_issues.py` — **0 failing records**.

## 1. Validation pipeline

| Check | Implemented | Result (combined corpus) |
|-------|-------------|--------------------------|
| JSON / schema validation | Yes | **0 failures** |
| Taxonomy label validity | Yes | **0 failures** |
| Label coverage | Yes | All causes/actions/severities covered; **5 of 12** taxonomy services unused in data |
| Cause–action consistency | Yes | **0 failures** |
| Evidence-line validation | Yes | **0 failures** |
| Label leakage detection | Yes | **0 failures** (500 leaky sessions fixed) |
| Noise-level consistency | Yes | **0 failures** (2 sessions reclassified to `high`) |
| Duplicate sessions | Yes | **0 duplicates** (1 duplicate pair resolved) |
| External realism (LogHub) | Yes | **Acceptable structural similarity** (3/4 checks) |

Full machine-readable output: `results/data_validation_report.json`

## 2. Fixes applied

| Issue | Count | Fix |
|-------|-------|-----|
| Label leakage (`reason=configuration_error`) | 500 | Replaced with natural reasons (`missing_runtime_config`, etc.) |
| Noise mismatch | 2 | `sess_00599`, `sess_02270` reclassified from `medium` → `high` |
| Duplicate logs | 1 pair | `sess_04681` rewritten with distinct timestamps/content |

Script: `src/fix_validation_issues.py`

## 3. Noise-level measurable properties

| Noise level | Avg log lines | Avg evidence ratio | Avg unrelated services | Avg distractor INFO/WARN lines |
|-------------|---------------|--------------------|------------------------|--------------------------------|
| none | 4.21 | 0.79 | 0.70 | 0.89 |
| low | 6.19 | 0.53 | 1.36 | 2.91 |
| medium | 8.20 | 0.40 | 2.04 | 4.90 |
| high | 10.20 | 0.32 | 2.56 | 6.88 |

## 4. External comparison (LogHub HDFS 2k sample)

Reference: `data/external_refs/loghub_hdfs_2k.log`

| Feature | Synthetic LogTriage | LogHub HDFS |
|---------|---------------------|-------------|
| Timestamp rate | 100% | 100% |
| Avg token count | 7.65 | 10.19 |
| Key-value / structured fields | 91.5% | 0% |
| ERROR line rate | 27.4% | 0% |

Full output: `results/external_log_comparison.json`

## 5. Note on model results

Baseline metrics in `results/` were computed **before** the leakage fix. Re-run `python src/train_baselines.py` for updated scores (configuration_error classification may drop slightly).
