# LogTriage Model Evaluation Report

Generated from real local test-set evaluation on the synthetic e-commerce microservice log dataset (post-validation fix, n=750 test).

## 1. Dataset Summary

| Statistic | Value |
|-----------|-------|
| total_examples | 5000.0 |
| train_examples | 3500.0 |
| validation_examples | 750.0 |
| test_examples | 750.0 |
| failure_causes | 10.0 |
| affected_services | 7.0 |
| severity_levels | 4.0 |
| recommended_actions | 10.0 |
| avg_log_lines | 7.67 |
| validation_issues | 0.0 |

Data source: pre-generated synthetic JSONL splits in `data/` (train/valid/test). Label leakage in 500 configuration_error sessions was fixed before this run.

## 2. Models Evaluated

1. **Majority** — always predicts the most frequent training label per field.
2. **Rules** — keyword/pattern rule baseline with cause→action mapping.
3. **TF-IDF LogReg** — word bigram TF-IDF + One-vs-Rest Logistic Regression.
4. **TF-IDF SVM** — word bigram TF-IDF + Linear SVM.
5. **TF-IDF NB** — word bigram TF-IDF + Multinomial Naive Bayes.
6. **Char SVM** — character 3–4 gram TF-IDF + linear SVM (SGD hinge).

## 3. Evaluation Metrics

Per output field: **accuracy** and **macro-F1** for failure cause, affected service, severity, and recommended action.
Structured task: **full exact match** (all 4 labels correct) and **evidence-line F1** (heuristic baseline).

## 4. Compact Results (Test Set, n=750)

| Model | Cause F1 | Service F1 | Severity F1 | Action F1 | Full Exact | Evidence F1 |
|-------|----------|------------|-------------|-----------|------------|-------------|
| Majority | 0.018 | 0.066 | 0.148 | 0.018 | 0.0% | 0.822 |
| Rules | 0.804 | 0.253 | 0.481 | 0.804 | 17.5% | 0.861 |
| TF-IDF LogReg | 1.000 | 1.000 | 0.585 | 1.000 | 54.5% | 0.886 |
| TF-IDF SVM | 1.000 | 1.000 | 0.640 | 1.000 | **56.1%** | 0.886 |
| TF-IDF NB | 1.000 | 1.000 | 0.421 | 1.000 | 49.2% | 0.886 |
| Char SVM | 1.000 | 1.000 | 0.607 | 1.000 | 55.6% | 0.886 |

## 5. Full Metrics Table

| Model | Cause Acc | Cause F1 | Service Acc | Service F1 | Severity Acc | Severity F1 | Action Acc | Action F1 | Full Exact | Evid P | Evid R | Evid F1 |
|-------|-----------|----------|-------------|------------|--------------|-------------|------------|-----------|------------|--------|--------|---------|
| Majority | 0.100 | 0.018 | 0.300 | 0.066 | 0.421 | 0.148 | 0.100 | 0.018 | 0.0% | 0.767 | 0.886 | 0.822 |
| Rules | 0.825 | 0.804 | 0.412 | 0.253 | 0.453 | 0.481 | 0.825 | 0.804 | 17.5% | 0.795 | 0.941 | 0.861 |
| TF-IDF LogReg | 1.000 | 1.000 | 1.000 | 1.000 | 0.545 | 0.585 | 1.000 | 1.000 | 54.5% | 0.838 | 0.940 | 0.886 |
| TF-IDF SVM | 1.000 | 1.000 | 1.000 | 1.000 | 0.561 | 0.640 | 1.000 | 1.000 | 56.1% | 0.838 | 0.940 | 0.886 |
| TF-IDF NB | 1.000 | 1.000 | 1.000 | 1.000 | 0.492 | 0.421 | 1.000 | 1.000 | 49.2% | 0.838 | 0.940 | 0.886 |
| Char SVM | 1.000 | 1.000 | 1.000 | 1.000 | 0.556 | 0.607 | 1.000 | 1.000 | 55.6% | 0.838 | 0.940 | 0.886 |

## 6. Key Findings

- **Best full exact match:** TF-IDF SVM (56.1%)
- **Best severity macro-F1:** TF-IDF SVM (0.640)
- **Rules baseline** captures lexical cause/action patterns (~80% cause F1) but struggles on affected service (~25% F1).
- **TF-IDF models** still reach perfect cause/service/action on this synthetic test set due to strong lexical cues (leakage fix had minimal impact on overall scores).
- **Severity** remains the hardest field (best macro-F1 ~0.64 for TF-IDF SVM).

## 7. Output Artifacts

| File | Description |
|------|-------------|
| `results/baseline_metrics_all.csv` | Full per-model metrics |
| `results/baseline_metrics_compact.csv` | Summary metrics for comparison |
| `results/baseline_predictions_test.jsonl` | Per-example predictions for all models |

## 8. Limitations

- Synthetic logs contain strong lexical cues that inflate TF-IDF performance.
- Severity labels are harder and more context-dependent than cause/action.
- Evidence-line scoring uses a shared heuristic, not a learned evidence model.
