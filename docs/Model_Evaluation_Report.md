# LogTriage Model Evaluation Report

Generated from real local test-set evaluation on the synthetic e-commerce microservice log dataset.

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
| median_log_lines | 8.0 |
| avg_evidence_lines | 3.3 |
| noise_low | 1311.0 |
| noise_medium | 1805.0 |
| noise_high | 1250.0 |
| validation_issues | 0.0 |

Data source: pre-generated synthetic JSONL splits in `data/` (train/valid/test).
All splits passed schema and taxonomy validation with 0 bad records.

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
| Majority | 0.018 | 0.066 | 0.148 | 0.018 | 0.000 | 0.822 |
| Rules | 0.804 | 0.253 | 0.481 | 0.804 | 0.175 | 0.861 |
| TF-IDF LogReg | 1.000 | 1.000 | 0.584 | 1.000 | 0.544 | 0.886 |
| TF-IDF SVM | 1.000 | 1.000 | 0.649 | 1.000 | 0.572 | 0.886 |
| TF-IDF NB | 1.000 | 0.997 | 0.421 | 1.000 | 0.492 | 0.886 |
| Char SVM | 1.000 | 1.000 | 0.609 | 1.000 | 0.552 | 0.886 |

## 5. Full Metrics Table

| Model | Cause Acc | Cause F1 | Service Acc | Service F1 | Severity Acc | Severity F1 | Action Acc | Action F1 | Full Exact | Evid P | Evid R | Evid F1 |
|-------|-----------|----------|-------------|------------|--------------|-------------|------------|-----------|------------|--------|--------|---------|
| Majority | 0.100 | 0.018 | 0.300 | 0.066 | 0.421 | 0.148 | 0.100 | 0.018 | 0.000 | 0.767 | 0.886 | 0.822 |
| Rules | 0.825 | 0.804 | 0.412 | 0.253 | 0.453 | 0.481 | 0.825 | 0.804 | 0.175 | 0.795 | 0.941 | 0.861 |
| TF-IDF LogReg | 1.000 | 1.000 | 1.000 | 1.000 | 0.544 | 0.584 | 1.000 | 1.000 | 0.544 | 0.838 | 0.940 | 0.886 |
| TF-IDF SVM | 1.000 | 1.000 | 1.000 | 1.000 | 0.572 | 0.649 | 1.000 | 1.000 | 0.572 | 0.838 | 0.940 | 0.886 |
| TF-IDF NB | 1.000 | 1.000 | 0.997 | 0.997 | 0.492 | 0.421 | 1.000 | 1.000 | 0.492 | 0.838 | 0.940 | 0.886 |
| Char SVM | 1.000 | 1.000 | 1.000 | 1.000 | 0.552 | 0.609 | 1.000 | 1.000 | 0.552 | 0.838 | 0.940 | 0.886 |

## 6. Key Findings

- **Best full exact match:** TF-IDF SVM (57.2%)
- **Best failure-cause macro-F1:** TF-IDF SVM (1.000)
- **Best severity macro-F1:** TF-IDF SVM (0.649)
- **Majority baseline** achieves 0% full exact match — confirms the task is non-trivial.
- **Rules baseline** captures lexical cause/action patterns (~80% cause F1) but struggles on affected service (~25% F1).
- **TF-IDF models** reach perfect cause/service/action classification on this synthetic test set due to strong lexical cues.
- **Severity** remains the hardest field (best macro-F1 ~0.65 for TF-IDF SVM); it depends on contextual signals beyond keywords.
- **Evidence-line F1** is similar across models (~0.86–0.89) because all use the same heuristic for evidence selection.

## 7. Output Artifacts

| File | Description |
|------|-------------|
| `results/baseline_metrics_all.csv` | Full per-model metrics |
| `results/baseline_metrics_compact.csv` | Summary metrics for comparison |
| `results/baseline_predictions_test.jsonl` | Per-example predictions for all models |
| `results/per_field_metrics.csv` | Accuracy/macro-F1/weighted-F1 per field |
| `results/confusion_*_*.csv` | Confusion matrices per model and field |
| `results/error_examples.json` | Sample misclassification cases |
| `results/failure_cause_confusion_matrix.csv` | Best-model failure-cause confusion matrix |
| `visuals/model_comparison.png` | Bar chart of macro-F1 by field |
| `visuals/full_exact_evidence_f1.png` | Full exact match and evidence F1 |
| `visuals/class_distribution.png` | Failure cause distribution |
| `visuals/severity_distribution.png` | Severity label distribution |

## 8. Limitations

- Synthetic logs contain strong lexical cues that inflate TF-IDF performance.
- Severity labels are harder and more context-dependent than cause/action.
- Evidence-line scoring uses a shared heuristic, not a learned evidence model.
- Results reflect test-set performance only; production logs may differ.
