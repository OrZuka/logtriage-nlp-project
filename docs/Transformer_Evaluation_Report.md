# LogTriage Transformer Evaluation Report

Fine-tuned RoBERTa-family encoder (`AutoModelForSequenceClassification`) on log session text.
One classifier is trained per output field, matching the local baseline protocol.

## 1. Run configuration

- **HuggingFace model:** `distilroberta-base`
- **Device:** cpu
- **Train examples:** 300 (limit=300)
- **Validation examples:** 60
- **Test examples:** 75 (limit=75)
- **Epochs:** 2, batch size=8, max_length=256

Full dataset has 3500 train / 750 test. This run may use a subset for CPU-friendly training.

## 2. Compact results

| Model | Cause F1 | Service F1 | Severity F1 | Action F1 | Full exact | Evidence F1 |
|-------|----------|------------|-------------|-----------|------------|-------------|
| RoBERTa (distilroberta-base) | 0.788 | 0.176 | 0.129 | 0.750 | 0.080 | 0.873 |

## 3. Full metrics

| Model | Cause Acc | Cause F1 | Service Acc | Service F1 | Severity Acc | Severity F1 | Action Acc | Action F1 | Full exact | Evid P | Evid R | Evid F1 |
|-------|-----------|----------|-------------|------------|--------------|-------------|------------|-----------|------------|--------|--------|---------|
| RoBERTa (distilroberta-base) | 0.787 | 0.788 | 0.440 | 0.176 | 0.347 | 0.129 | 0.747 | 0.750 | 0.080 | 0.813 | 0.943 | 0.873 |

## 4. Key findings

- **Full exact match:** 8.0% on n=75 test sessions
- **Hardest field:** severity macro-F1 = 0.129
- Evidence-line F1 uses the same heuristic as other baselines (not a learned evidence head).
- Compare with TF-IDF / LLM reports; subset sizes may differ.

## 5. Output artifacts

| File | Description |
|------|-------------|
| `results/transformer_metrics_compact.csv` | Summary metrics |
| `results/transformer_metrics_all.csv` | Full metrics row |
| `results/transformer_predictions_test.jsonl` | Test predictions |
| `results/transformer_run_metadata.json` | Run configuration |
| `results/transformer_error_examples.json` | Sample errors |
| `models/transformer_*` | Saved per-field checkpoints |
| `visuals/transformer_model_comparison.png` | Macro-F1 bar chart |
