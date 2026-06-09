# LogTriage LLM Evaluation Report

Small-sample evaluation using the OpenAI API on a held-out test subset.

## 1. Run configuration

- **zero-shot:** model `gpt-4o-mini`, n=30 (limit=30), shots=0, temperature=0.0
- **few-shot:** model `gpt-4o-mini`, n=30 (limit=30), shots=4, temperature=0.0

Full test set size is 750; this report uses a **small subset only** to limit API cost.

## 2. Compact results

| Model | OpenAI model | N | Cause F1 | Service F1 | Severity F1 | Action F1 | Full exact | Evidence F1 | Valid JSON |
|-------|--------------|---|----------|------------|-------------|-----------|------------|-------------|------------|
| LLM zero-shot | gpt-4o-mini | 30 | 1.000 | 0.734 | 0.441 | 1.000 | 0.300 | 0.942 | 1.000 |
| LLM few-shot | gpt-4o-mini | 30 | 1.000 | 1.000 | 0.643 | 1.000 | 0.533 | 0.990 | 1.000 |

## 3. Full metrics

| Model | Cause Acc | Cause F1 | Service Acc | Service F1 | Severity Acc | Severity F1 | Action Acc | Action F1 | Full exact | Evid P | Evid R | Evid F1 | Valid JSON |
|-------|-----------|----------|-------------|------------|--------------|-------------|------------|-----------|------------|--------|--------|---------|------------|
| LLM zero-shot | 1.000 | 1.000 | 0.800 | 0.734 | 0.367 | 0.441 | 1.000 | 1.000 | 0.300 | 0.989 | 0.900 | 0.942 | 1.000 |
| LLM few-shot | 1.000 | 1.000 | 1.000 | 1.000 | 0.533 | 0.643 | 1.000 | 1.000 | 0.533 | 0.990 | 0.990 | 0.990 | 1.000 |

## 4. Key findings

- **Best full exact (LLM subset):** LLM few-shot (53.3%)
- **Valid JSON rate:** zero-shot=1.0
- Compare against local baselines in `docs/Model_Evaluation_Report.md` (full 750 test set).
- LLM subset scores are indicative only; rerun with `--limit 0` for full evaluation when budget allows.

## 5. Output artifacts

| File | Description |
|------|-------------|
| `results/llm_*_metrics.csv` | Per-mode metrics |
| `results/llm_*_predictions.jsonl` | Raw and parsed predictions |
| `results/llm_*_run_metadata.json` | Run configuration |
| `results/llm_baseline_metrics_compact.csv` | Combined compact table |
| `results/llm_error_examples.json` | Sample misclassifications |
| `visuals/llm_model_comparison.png` | Macro-F1 comparison chart |
