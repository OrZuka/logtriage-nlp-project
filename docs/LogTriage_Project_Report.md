# LogTriage Project Report

## 1. Motivation and use case
In an e-commerce microservice system, checkout failures generate logs across gateway, checkout, payment, order, inventory, database, and message-queue services. SREs and backend engineers need to quickly identify the failure cause, affected service, severity, recommended action, and supporting evidence lines.

## 2. ML task definition
**Input:** a multi-line microservice log session.  
**Output:** structured triage JSON:
```json
{
  "failure_cause": "payment_gateway_timeout",
  "affected_service": "payment-service",
  "severity": "high",
  "recommended_action": "check_payment_gateway_and_retry",
  "evidence_lines": [4, 5]
}
```

## 3. Failure taxonomy
The fixed taxonomy contains 12 failure causes, 7 affected services, 4 severity levels, and 12 recommended actions. The full taxonomy is in `data/metadata/label_taxonomy.json`.

## 4. Synthetic data generation method
The dataset is generated from controlled attributes: failure cause, affected service, severity, recommended action, and noise level. Each generated session has 7-12 numbered log lines and 2-4 evidence lines. The recommended LLM generation and validation prompts are stored under `prompts/`.

## 5. Dataset description
- Total examples: 5000
- Train / validation / test split: 3500 / 750 / 750
- Average log length: 7.84 lines
- Average evidence lines: 2.89
- Validation issues found by schema checks: 0

## 6. Models and baselines
Implemented baselines:
1. Rule-based keyword baseline.
2. TF-IDF + Logistic Regression classical baseline.

Implemented but API-key dependent:
1. Zero-shot LLM baseline.
2. Few-shot LLM baseline.

Recommended next training experiment:
1. DistilBERT / RoBERTa multi-output classifier.

## 7. Evaluation metrics
The project reports accuracy and macro-F1 for failure cause, affected service, severity, and recommended action. It also reports evidence-line precision/recall/F1 and full all-fields exact match.

## 8. Results and analysis
See `results/model_scores.csv`, `results/metrics_table.csv`, `results/evidence_line_scores.json`, and `visuals/model_comparison.png`.

## 9. Limitations
The logs are synthetic and may not fully match production systems. Labels are simplified. Recommended actions are taxonomy-based and not a replacement for expert operations judgment. Evidence-line evaluation is strict and can penalize partially valid explanations. A model may learn keywords instead of true causal reasoning.

## 10. Repository artifacts
The repository contains source code, prompts, data splits, results, visuals, slides, and this report.