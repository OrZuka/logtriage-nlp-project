"""Run real local baseline experiments on the synthetic LogTriage dataset.

Evaluated models:
- Majority baseline
- Keyword/rule baseline
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + Multinomial Naive Bayes
- Character TF-IDF + Linear SVM

The LLM and transformer experiments are intentionally not reported here unless
executed by their own scripts. This file produces real local test-set metrics.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from utils import (
    DATA_DIR,
    FIELDS,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_dirs,
    evidence_heuristic,
    evidence_metrics,
    field_metrics,
    full_exact,
    labels,
    load_split,
    record_text,
)


def load_taxonomy() -> Dict[str, List[str]]:
    with (DATA_DIR / "label_taxonomy.json").open("r", encoding="utf-8") as f:
        tax = json.load(f)
    return {k: v for k, v in tax.items() if isinstance(v, list)}


CAUSE_RULES = [
    ("authentication_token_expired", ["expired_token", "token validation", "invalid_signature", "status=401", "401", "auth_error"]),
    ("payment_gateway_timeout", ["payment gateway timeout", "provider timeout", "provider_latency", "authorization missing", "payment provider"]),
    ("database_connection_pool_exhaustion", ["db.pool", "connection pool", "failed to acquire database connection", "db_connection_unavailable"]),
    ("database_query_failure", ["sqlstate", "duplicate key", "query error", "database query", "order persistence failed"]),
    ("inventory_out_of_sync", ["inventory", "stock", "reservation", "out_of_sync", "stock sync"]),
    ("message_queue_backlog", ["queue", "backlog", "consumer lag", "kafka", "message"]),
    ("configuration_error", ["configuration", "config", "missing env", "invalid config", "feature flag"]),
    ("resource_exhaustion", ["out of memory", "oom", "cpu", "heap", "disk full", "resource"]),
    ("deployment_regression", ["deployment", "rollback", "new version", "release", "regression"]),
    ("normal_no_issue", ["healthcheck status=ok", "heartbeat ok", "no error", "success"]),
]

ACTION_BY_CAUSE = {
    "normal_no_issue": "no_action_monitor",
    "authentication_token_expired": "rotate_or_check_credentials",
    "payment_gateway_timeout": "check_payment_gateway_and_retry",
    "database_connection_pool_exhaustion": "scale_database_pool",
    "database_query_failure": "check_database_query",
    "inventory_out_of_sync": "sync_inventory_and_retry",
    "message_queue_backlog": "scale_queue_consumers",
    "configuration_error": "rollback_or_fix_configuration",
    "resource_exhaustion": "scale_resource_or_restart_service",
    "deployment_regression": "rollback_deployment",
}


def infer_service(text: str, taxonomy: Dict[str, List[str]]) -> str:
    low = text.lower()
    counts = {svc: low.count(svc.lower()) for svc in taxonomy["affected_service"]}
    best, score = max(counts.items(), key=lambda kv: kv[1])
    return best if score > 0 else taxonomy["affected_service"][0]


def infer_severity(text: str) -> str:
    low = text.lower()
    # Simple interpretable heuristic. Severity is intentionally harder than cause/action.
    err = low.count("error")
    warn = low.count("warn")
    if any(x in low for x in ["critical", "outage", "status=503", "unavailable"]) or err >= 3:
        return "critical"
    if err >= 2 or "timeout" in low or "failed" in low:
        return "high"
    if warn >= 1 or err == 1:
        return "medium"
    return "low"


def rule_predict_one(record: Dict[str, Any], taxonomy: Dict[str, List[str]]) -> Dict[str, str]:
    txt = record_text(record).lower()
    cause_scores = Counter()
    for cause, patterns in CAUSE_RULES:
        for p in patterns:
            if p in txt:
                cause_scores[cause] += 1
    if cause_scores:
        cause = cause_scores.most_common(1)[0][0]
    else:
        cause = "normal_no_issue" if "error" not in txt and "warn" not in txt else taxonomy["failure_cause"][0]
    return {
        "failure_cause": cause,
        "affected_service": infer_service(txt, taxonomy),
        "severity": infer_severity(txt),
        "recommended_action": ACTION_BY_CAUSE.get(cause, taxonomy["recommended_action"][0]),
    }


def build_models() -> Dict[str, Pipeline | DummyClassifier]:
    return {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "TF-IDF LogReg": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000)),
            ("clf", OneVsRestClassifier(LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear"))),
        ]),
        "TF-IDF SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000)),
            ("clf", LinearSVC(class_weight="balanced")),
        ]),
        "TF-IDF NB": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000)),
            ("clf", MultinomialNB()),
        ]),
        "Char SVM": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_features=6000)),
            ("clf", SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3, class_weight="balanced", random_state=42)),
        ]),
    }


def evaluate_prediction_set(model_name: str, test: List[Dict[str, Any]], preds_by_field: Dict[str, List[str]]) -> Dict[str, Any]:
    pred_rows = []
    for i in range(len(test)):
        pred_rows.append({field: preds_by_field[field][i] for field in FIELDS})

    row: Dict[str, Any] = {"model": model_name}
    for field in FIELDS:
        m = field_metrics(labels(test, field), preds_by_field[field])
        prefix = field.replace("recommended_action", "action").replace("failure_cause", "cause").replace("affected_service", "service")
        row[f"{prefix}_accuracy"] = m["accuracy"]
        row[f"{prefix}_macro_f1"] = m["macro_f1"]
    row["full_exact"] = full_exact(test, pred_rows)

    ev_pred = [evidence_heuristic(r, p) for r, p in zip(test, pred_rows)]
    ev_gold = [r["labels"]["evidence_lines"] for r in test]
    ev = evidence_metrics(ev_gold, ev_pred)
    row["evidence_precision"] = ev["precision"]
    row["evidence_recall"] = ev["recall"]
    row["evidence_f1"] = ev["f1"]

    # Store per-example predictions for traceability.
    pred_out = []
    for record, pred, ev_lines in zip(test, pred_rows, ev_pred):
        pred_out.append({
            "session_id": record["session_id"],
            "model": model_name,
            "gold": record["labels"],
            "prediction": {**pred, "evidence_lines": ev_lines},
        })
    return row, pred_out


def main() -> None:
    ensure_dirs()
    taxonomy = load_taxonomy()
    train = load_split("train")
    valid = load_split("valid")
    test = load_split("test")
    train_full = train  # keep validation separate; final metrics are reported on test
    x_train = [record_text(r) for r in train_full]
    x_test = [record_text(r) for r in test]

    all_rows = []
    all_predictions = []

    # Majority and TF-IDF/classical models.
    for model_name, prototype in build_models().items():
        preds_by_field: Dict[str, List[str]] = {}
        for field in FIELDS:
            # Recreate model per output field so fitted states do not collide.
            model = build_models()[model_name]
            model.fit(x_train, labels(train_full, field))
            preds = list(model.predict(x_test))
            preds_by_field[field] = preds
            joblib.dump(model, MODELS_DIR / f"{model_name.lower().replace(' ', '_').replace('+','plus')}_{field}.joblib")
        row, pred_out = evaluate_prediction_set(model_name, test, preds_by_field)
        all_rows.append(row)
        all_predictions.extend(pred_out)

    # Rule baseline.
    rule_rows = [rule_predict_one(r, taxonomy) for r in test]
    preds_by_field = {field: [p[field] for p in rule_rows] for field in FIELDS}
    row, pred_out = evaluate_prediction_set("Rules", test, preds_by_field)
    all_rows.insert(1, row)
    all_predictions.extend(pred_out)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "baseline_metrics_all.csv", index=False)

    compact_cols = [
        "model", "cause_macro_f1", "service_macro_f1", "severity_macro_f1", "action_macro_f1", "full_exact", "evidence_f1"
    ]
    df[compact_cols].to_csv(RESULTS_DIR / "baseline_metrics_compact.csv", index=False)

    with (RESULTS_DIR / "baseline_predictions_test.jsonl").open("w", encoding="utf-8") as f:
        for row in all_predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Confusion matrix for best real local model by full exact.
    best_model = df.sort_values("full_exact", ascending=False).iloc[0]["model"]
    best_preds = [p for p in all_predictions if p["model"] == best_model]
    y_true = [p["gold"]["failure_cause"] for p in best_preds]
    y_pred = [p["prediction"]["failure_cause"] for p in best_preds]
    cm = confusion_matrix(y_true, y_pred, labels=taxonomy["failure_cause"])
    pd.DataFrame(cm, index=taxonomy["failure_cause"], columns=taxonomy["failure_cause"]).to_csv(RESULTS_DIR / "failure_cause_confusion_matrix.csv")

    print("Real local baseline results on test split:")
    print(df[compact_cols].to_string(index=False))
    print(f"\nBest model by full exact: {best_model}")


if __name__ == "__main__":
    main()
