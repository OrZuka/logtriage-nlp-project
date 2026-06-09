"""Run zero-shot and few-shot LLM baselines on the synthetic LogTriage data.

This script uses the real train/test JSONL files. It is intentionally separated
from local baselines because it requires an API key and can cost money.

Example:
    # Option A: put OPENAI_API_KEY in .env at the project root
    # Option B: export OPENAI_API_KEY=...
    python src/run_llm_baseline.py --mode zero-shot --limit 50
    python src/run_llm_baseline.py --mode few-shot --limit 50 --shots 4

Outputs:
    results/llm_zero-shot_predictions.jsonl
    results/llm_zero-shot_metrics.csv
    results/llm_few-shot_predictions.jsonl
    results/llm_few-shot_metrics.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List

import pandas as pd

from utils import (
    DATA_DIR,
    FIELDS,
    RESULTS_DIR,
    evidence_metrics,
    field_metrics,
    full_exact,
    labels,
    load_env,
    load_split,
    record_text,
)


def load_taxonomy() -> Dict[str, Any]:
    with (DATA_DIR / "label_taxonomy.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Extract first JSON object if extra text was returned.
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return text


def parse_prediction(text: str) -> Dict[str, Any]:
    try:
        obj = json.loads(clean_json_text(text))
    except Exception:
        return {"parse_error": True, "raw": text}
    return obj


def build_prompt(record: Dict[str, Any], taxonomy: Dict[str, Any], examples: List[Dict[str, Any]] | None = None) -> str:
    allowed = {field: taxonomy[field] for field in FIELDS}
    instructions = f"""
You are an SRE assistant for an e-commerce microservice platform.
Given a numbered log session, return ONLY valid JSON with these fields:
- failure_cause: one of {allowed['failure_cause']}
- affected_service: one of {allowed['affected_service']}
- severity: one of {allowed['severity']}
- recommended_action: one of {allowed['recommended_action']}
- evidence_lines: a list of 1-based log line numbers that support the prediction

Do not invent labels. Do not explain outside JSON.
""".strip()
    parts = [instructions]
    if examples:
        parts.append("\nExamples:")
        for ex in examples:
            gold = ex["labels"]
            parts.append("Log session:\n" + record_text(ex))
            parts.append("Answer:\n" + json.dumps(gold, ensure_ascii=False))
    parts.append("\nNow solve this log session:\n" + record_text(record))
    parts.append("Answer:")
    return "\n\n".join(parts)


def call_openai(prompt: str, model: str, temperature: float, max_retries: int = 3) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install openai first: pip install openai") from e
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env in the project root or set the environment variable."
        )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


def sanitize_prediction(pred: Dict[str, Any], taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for field in FIELDS:
        value = pred.get(field)
        clean[field] = value if value in taxonomy[field] else "__INVALID__"
    ev = pred.get("evidence_lines", [])
    if not isinstance(ev, list):
        ev = []
    clean["evidence_lines"] = [int(x) for x in ev if isinstance(x, int) or (isinstance(x, str) and x.isdigit())]
    return clean


def evaluate_llm(
    test_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
    raw_rows: List[str],
    mode: str,
    openai_model: str,
    n_shots: int,
) -> pd.DataFrame:
    rows = []
    row: Dict[str, Any] = {
        "model": f"LLM {mode}",
        "openai_model": openai_model,
        "n_evaluated": len(test_rows),
        "n_shots": n_shots if mode == "few-shot" else 0,
    }
    valid_json = sum(1 for raw in raw_rows if not parse_prediction(raw).get("parse_error"))
    row["valid_json_rate"] = valid_json / len(raw_rows) if raw_rows else 0.0

    for field in FIELDS:
        y_true = labels(test_rows, field)
        y_pred = [p.get(field, "__INVALID__") for p in pred_rows]
        m = field_metrics(y_true, y_pred)
        prefix = field.replace("recommended_action", "action").replace("failure_cause", "cause").replace("affected_service", "service")
        row[f"{prefix}_accuracy"] = m["accuracy"]
        row[f"{prefix}_macro_f1"] = m["macro_f1"]
    row["full_exact"] = full_exact(test_rows, pred_rows)
    ev = evidence_metrics([r["labels"]["evidence_lines"] for r in test_rows], [p.get("evidence_lines", []) for p in pred_rows])
    row["evidence_precision"] = ev["precision"]
    row["evidence_recall"] = ev["recall"]
    row["evidence_f1"] = ev["f1"]
    rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero-shot", "few-shot"], default="zero-shot")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=50, help="Number of test examples to run. Use 0 for full test set.")
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    taxonomy = load_taxonomy()
    train = load_split("train")
    test = load_split("test")
    if args.limit and args.limit > 0:
        test = test[: args.limit]
    examples = train[: args.shots] if args.mode == "few-shot" else None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / f"llm_{args.mode}_predictions.jsonl"
    meta_path = RESULTS_DIR / f"llm_{args.mode}_run_metadata.json"
    preds: List[Dict[str, Any]] = []
    raw_rows: List[str] = []
    with pred_path.open("w", encoding="utf-8") as f:
        for i, record in enumerate(test, start=1):
            prompt = build_prompt(record, taxonomy, examples)
            raw = call_openai(prompt, args.model, args.temperature)
            raw_rows.append(raw)
            parsed = parse_prediction(raw)
            pred = sanitize_prediction(parsed, taxonomy)
            out = {"session_id": record["session_id"], "gold": record["labels"], "prediction": pred, "raw": raw}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            preds.append(pred)
            print(f"{args.mode}: {i}/{len(test)}")

    metrics = evaluate_llm(test, preds, raw_rows, args.mode, args.model, args.shots)
    metrics.to_csv(RESULTS_DIR / f"llm_{args.mode}_metrics.csv", index=False)
    meta = {
        "mode": args.mode,
        "openai_model": args.model,
        "limit": args.limit,
        "n_evaluated": len(test),
        "shots": args.shots if args.mode == "few-shot" else 0,
        "temperature": args.temperature,
        "predictions_file": pred_path.name,
        "metrics_file": f"llm_{args.mode}_metrics.csv",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"\nSaved predictions to {pred_path.name}")
    print(f"Saved metadata to {meta_path.name}")


if __name__ == "__main__":
    main()
