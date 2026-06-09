"""Fine-tune a RoBERTa-family encoder for LogTriage multi-field classification.

Trains one sequence classifier per output field (cause, service, severity, action),
evaluates on a held-out test subset, and writes metrics compatible with other baselines.

Example (CPU-friendly subset):
    pip install torch transformers datasets accelerate
    python src/train_transformer.py --model distilroberta-base --train-limit 400 --test-limit 100 --epochs 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

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
    load_split,
    record_text,
)

DEFAULT_MODEL = "distilroberta-base"


def load_taxonomy() -> Dict[str, List[str]]:
    with (DATA_DIR / "label_taxonomy.json").open("r", encoding="utf-8") as f:
        tax = json.load(f)
    return {k: v for k, v in tax.items() if isinstance(v, list)}


def build_dataset(records: List[Dict[str, Any]], field: str, label2id: Dict[str, int]) -> Dataset:
    return Dataset.from_dict({
        "text": [record_text(r) for r in records],
        "label": [label2id[r["labels"][field]] for r in records],
    })


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels_arr = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels_arr, preds)),
        "macro_f1": float(f1_score(labels_arr, preds, average="macro", zero_division=0)),
    }


def train_field_model(
    field: str,
    model_name: str,
    train_records: List[Dict[str, Any]],
    valid_records: List[Dict[str, Any]],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    args: argparse.Namespace,
) -> tuple[Trainer, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = build_dataset(train_records, field, label2id)
    valid_ds = build_dataset(valid_records, field, label2id)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    valid_ds = valid_ds.map(tokenize, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=cols)
    valid_ds.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    out_dir = MODELS_DIR / f"transformer_{field.replace(' ', '_')}"
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return trainer, tokenizer


def predict_field(
    trainer: Trainer,
    tokenizer: AutoTokenizer,
    records: List[Dict[str, Any]],
    field: str,
    id2label: Dict[int, str],
    args: argparse.Namespace,
) -> List[str]:
    texts = [record_text(r) for r in records]
    preds: List[str] = []
    model = trainer.model
    model.eval()
    device = next(model.parameters()).device
    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i : i + args.batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        batch_preds = logits.argmax(dim=-1).cpu().tolist()
        preds.extend(id2label[p] for p in batch_preds)
    return preds


def evaluate_run(
    test_records: List[Dict[str, Any]],
    preds_by_field: Dict[str, List[str]],
    model_name: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    pred_rows = [{field: preds_by_field[field][i] for field in FIELDS} for i in range(len(test_records))]
    row: Dict[str, Any] = {
        "model": f"RoBERTa ({Path(model_name).name})",
        "hf_model": model_name,
        "n_train": args.train_limit if args.train_limit > 0 else len(load_split("train")),
        "n_test": len(test_records),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }
    for field in FIELDS:
        y_true = [r["labels"][field] for r in test_records]
        y_pred = preds_by_field[field]
        m = field_metrics(y_true, y_pred)
        prefix = field.replace("recommended_action", "action").replace("failure_cause", "cause").replace("affected_service", "service")
        row[f"{prefix}_accuracy"] = m["accuracy"]
        row[f"{prefix}_macro_f1"] = m["macro_f1"]
    row["full_exact"] = full_exact(test_records, pred_rows)
    ev_pred = [evidence_heuristic(r, p) for r, p in zip(test_records, pred_rows)]
    ev_gold = [r["labels"]["evidence_lines"] for r in test_records]
    ev = evidence_metrics(ev_gold, ev_pred)
    row["evidence_precision"] = ev["precision"]
    row["evidence_recall"] = ev["recall"]
    row["evidence_f1"] = ev["f1"]
    return row, pred_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id, e.g. distilroberta-base or roberta-base")
    parser.add_argument("--train-limit", type=int, default=400, help="Max train examples (0 = all)")
    parser.add_argument("--test-limit", type=int, default=100, help="Max test examples (0 = all)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    ensure_dirs()
    taxonomy = load_taxonomy()
    train = load_split("train")
    valid = load_split("valid")
    test = load_split("test")

    if args.train_limit > 0:
        train = train[: args.train_limit]
    if args.test_limit > 0:
        test = test[: args.test_limit]
    if args.train_limit > 0:
        valid = valid[: max(50, min(len(valid), args.train_limit // 5))]

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Model: {args.model}")
    print(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

    preds_by_field: Dict[str, List[str]] = {}
    for field in FIELDS:
        print(f"\n=== Training field: {field} ===")
        label_names = taxonomy[field]
        label2id = {name: i for i, name in enumerate(label_names)}
        id2label = {i: name for name, i in label2id.items()}
        trainer, tokenizer = train_field_model(field, args.model, train, valid, label2id, id2label, args)
        preds_by_field[field] = predict_field(trainer, tokenizer, test, field, id2label, args)
        m = field_metrics([r["labels"][field] for r in test], preds_by_field[field])
        print(f"Test {field}: acc={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f}")

    metrics_row, pred_rows = evaluate_run(test, preds_by_field, args.model, args)
    metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_csv(RESULTS_DIR / "transformer_metrics.csv", index=False)

    compact = metrics_df[[
        "model", "cause_macro_f1", "service_macro_f1", "severity_macro_f1",
        "action_macro_f1", "full_exact", "evidence_f1",
    ]].copy()
    compact.to_csv(RESULTS_DIR / "transformer_metrics_compact.csv", index=False)
    metrics_df.to_csv(RESULTS_DIR / "transformer_metrics_all.csv", index=False)

    meta = {
        "hf_model": args.model,
        "train_limit": args.train_limit,
        "test_limit": args.test_limit,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    (RESULTS_DIR / "transformer_run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pred_path = RESULTS_DIR / "transformer_predictions_test.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for record, pred in zip(test, pred_rows):
            ev = evidence_heuristic(record, pred)
            f.write(json.dumps({
                "session_id": record["session_id"],
                "model": metrics_row["model"],
                "gold": record["labels"],
                "prediction": {**pred, "evidence_lines": ev},
            }, ensure_ascii=False) + "\n")

    print("\nTransformer results:")
    print(compact.to_string(index=False))
    print(f"\nSaved to results/transformer_metrics*.csv and {pred_path.name}")


if __name__ == "__main__":
    main()
