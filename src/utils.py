"""Shared utilities for the LogTriage project.

The code intentionally keeps the task transparent: every experiment reads the same
synthetic JSONL train/valid/test files and evaluates the same output fields.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
FIELDS = ["failure_cause", "affected_service", "severity", "recommended_action"]


def load_env() -> None:
    """Load environment variables from `.env` in the project root if present."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError as e:
        raise RuntimeError("Install python-dotenv first: pip install python-dotenv") from e
    load_dotenv(env_path)


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path | str, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_split(split: str) -> List[Dict[str, Any]]:
    return read_jsonl(DATA_DIR / f"logtriage_{split}.jsonl")


def record_text(record: Dict[str, Any]) -> str:
    return "\n".join(record["logs"])


def labels(data: Sequence[Dict[str, Any]], field: str) -> List[str]:
    return [r["labels"][field] for r in data]


def field_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evidence_metrics(gold: Sequence[Sequence[int]], pred: Sequence[Sequence[int]]) -> Dict[str, float]:
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        gs, ps = set(g), set(p)
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def full_exact(gold_rows: Sequence[Dict[str, Any]], pred_rows: Sequence[Dict[str, str]]) -> float:
    correct = 0
    for gold, pred in zip(gold_rows, pred_rows):
        gl = gold["labels"]
        if all(gl[f] == pred.get(f) for f in FIELDS):
            correct += 1
    return correct / len(gold_rows) if gold_rows else 0.0


def extract_line_no(line: str, fallback: int) -> int:
    m = re.match(r"\s*\[(\d+)\]", line)
    return int(m.group(1)) if m else fallback


def evidence_heuristic(record: Dict[str, Any], predicted: Dict[str, str] | None = None, max_lines: int = 4) -> List[int]:
    """Simple real evidence-selection baseline.

    It gives priority to ERROR/WARN lines and lines containing terms related to the
    predicted cause/action. This is not an oracle; it is deliberately a simple
    explainability baseline evaluated against gold evidence lines.
    """
    predicted = predicted or {}
    cause = predicted.get("failure_cause", "")
    action = predicted.get("recommended_action", "")
    service = predicted.get("affected_service", "")
    query_terms = set(re.split(r"[_\W]+", " ".join([cause, action, service]).lower()))
    query_terms = {t for t in query_terms if len(t) >= 3 and t not in {"service", "action", "failure"}}

    scored = []
    for i, line in enumerate(record["logs"], start=1):
        low = line.lower()
        score = 0.0
        if "error" in low:
            score += 4
        if "warn" in low:
            score += 2
        if any(x in low for x in ["failed", "failure", "timeout", "exception", "unavailable", "503", "401"]):
            score += 2
        if service and service.lower() in low:
            score += 1.5
        score += sum(0.7 for t in query_terms if t and t in low)
        if score > 0:
            scored.append((score, extract_line_no(line, i)))
    if not scored:
        return [extract_line_no(line, i) for i, line in enumerate(record["logs"][:2], start=1)]
    scored.sort(key=lambda x: (-x[0], x[1]))
    out = sorted({ln for _, ln in scored[:max_lines]})
    return out


def save_metrics_csv(rows: List[Dict[str, Any]], filename: str) -> pd.DataFrame:
    ensure_dirs()
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / filename, index=False)
    return df
