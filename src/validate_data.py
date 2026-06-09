"""Validate LogTriage JSONL datasets before training or release.

Checks:
- JSON/schema and taxonomy validity
- Label coverage across the corpus
- Cause-action consistency
- Evidence-line index and content support
- Label leakage (verbatim taxonomy strings in log text)
- Noise-level consistency via measurable log properties
- Near-duplicate sessions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TAX = json.loads((DATA_DIR / "label_taxonomy.json").read_text(encoding="utf-8"))
RULES = json.loads((DATA_DIR / "validation_rules.json").read_text(encoding="utf-8"))

SERVICE_RE = re.compile(
    r'(?:service[=:"\s]+|"service"\s*:\s*"|service=)([a-z][a-z0-9-]*service)',
    re.IGNORECASE,
)
LEVEL_RE = re.compile(r"\b(ERROR|WARN|WARNING|INFO|DEBUG|CRITICAL|FATAL)\b", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}|\d{6}\s+\d{6}")


def extract_services(logs: List[str]) -> set[str]:
    services: set[str] = set()
    for line in logs:
        services.update(s.lower() for s in SERVICE_RE.findall(line))
    return services


def session_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    logs = record.get("logs", [])
    labels = record.get("labels", {})
    evidence = set(labels.get("evidence_lines", []))
    affected = labels.get("affected_service", "").lower()
    services = extract_services(logs)

    unrelated = services - {affected, "gateway-service"}
    distractors = 0
    errors_outside = 0
    for idx, line in enumerate(logs, start=1):
        low = line.lower()
        if idx not in evidence and LEVEL_RE.search(line):
            level = LEVEL_RE.search(line).group(1).upper()
            if level in {"INFO", "DEBUG", "WARN", "WARNING"}:
                distractors += 1
            if level in {"ERROR", "CRITICAL", "FATAL"}:
                errors_outside += 1

    return {
        "num_lines": len(logs),
        "evidence_ratio": len(evidence) / len(logs) if logs else 0.0,
        "unrelated_services": len(unrelated),
        "distractors": distractors,
        "errors_outside_evidence": errors_outside,
        "has_timestamp": any(TIMESTAMP_RE.search(line) for line in logs),
        "has_level": any(LEVEL_RE.search(line) for line in logs),
    }


def validate_schema(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    for field in RULES["required_fields"]:
        if field not in record:
            issues.append(f"missing_field:{field}")

    logs = record.get("logs")
    if not isinstance(logs, list) or not logs or not all(isinstance(x, str) for x in logs):
        issues.append("invalid_logs")
        return issues

    labels = record.get("labels")
    if not isinstance(labels, dict):
        issues.append("invalid_labels")
        return issues

    for field in RULES["required_label_fields"]:
        if field not in labels:
            issues.append(f"missing_label:{field}")

    noise = record.get("noise_level")
    if noise not in RULES["allowed_noise_levels"]:
        issues.append(f"invalid_noise_level:{noise}")

    log_format = record.get("log_format")
    if log_format not in RULES["allowed_log_formats"]:
        issues.append(f"invalid_log_format:{log_format}")

    return issues


def validate_taxonomy(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    labels = record.get("labels", {})
    for field in ["failure_cause", "affected_service", "severity", "recommended_action"]:
        if labels.get(field) not in TAX[field]:
            issues.append(f"invalid_{field}:{labels.get(field)}")
    return issues


def validate_cause_action(record: Dict[str, Any]) -> List[str]:
    labels = record.get("labels", {})
    cause = labels.get("failure_cause")
    action = labels.get("recommended_action")
    expected = RULES["cause_to_action"].get(cause)
    if expected and action != expected:
        return [f"cause_action_mismatch:{cause}->{action},expected={expected}"]
    return []


def validate_evidence_lines(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    logs = record.get("logs", [])
    labels = record.get("labels", {})
    evidence = labels.get("evidence_lines", [])

    if not isinstance(evidence, list) or not evidence:
        return ["empty_evidence_lines"]

    if len(set(evidence)) != len(evidence):
        issues.append("duplicate_evidence_lines")

    for idx in evidence:
        if not isinstance(idx, int) or idx < 1 or idx > len(logs):
            issues.append(f"bad_evidence_index:{idx}")
            continue
        line = logs[idx - 1].lower()
        if labels.get("failure_cause") == "normal_no_issue":
            if not any(token in line for token in ["healthcheck", "heartbeat", "ok", "info", "monitor"]):
                issues.append(f"weak_evidence_support:line={idx}")
        elif not any(token in line for token in ["error", "warn", "fail", "timeout", "exception", "status=", "rejected", "missing", "backlog", "oom", "sqlstate", "unauthorized", "503", "500", "401"]):
            issues.append(f"weak_evidence_support:line={idx}")

    return issues


def validate_label_leakage(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    labels = record.get("labels", {})
    text = "\n".join(record.get("logs", [])).lower()
    cause = labels.get("failure_cause", "")
    action = labels.get("recommended_action", "")

    if cause and cause != "normal_no_issue":
        if f"reason={cause}" in text.replace(" ", ""):
            issues.append("leakage:failure_cause_in_reason_field")
        elif cause in text:
            issues.append("leakage:failure_cause_verbatim")

    if action and action in text:
        issues.append("leakage:recommended_action_verbatim")

    return issues


def validate_noise_level(record: Dict[str, Any]) -> List[str]:
    noise = record.get("noise_level")
    if noise not in RULES["noise_level_ranges"]:
        return []

    metrics = session_metrics(record)
    bounds = RULES["noise_level_ranges"][noise]
    issues: List[str] = []

    for key, limit in bounds.items():
        value = metrics.get(key.replace("min_", "").replace("max_", ""))
        if value is None:
            continue
        if key.startswith("min_") and value < limit:
            issues.append(f"noise_mismatch:{noise}:{key}:{value}<{limit}")
        if key.startswith("max_") and value > limit:
            issues.append(f"noise_mismatch:{noise}:{key}:{value}>{limit}")

    return issues


def validate_record(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    for fn in (
        validate_schema,
        validate_taxonomy,
        validate_cause_action,
        validate_evidence_lines,
        validate_label_leakage,
        validate_noise_level,
    ):
        issues.extend(fn(record))
    return issues


def label_coverage(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    coverage: Dict[str, Any] = {}
    for field in ["failure_cause", "affected_service", "severity", "recommended_action"]:
        seen = Counter(r["labels"][field] for r in records)
        missing = [label for label in TAX[field] if seen[label] == 0]
        coverage[field] = {
            "seen": len(seen),
            "expected": len(TAX[field]),
            "missing_labels": missing,
            "counts": dict(seen),
        }
    return coverage


def duplicate_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    hashes: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        digest = hashlib.md5("".join(record["logs"]).encode("utf-8")).hexdigest()
        hashes[digest].append(record["session_id"])
    groups = {h: ids for h, ids in hashes.items() if len(ids) > 1}
    return {"duplicate_groups": len(groups), "duplicate_sessions": sum(len(v) for v in groups.values()), "examples": groups}


def summarize_records(records: List[Dict[str, Any]], bad: List[Tuple[int, List[str]]]) -> Dict[str, Any]:
    issue_types = Counter(issue.split(":")[0] for _, issues in bad for issue in issues)
    leakage = sum(1 for _, issues in bad if any(i.startswith("leakage:") for i in issues))
    noise = sum(1 for _, issues in bad if any(i.startswith("noise_mismatch:") for i in issues))

    noise_metrics: Dict[str, Dict[str, float]] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record.get("noise_level", "unknown")].append(session_metrics(record))

    for level, rows in grouped.items():
        if not rows:
            continue
        noise_metrics[level] = {
            "avg_log_lines": round(sum(r["num_lines"] for r in rows) / len(rows), 2),
            "avg_evidence_ratio": round(sum(r["evidence_ratio"] for r in rows) / len(rows), 2),
            "avg_unrelated_services": round(sum(r["unrelated_services"] for r in rows) / len(rows), 2),
            "avg_distractors": round(sum(r["distractors"] for r in rows) / len(rows), 2),
            "avg_errors_outside_evidence": round(sum(r["errors_outside_evidence"] for r in rows) / len(rows), 2),
        }

    return {
        "num_records": len(records),
        "bad_records": len(bad),
        "issue_type_counts": dict(issue_types),
        "records_with_label_leakage": leakage,
        "records_with_noise_mismatch": noise,
        "label_coverage": label_coverage(records),
        "duplicates": duplicate_report(records),
        "noise_metrics_by_level": noise_metrics,
        "sample_bad_records": [{"line": line_no, "session_id": records[line_no - 1]["session_id"], "issues": issues} for line_no, issues in bad[:10]],
    }


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_file(path: Path) -> Dict[str, Any]:
    records = load_records(path)
    bad: List[Tuple[int, List[str]]] = []
    for line_no, record in enumerate(records, start=1):
        issues = validate_record(record)
        if issues:
            bad.append((line_no, issues))
    return summarize_records(records, bad)


def validate_all_splits() -> Dict[str, Any]:
    split_reports: Dict[str, Any] = {}
    all_records: List[Dict[str, Any]] = []
    all_bad: List[Tuple[int, List[str]]] = []

    for split in ["train", "valid", "test"]:
        path = DATA_DIR / f"logtriage_{split}.jsonl"
        records = load_records(path)
        bad = []
        for line_no, record in enumerate(records, start=1):
            issues = validate_record(record)
            if issues:
                bad.append((line_no, issues))
        split_reports[split] = summarize_records(records, bad)
        all_records.extend(records)
        offset = len(all_records) - len(records)
        all_bad.extend([(offset + line_no, issues) for line_no, issues in bad])

    report = {
        "splits": split_reports,
        "combined": summarize_records(all_records, all_bad),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "data_validation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LogTriage JSONL data")
    parser.add_argument("path", nargs="?", help="Path to a JSONL file. If omitted, validate all splits.")
    args = parser.parse_args()

    if args.path:
        report = validate_file(Path(args.path))
    else:
        report = validate_all_splits()

    print(json.dumps(report if args.path else report["combined"], indent=2))


if __name__ == "__main__":
    main()
