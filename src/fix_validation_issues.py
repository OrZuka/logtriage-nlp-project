"""Fix known validation failures in LogTriage JSONL datasets."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

REASON_REPLACEMENTS = [
    "reason=missing_runtime_config",
    "reason=invalid_service_configuration",
    "reason=checkout_config_unavailable",
    "reason=upstream_config_failure",
]

LEAK_PATTERN = re.compile(r"reason=configuration_error", re.IGNORECASE)


def fix_leakage_in_line(line: str, variant_idx: int) -> str:
    replacement = REASON_REPLACEMENTS[variant_idx % len(REASON_REPLACEMENTS)]
    return LEAK_PATTERN.sub(replacement, line)


def fix_record(record: Dict[str, Any], variant_idx: int) -> Dict[str, Any]:
    record = dict(record)
    record["logs"] = [fix_leakage_in_line(line, variant_idx) for line in record["logs"]]

    if record["session_id"] == "sess_04681":
        record["logs"] = [
            '{"ts":"2026-05-01 10:16:01","service":"checkout-service","level":"INFO","msg":"loading runtime configuration version=8241"}',
            '{"ts":"2026-05-01 10:16:02","service":"checkout-service","level":"ERROR","msg":"missing required config PAYMENT_PROVIDER_URL"}',
            '{"ts":"2026-05-01 10:16:03","service":"payment-service","level":"ERROR","msg":"cannot initialize payment client because endpoint is empty"}',
            '{"ts":"2026-05-01 10:16:04","service":"gateway-service","level":"ERROR","msg":"status=500 route=/checkout reason=missing_runtime_config"}',
        ]
        record["log_format"] = "json_like"

    if record["session_id"] in {"sess_00599", "sess_02270"}:
        record["noise_level"] = "high"

    return record


def process_file(path: Path) -> int:
    rows: List[Dict[str, Any]] = []
    changed = 0
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            fixed = fix_record(record, idx)
            if fixed != record:
                changed += 1
            rows.append(fixed)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return changed


def rebuild_all_jsonl() -> None:
    rows: List[Dict[str, Any]] = []
    for split in ["train", "valid", "test"]:
        with (DATA_DIR / f"logtriage_{split}.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    with (DATA_DIR / "logtriage_all.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_metadata() -> None:
    meta_path = DATA_DIR / "metadata.csv"
    records: Dict[str, Dict[str, Any]] = {}
    for split in ["train", "valid", "test"]:
        with (DATA_DIR / f"logtriage_{split}.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                records[record["session_id"]] = record

    with meta_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    for row in rows:
        session_id = row["session_id"]
        if session_id not in records:
            continue
        record = records[session_id]
        row["noise_level"] = record.get("noise_level", row.get("noise_level", ""))
        row["log_format"] = record.get("log_format", row.get("log_format", ""))
        row["num_lines"] = str(len(record.get("logs", [])))

    with meta_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    total = 0
    for split in ["train", "valid", "test"]:
        path = DATA_DIR / f"logtriage_{split}.jsonl"
        changed = process_file(path)
        print(f"{path.name}: updated {changed} records")
        total += changed

    rebuild_all_jsonl()
    update_metadata()
    print(f"Done. Updated {total} records across splits.")


if __name__ == "__main__":
    main()
