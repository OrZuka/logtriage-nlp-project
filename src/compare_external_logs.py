"""Compare synthetic LogTriage logs against external reference log corpora.

Uses a bundled LogHub HDFS sample when available, otherwise downloads the
public 2k-line sample from the LogPAI LogHub repository.
"""
from __future__ import annotations

import json
import re
import statistics
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
REF_DIR = DATA_DIR / "external_refs"
LOGHUB_SAMPLE = REF_DIR / "loghub_hdfs_2k.log"
LOGHUB_URL = "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log"

LEVEL_RE = re.compile(r"\b(ERROR|WARN|WARNING|INFO|DEBUG|CRITICAL|FATAL)\b", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}|\d{6}\s+\d{6}")
COMPONENT_RE = re.compile(r"\b[a-zA-Z0-9_.$]+\.[a-zA-Z0-9_$]+\b")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{2,}")


def ensure_loghub_sample() -> Path:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    if LOGHUB_SAMPLE.exists() and LOGHUB_SAMPLE.stat().st_size > 1000:
        return LOGHUB_SAMPLE
    with urllib.request.urlopen(LOGHUB_URL, timeout=30) as resp:
        LOGHUB_SAMPLE.write_bytes(resp.read())
    return LOGHUB_SAMPLE


def load_synthetic_lines() -> List[str]:
    lines: List[str] = []
    for split in ["train", "valid", "test"]:
        path = DATA_DIR / f"logtriage_{split}.jsonl"
        with path.open("r", encoding="utf-8") as f:
            for row in f:
                record = json.loads(row)
                lines.extend(record["logs"])
    return lines


def load_external_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def profile_lines(lines: Iterable[str]) -> Dict[str, Any]:
    rows = list(lines)
    if not rows:
        return {}

    lengths = [len(line) for line in rows]
    tokens = [len(TOKEN_RE.findall(line)) for line in rows]
    levels = Counter(LEVEL_RE.search(line).group(1).upper() for line in rows if LEVEL_RE.search(line))
    timestamps = sum(1 for line in rows if TIMESTAMP_RE.search(line))
    components = sum(1 for line in rows if COMPONENT_RE.search(line))
    json_like = sum(1 for line in rows if line.lstrip().startswith("{"))
    key_value = sum(1 for line in rows if "=" in line or '":"' in line)

    return {
        "num_lines": len(rows),
        "avg_line_length": round(statistics.mean(lengths), 2),
        "median_line_length": round(statistics.median(lengths), 2),
        "avg_token_count": round(statistics.mean(tokens), 2),
        "timestamp_rate": round(timestamps / len(rows), 3),
        "component_like_rate": round(components / len(rows), 3),
        "json_like_rate": round(json_like / len(rows), 3),
        "key_value_rate": round(key_value / len(rows), 3),
        "level_distribution": dict(levels),
    }


def compare_profiles(synthetic: Dict[str, Any], external: Dict[str, Any]) -> Dict[str, Any]:
    shared_metrics = [
        "avg_line_length",
        "avg_token_count",
        "timestamp_rate",
        "component_like_rate",
        "key_value_rate",
    ]
    comparison: Dict[str, Any] = {}
    for metric in shared_metrics:
        syn = synthetic.get(metric, 0.0)
        ext = external.get(metric, 0.0)
        diff = round(abs(syn - ext), 3)
        comparison[metric] = {
            "synthetic": syn,
            "external_loghub_hdfs": ext,
            "abs_diff": diff,
            "within_30pct_of_external": ext == 0 or diff <= 0.3 * ext,
        }

    syn_levels = synthetic.get("level_distribution", {})
    ext_levels = external.get("level_distribution", {})
    syn_total = sum(syn_levels.values()) or 1
    ext_total = sum(ext_levels.values()) or 1
    comparison["level_distribution_normalized"] = {
        level: {
            "synthetic": round(syn_levels.get(level, 0) / syn_total, 3),
            "external_loghub_hdfs": round(ext_levels.get(level, 0) / ext_total, 3),
        }
        for level in sorted(set(syn_levels) | set(ext_levels))
    }

    notes = [
        "LogHub HDFS is a different domain (distributed filesystem) but provides a public reference for timestamped, leveled log structure.",
        "Synthetic LogTriage logs intentionally include JSON-like and key-value microservice formats not present in HDFS.",
        "Similarity is assessed on structural features (timestamps, severity levels, token length), not label semantics.",
    ]
    return {"metric_comparison": comparison, "notes": notes}


def realism_score(comparison: Dict[str, Any]) -> Dict[str, Any]:
    metrics = comparison["metric_comparison"]
    checks = [
        metrics["timestamp_rate"]["within_30pct_of_external"] or metrics["timestamp_rate"]["synthetic"] >= 0.8,
        metrics["avg_token_count"]["synthetic"] >= 8,
        metrics["key_value_rate"]["synthetic"] >= 0.5,
        "ERROR" in comparison["metric_comparison"]["level_distribution_normalized"],
    ]
    passed = sum(1 for ok in checks if ok)
    return {
        "checks_passed": passed,
        "checks_total": len(checks),
        "overall": "acceptable_structural_similarity" if passed >= 3 else "review_recommended",
    }


def run_comparison() -> Dict[str, Any]:
    sample_path = ensure_loghub_sample()
    synthetic_lines = load_synthetic_lines()
    external_lines = load_external_lines(sample_path)

    synthetic_profile = profile_lines(synthetic_lines)
    external_profile = profile_lines(external_lines)
    comparison = compare_profiles(synthetic_profile, external_profile)
    score = realism_score(comparison)

    report = {
        "reference_dataset": "LogHub HDFS 2k sample",
        "reference_path": str(sample_path.relative_to(ROOT)),
        "synthetic_profile": synthetic_profile,
        "external_profile": external_profile,
        "comparison": comparison,
        "realism_assessment": score,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "external_log_comparison.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = run_comparison()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
