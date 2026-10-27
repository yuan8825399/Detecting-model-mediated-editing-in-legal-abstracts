"""Validate CLAW-Edit integrity, provenance, hashes, and group isolation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research_benchmark_legal_edit_v1"))
    args = parser.parse_args()
    reg = args.root / "source_registry"
    sources = read_csv(reg / "sources.csv")
    queue = read_csv(reg / "generation_queue.csv")
    errors: list[str] = []
    source_ids = [x["source_id"] for x in sources]
    if len(source_ids) != len(set(source_ids)):
        errors.append("duplicate source_id")
    record_ids = [x["record_id"] for x in queue]
    if len(record_ids) != len(set(record_ids)):
        errors.append("duplicate queue record_id")
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in sources:
        title = row.get("source_title_clean", "")
        title_hash = hashlib.sha256(title.encode("utf-8")).hexdigest()
        if not title or row.get("title_sha256") != title_hash:
            errors.append(f"invalid title or title hash: {row['source_id']}")
        group_splits[row["group_id"]].add(row["group_split"])
    if any(len(x) != 1 for x in group_splits.values()):
        errors.append("group crosses fixed splits")
    source_map = {x["source_id"]: x for x in sources}
    for row in sources:
        title = row.get("source_title_clean", "")
        expected = hashlib.sha256(title.encode("utf-8")).hexdigest()
        if not title or row.get("title_sha256") != expected:
            errors.append(f"invalid title or title hash: {row['source_id']}")
    for row in queue:
        if row["source_id"] not in source_map:
            errors.append(f"queue source missing: {row['record_id']}")
        if row["parent_record_id"] != f"{row['source_id']}-H":
            errors.append(f"invalid parent_record_id: {row['record_id']}")
        source = source_map.get(row["source_id"])
        if source and (row.get("source_title_clean") != source.get("source_title_clean") or
                       row.get("title_sha256") != source.get("title_sha256")):
            errors.append(f"queue title mismatch: {row['record_id']}")
        if row["intervention_level"] not in {"1", "2", "3", "4"}:
            errors.append(f"invalid intervention level: {row['record_id']}")
    processed_path = args.root / "processed/records.jsonl"
    processed = []
    if processed_path.exists():
        with processed_path.open("r", encoding="utf-8") as handle:
            processed = [json.loads(line) for line in handle if line.strip()]
    processed_ids = [x["record_id"] for x in processed]
    if len(processed_ids) != len(set(processed_ids)):
        errors.append("duplicate processed record_id")
    h_ids = {x["record_id"] for x in processed if x.get("corpus_id") == "H"}
    for row in processed:
        if row.get("corpus_id") == "E" and row.get("parent_record_id") not in h_ids:
            errors.append(f"edited record has no H parent: {row['record_id']}")
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    counts = Counter((x.get("provider", ""), x.get("intervention_level", "")) for x in processed if x.get("corpus_id") == "E")
    report = {
        "source_rows": len(sources), "queue_rows": len(queue), "processed_rows": len(processed),
        "source_split_counts": Counter(x["group_split"] for x in sources),
        "queue_provider_counts": Counter(x["provider"] for x in queue),
        "processed_edit_counts": {f"{a}:{b}": c for (a, b), c in counts.items()},
        "human_editing_conditions": manifest.get("human_editing_conditions"),
        "errors": errors,
    }
    out = args.root / "evaluation/validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
