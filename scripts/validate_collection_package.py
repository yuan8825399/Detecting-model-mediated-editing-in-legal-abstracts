"""Validate the manual web-collection queue without changing any records."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research_benchmark_v1"))
    args = parser.parse_args()
    reg = args.root / "source_registry"
    history = pd.read_csv(reg / "history_selected.csv", dtype=str, keep_default_na=False)
    unknown = pd.read_csv(reg / "unknown_2021_2025.csv", dtype=str, keep_default_na=False)
    queue_paths = [reg / "generation_queue_p0.csv", reg / "generation_queue_challenge_p1_p2.csv"]
    queues = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in queue_paths if path.exists()]
    queue = pd.concat(queues, ignore_index=True) if queues else pd.DataFrame()
    errors = []
    if not history.source_id.is_unique or not unknown.source_id.is_unique:
        errors.append("duplicate source_id in registry")
    if not queue.record_id.is_unique:
        errors.append("duplicate record_id in queue")
    if set(history.group_id) & set(unknown.group_id):
        errors.append("history and unknown groups overlap")
    split_sets = [set(history.loc[history.group_split == split, "group_id"]) for split in ["train", "dev", "calibration", "test"]]
    for i, left in enumerate(split_sets):
        for right in split_sets[i + 1:]:
            if left & right:
                errors.append("source group crosses fixed splits")
    if unknown.label.fillna("").ne("").any():
        errors.append("unknown corpus has non-empty labels")
    accepted = queue[queue.status == "accepted"].copy() if not queue.empty else queue
    evidence_missing = 0
    hash_mismatch = 0
    for row in accepted.to_dict("records"):
        path_value = row.get("raw_evidence_path", "")
        hash_value = row.get("evidence_sha256", "")
        path = args.root / path_value if path_value else None
        if path is None or not path.exists():
            evidence_missing += 1
        elif not hash_value or sha256(path) != hash_value:
            hash_mismatch += 1
    if evidence_missing:
        errors.append(f"accepted records missing evidence: {evidence_missing}")
    if hash_mismatch:
        errors.append(f"accepted records with hash mismatch: {hash_mismatch}")
    status_counts = queue.status.value_counts(dropna=False).to_dict() if not queue.empty else {}
    report = {"history_rows": len(history), "unknown_rows": len(unknown), "queue_rows": len(queue),
              "status_counts": status_counts, "accepted_evidence_missing": evidence_missing,
              "accepted_hash_mismatch": hash_mismatch, "errors": errors}
    (args.root / "evaluation" / "collection_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
