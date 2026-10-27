"""Build an auditable CLAW-Edit v2 from reviewed source exclusions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from audit_claw_edit_dataset import response_content
from process_claw_edit_outputs import chinese_ratio, clean_text, edit_distance, retained_numbers


MARKER = re.compile(r"<正>")
TEXT_FIELDS = ("abstract_clean", "abstract_raw")
IDENTITY_FIELDS = (
    "source_id", "group_id", "parent_record_id", "corpus_id", "operation", "provider",
    "model_name", "model_snapshot", "prompt_id", "group_split", "source_year",
    "evaluation_role", "source_title_clean", "title_sha256", "input_sha256",
    "output_sha256", "quality_status", "quality_reason", "raw_evidence_sha256",
)


def sha256(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return MARKER.sub("", text).strip() if "<正>" in text else text


def inspect_reviewed_csv(csv_bytes: bytes, processed: list[dict]) -> tuple[set[str], dict]:
    reviewed = list(csv.DictReader(io.StringIO(csv_bytes.decode("gbk"), newline="")))
    old = {row["record_id"]: row for row in processed}
    ids = [row["record_id"] for row in reviewed]
    if len(ids) != len(set(ids)) or len(old) != len(processed):
        raise ValueError("Duplicate record IDs in reviewed CSV or original JSONL")
    if set(ids) - set(old):
        raise ValueError("Reviewed CSV contains IDs outside the original processed JSONL")

    old_by_source: defaultdict[str, set[str]] = defaultdict(set)
    retained_by_source: defaultdict[str, set[str]] = defaultdict(set)
    for row in processed:
        old_by_source[row["source_id"]].add(row["record_id"])
    for row in reviewed:
        retained_by_source[row["source_id"]].add(row["record_id"])
        original = old[row["record_id"]]
        for field in IDENTITY_FIELDS:
            if str(row.get(field, "")) != str(original.get(field, "")):
                raise ValueError(f"Unexpected CSV edit: {row['record_id']} {field}")
        if row["source_title_raw"] not in (original["source_title_raw"], original["source_title_clean"]):
            raise ValueError(f"Unexpected raw title edit: {row['record_id']}")
        for field in TEXT_FIELDS:
            if row[field] not in (original[field], normalize(original[field])):
                raise ValueError(f"Unexpected text edit: {row['record_id']} {field}")
    partial = [sid for sid, old_ids in old_by_source.items()
               if retained_by_source.get(sid) and retained_by_source[sid] != old_ids]
    if partial:
        raise ValueError(f"Partial source groups in reviewed CSV: {partial[:5]}")
    excluded = set(old_by_source) - set(retained_by_source)
    if any(len(old_by_source[sid]) != 17 for sid in old_by_source):
        raise ValueError("Original processed data has a non-17-record source group")
    details = {
        "reviewed_csv_sha256": sha256(csv_bytes),
        "reviewed_csv_rows": len(reviewed),
        "excluded_source_count": len(excluded),
        "excluded_record_count": sum(len(old_by_source[sid]) for sid in excluded),
        "retained_pilot_count": sum(row["evaluation_role"] == "pilot_only" for row in reviewed),
        "csv_title_raw_replaced_by_clean_count": sum(
            row["source_title_raw"] != old[row["record_id"]]["source_title_raw"] for row in reviewed
        ),
        "csv_marker_cleaned_count": sum(
            row["abstract_clean"] != old[row["record_id"]]["abstract_clean"] for row in reviewed
        ),
    }
    return excluded, details


def build_flags(records: list[dict]) -> list[dict]:
    flags: list[dict] = []
    h_by_source = {row["source_id"]: row for row in records if row["corpus_id"] == "H"}
    grouped: defaultdict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for row in records:
        if row["corpus_id"] != "E":
            continue
        rid = row["record_id"]
        distance = float(row["normalized_edit_distance"])
        length_ratio = len(row["abstract_clean"]) / len(h_by_source[row["source_id"]]["abstract_clean"])
        grouped[(row["source_id"], row["provider"])][int(row["intervention_level"])] = row
        if distance == 0:
            flags.append({"record_id": rid, "flag": "unchanged_text", "value": distance})
        if distance > 1:
            flags.append({"record_id": rid, "flag": "large_edit_distance", "value": distance})
        if not 0.5 <= length_ratio <= 2:
            flags.append({"record_id": rid, "flag": "length_ratio_outside_0.5_2.0", "value": round(length_ratio, 4)})
        if float(row["numeric_retention"]) < 1:
            flags.append({"record_id": rid, "flag": "numeric_content_changed", "value": row["numeric_retention"]})
    for (sid, provider), levels in grouped.items():
        if set(levels) != {1, 2, 3, 4}:
            raise ValueError(f"Missing levels: {sid}, {provider}")
        changes = [float(levels[level]["normalized_edit_distance"]) for level in range(1, 5)]
        if any(left > right for left, right in zip(changes, changes[1:])):
            flags.append({"record_id": f"{provider}-{sid}", "flag": "edit_distance_nonmonotonic", "value": changes})
        hashes = [levels[level]["output_sha256"] for level in range(1, 5)]
        if len(set(hashes)) < 4:
            flags.append({"record_id": f"{provider}-{sid}", "flag": "duplicate_output_across_levels", "value": 4 - len(set(hashes))})
    return flags


def build(v1: Path, reviewed_csv: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing v2 directory: {out}")
    previous_audit = json.loads((v1 / "evaluation/provenance_audit.json").read_text(encoding="utf-8"))
    if previous_audit["error_count"]:
        raise ValueError("Original v1 audit has errors")
    processed = read_jsonl(v1 / "processed/records.jsonl")
    excluded, csv_inspection = inspect_reviewed_csv(reviewed_csv.read_bytes(), processed)
    old_benchmark = read_jsonl(v1 / "evaluation/benchmark_records.jsonl")
    old_by_id = {row["record_id"]: row for row in old_benchmark}
    selected = [row for row in old_benchmark if row["source_id"] not in excluded]
    h_original = {row["source_id"]: row for row in selected if row["corpus_id"] == "H"}
    expected_selected = csv_inspection["reviewed_csv_rows"] - csv_inspection["retained_pilot_count"]
    if len(selected) != expected_selected:
        raise ValueError(f"Selected count {len(selected)} differs from reviewed CSV benchmark count {expected_selected}")
    if len(selected) != len(h_original) * 17:
        raise ValueError("Selected benchmark groups are incomplete")

    queue_path = v1 / "source_registry/generation_queue.csv"
    with queue_path.open(encoding="utf-8-sig", newline="") as handle:
        queue = {row["record_id"]: row for row in csv.DictReader(handle)}
    records: list[dict] = []
    normalization_log: list[dict] = []
    raw_checked = 0
    for original in selected:
        row = dict(original)
        current_text = normalize(original["abstract_clean"])
        row["v1_input_sha256"] = original["input_sha256"]
        row["v1_output_sha256"] = original["output_sha256"]
        row["normalization_applied"] = current_text != original["abstract_clean"]
        row["abstract_clean"] = current_text
        row["output_sha256"] = sha256(current_text)
        row["char_count"] = len(current_text)
        row["chinese_ratio"] = chinese_ratio(current_text)
        if row["normalization_applied"]:
            normalization_log.append({
                "record_id": row["record_id"], "rule": "remove_all_<正>_markers_and_edge_whitespace",
                "v1_output_sha256": original["output_sha256"], "v2_output_sha256": row["output_sha256"],
            })
        source_text = normalize(h_original[row["source_id"]]["abstract_clean"])
        row["input_sha256"] = sha256(source_text)
        if row["corpus_id"] == "H":
            row["normalized_edit_distance"] = 0.0
            row["numeric_retention"] = 1.0
        else:
            evidence = queue.get(row["record_id"])
            if not evidence:
                raise ValueError(f"Missing v1 queue entry: {row['record_id']}")
            raw_path = (v1 / evidence["raw_path"]).resolve()
            if not raw_path.is_relative_to(v1.resolve()):
                raise ValueError(f"Raw evidence path escapes v1: {row['record_id']}")
            raw_bytes = raw_path.read_bytes()
            if sha256(raw_bytes) != original["raw_evidence_sha256"]:
                raise ValueError(f"Raw response changed: {row['record_id']}")
            payload = json.loads(raw_bytes)
            if (response_content(payload) != original["abstract_raw"] or
                    clean_text(original["abstract_raw"]) != original["abstract_clean"]):
                raise ValueError(f"Original raw response mismatch: {row['record_id']}")
            raw_checked += 1
            if (source_text == h_original[row["source_id"]]["abstract_clean"] and
                    current_text == original["abstract_clean"]):
                row["normalized_edit_distance"] = original["normalized_edit_distance"]
                row["numeric_retention"] = original["numeric_retention"]
            else:
                row["normalized_edit_distance"] = edit_distance(source_text, current_text) / max(len(source_text), 1)
                row["numeric_retention"] = retained_numbers(source_text, current_text)
        records.append(row)

    by_source: defaultdict[str, list[dict]] = defaultdict(list)
    by_group: defaultdict[str, set[str]] = defaultdict(set)
    for row in records:
        by_source[row["source_id"]].append(row)
        by_group[row["group_id"]].add(row["group_split"])
        if sha256(row["source_title_clean"]) != row["title_sha256"]:
            raise ValueError(f"Title hash mismatch: {row['record_id']}")
        if sha256(row["abstract_clean"]) != row["output_sha256"]:
            raise ValueError(f"Derived output hash mismatch: {row['record_id']}")
        if row["char_count"] != len(row["abstract_clean"]):
            raise ValueError(f"Derived length mismatch: {row['record_id']}")
        if row["input_sha256"] != sha256(normalize(h_original[row["source_id"]]["abstract_clean"])):
            raise ValueError(f"Derived input hash mismatch: {row['record_id']}")
        if row["record_id"] not in old_by_id or row["abstract_raw"] != old_by_id[row["record_id"]]["abstract_raw"]:
            raise ValueError(f"Raw text provenance mismatch: {row['record_id']}")
    if any(len(group) != 17 for group in by_source.values()) or any(len(splits) != 1 for splits in by_group.values()):
        raise ValueError("Incomplete source group or cross-split group")

    flags = build_flags(records)
    excluded_sources = [{
        "source_id": sid,
        "group_id": next(row["group_id"] for row in old_benchmark if row["source_id"] == sid),
        "group_split": next(row["group_split"] for row in old_benchmark if row["source_id"] == sid),
        "source_title_clean": next(row["source_title_clean"] for row in old_benchmark if row["source_id"] == sid),
        "decision": "excluded_after_manual_review",
    } for sid in sorted(excluded)]
    audit = {
        "error_count": 0,
        "quality_scope": "Source-group exclusions, derived hashes and raw-response lineage; legal meaning not certified.",
        "source_count": len(by_source),
        "benchmark_count": len(records),
        "benchmark_h_count": len(h_original),
        "benchmark_e_count": len(records) - len(h_original),
        "benchmark_by_split_and_type": dict(Counter(f"{r['group_split']}:{r['corpus_id']}" for r in records)),
        "excluded_source_count": len(excluded),
        "raw_responses_rechecked": raw_checked,
        "normalization_count": len(normalization_log),
        "normalization_by_type": dict(Counter(r["corpus_id"] for r in records if r["normalization_applied"])),
        "review_flags_by_code": dict(Counter(flag["flag"] for flag in flags)),
        "reviewed_csv_inspection": csv_inspection,
        "v1_processed_jsonl_sha256": sha256((v1 / "processed/records.jsonl").read_bytes()),
        "v1_benchmark_jsonl_sha256": sha256((v1 / "evaluation/benchmark_records.jsonl").read_bytes()),
    }
    out.mkdir(parents=True)
    evaluation = out / "evaluation"
    evaluation.mkdir()
    write_jsonl(evaluation / "benchmark_records.jsonl", records)
    write_jsonl(evaluation / "content_review_flags.jsonl", flags)
    write_jsonl(out / "normalization_log.jsonl", normalization_log)
    (out / "excluded_sources.json").write_text(json.dumps(excluded_sources, ensure_ascii=False, indent=2), encoding="utf-8")
    (evaluation / "provenance_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "dataset": "CLAW-Edit", "version": "2.0", "parent_version": "1.0",
        "source_of_exclusion_decisions": str(reviewed_csv),
        "source_of_record_text_and_evidence": str(v1 / "processed/records.jsonl"),
        "human_editing_conditions": False,
        "exclusion_unit": "source_id",
        "normalization": "All <正> markers are removed from derived abstract_clean; original abstract_raw is preserved.",
        "raw_fields_preserved": ["source_title_raw", "abstract_raw", "raw_evidence_sha256"],
        "counts": {"source_groups": len(by_source), "H": len(h_original), "E": len(records) - len(h_original)},
        "reviewed_csv_sha256": csv_inspection["reviewed_csv_sha256"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, default=Path("research_benchmark_legal_edit_v1"))
    parser.add_argument("--reviewed-csv", type=Path,
                        default=Path("research_benchmark_legal_edit_v1/processed/records.csv"))
    parser.add_argument("--out", type=Path, default=Path("research_benchmark_legal_edit_v2"))
    args = parser.parse_args()
    result = build(args.v1, args.reviewed_csv, args.out)
    print(json.dumps({key: value for key, value in result.items() if key != "reviewed_csv_inspection"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
