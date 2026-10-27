"""Build the final source-screened CLAW-Edit benchmark from audited v2 records."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_claw_edit_dataset import response_content
from build_claw_edit_v2 import build_flags, read_jsonl, sha256, write_jsonl
from export_claw_edit_analysis import missing_terminal_punctuation
from review_claw_edit_content import REFUSAL_RE


TRUNCATION_EXCLUSIONS = frozenset({
    "china-law-2004-00043", "china-law-2004-00047", "china-law-2004-00094",
    "china-law-2004-00115", "china-law-2005-00093", "china-law-2005-00094",
    "china-law-2008-00017",
})
PROVIDERS = frozenset({"deepseek", "doubao", "kimi", "wenxin"})


def inspect_selection(csv_path: Path, v2_records: list[dict]) -> tuple[set[str], dict]:
    csv_bytes = csv_path.read_bytes()
    reviewed = list(csv.DictReader(io.StringIO(csv_bytes.decode("gbk"), newline="")))
    ids = [row["record_id"] for row in reviewed]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate record_id in reviewed CSV")
    csv_by_source: defaultdict[str, set[str]] = defaultdict(set)
    for row in reviewed:
        csv_by_source[row["source_id"]].add(row["record_id"])
    if any(len(group) != 17 for group in csv_by_source.values()):
        raise ValueError("Reviewed CSV contains an incomplete source group")
    v2_by_source: defaultdict[str, set[str]] = defaultdict(set)
    for row in v2_records:
        v2_by_source[row["source_id"]].add(row["record_id"])
    by_id = {row["record_id"]: row for row in v2_records}
    formal = {row["source_id"] for row in reviewed if row["evaluation_role"] != "pilot_only"}
    pilot = {row["source_id"] for row in reviewed if row["evaluation_role"] == "pilot_only"}
    if formal & pilot or len(pilot) != 1 or formal - set(v2_by_source):
        raise ValueError("Unexpected formal or pilot source groups in reviewed CSV")
    for source_id in formal:
        if csv_by_source[source_id] != v2_by_source[source_id]:
            raise ValueError(f"Record identity changed in reviewed CSV: {source_id}")
    if not TRUNCATION_EXCLUSIONS <= formal:
        raise ValueError("One of the seven authorized truncation exclusions is not present")
    h = {row["source_id"]: row for row in reviewed if row["corpus_id"] == "H"}
    if any(not missing_terminal_punctuation(h[source_id]["abstract_clean"])
           for source_id in TRUNCATION_EXCLUSIONS):
        raise ValueError("An authorized truncation exclusion no longer has the reviewed ending")
    retained = formal - TRUNCATION_EXCLUSIONS
    details = {
        "reviewed_csv_sha256": sha256(csv_bytes),
        "csv_formal_source_count": len(formal),
        "csv_pilot_source_count": len(pilot),
        "csv_removed_v2_sources": sorted(set(v2_by_source) - formal),
        "manual_truncation_exclusions": sorted(TRUNCATION_EXCLUSIONS),
        "csv_text_overrides_ignored": sum(
            row["abstract_raw"] != by_id[row["record_id"]]["abstract_raw"] or
            row["abstract_clean"] != by_id[row["record_id"]]["abstract_clean"]
            for row in reviewed if row["record_id"] in by_id
        ),
    }
    return retained, details


def build(v1: Path, v2: Path, reviewed_csv: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    parent_audit = json.loads((v2 / "evaluation/provenance_audit.json").read_text(encoding="utf-8"))
    if parent_audit["error_count"]:
        raise ValueError("Parent provenance audit has errors")
    v2_path = v2 / "evaluation/benchmark_records.jsonl"
    v2_records = read_jsonl(v2_path)
    retained_sources, selection = inspect_selection(reviewed_csv, v2_records)
    records = [row for row in v2_records if row["source_id"] in retained_sources]
    original = {row["record_id"]: row for row in read_jsonl(v1 / "processed/records.jsonl")}
    queue_path = v1 / "source_registry/generation_queue.csv"
    with queue_path.open(encoding="utf-8-sig", newline="") as handle:
        queue = {row["record_id"]: row for row in csv.DictReader(handle)}
    h_by_source = {row["source_id"]: row for row in records if row["corpus_id"] == "H"}
    groups: defaultdict[str, list[dict]] = defaultdict(list)
    split_by_group: defaultdict[str, set[str]] = defaultdict(set)
    raw_checked = 0
    for row in records:
        rid = row["record_id"]
        source_id = row["source_id"]
        groups[source_id].append(row)
        split_by_group[row["group_id"]].add(row["group_split"])
        if (rid not in original or row["abstract_raw"] != original[rid]["abstract_raw"] or
                row["source_title_raw"] != original[rid]["source_title_raw"] or
                row["raw_evidence_sha256"] != original[rid]["raw_evidence_sha256"]):
            raise ValueError(f"Original evidence lineage changed: {rid}")
        if ("<正>" in row["abstract_clean"] or
                sha256(row["abstract_clean"]) != row["output_sha256"] or
                sha256(row["source_title_clean"]) != row["title_sha256"] or
                sha256(h_by_source[source_id]["abstract_clean"]) != row["input_sha256"] or
                len(row["abstract_clean"]) != row["char_count"]):
            raise ValueError(f"Derived text or hash mismatch: {rid}")
        if REFUSAL_RE.search(row["abstract_clean"]):
            raise ValueError(f"Refusal output remains: {rid}")
        if row["corpus_id"] == "E":
            evidence = queue.get(rid)
            if not evidence:
                raise ValueError(f"Missing raw response queue entry: {rid}")
            raw_path = (v1 / evidence["raw_path"]).resolve()
            if not raw_path.is_relative_to(v1.resolve()):
                raise ValueError(f"Raw response path escapes v1: {rid}")
            raw_bytes = raw_path.read_bytes()
            if (sha256(raw_bytes) != row["raw_evidence_sha256"] or
                    response_content(json.loads(raw_bytes)) != row["abstract_raw"]):
                raise ValueError(f"Raw response evidence mismatch: {rid}")
            raw_checked += 1
    if len(h_by_source) != len(retained_sources) or any(len(group) != 17 for group in groups.values()):
        raise ValueError("Retained source groups are incomplete")
    if any(len(splits) != 1 for splits in split_by_group.values()):
        raise ValueError("A source group crosses dataset splits")
    for source_id, group in groups.items():
        levels = {(row["provider"], int(row["intervention_level"])) for row in group if row["corpus_id"] == "E"}
        if (sum(row["corpus_id"] == "H" for row in group) != 1 or
                levels != {(provider, level) for provider in PROVIDERS for level in range(1, 5)}):
            raise ValueError(f"Missing H or provider/level record: {source_id}")

    flags = build_flags(records)
    analysis_records = [{key: value for key, value in row.items() if key != "abstract_raw"}
                        for row in records]
    if any("<正>" in json.dumps(row, ensure_ascii=False) for row in analysis_records):
        raise ValueError("An analysis record still contains a <正> marker")
    v2_log = read_jsonl(v2 / "normalization_log.jsonl")
    record_ids = {row["record_id"] for row in records}
    normalization_log = [row for row in v2_log if row["record_id"] in record_ids]
    accepted_no_punctuation = sorted(source_id for source_id, row in h_by_source.items()
                                     if missing_terminal_punctuation(row["abstract_clean"]))
    if len(accepted_no_punctuation) != 33:
        raise ValueError(f"Expected 33 accepted punctuation-only cases, got {len(accepted_no_punctuation)}")
    audit = {
        "error_count": 0,
        "source_count": len(groups),
        "benchmark_count": len(records),
        "benchmark_h_count": len(h_by_source),
        "benchmark_e_count": raw_checked,
        "benchmark_by_split_and_type": dict(sorted(Counter(
            f"{row['group_split']}:{row['corpus_id']}" for row in records).items())),
        "raw_responses_rechecked": raw_checked,
        "refusal_output_count": 0,
        "analysis_marker_count": 0,
        "accepted_h_without_terminal_punctuation_count": len(accepted_no_punctuation),
        "accepted_h_without_terminal_punctuation_ids": accepted_no_punctuation,
        "review_flags_by_code": dict(sorted(Counter(item["flag"] for item in flags).items())),
        "selection": selection,
        "parent_v2_benchmark_sha256": sha256(v2_path.read_bytes()),
        "v1_processed_jsonl_sha256": sha256((v1 / "processed/records.jsonl").read_bytes()),
        "quality_scope": "Source-group decisions, hashes and raw-response lineage; legal meaning not certified.",
    }
    prior_excluded = json.loads((v2 / "excluded_sources.json").read_text(encoding="utf-8"))
    newly_excluded = []
    v2_h = {row["source_id"]: row for row in v2_records if row["corpus_id"] == "H"}
    for source_id in sorted(set(v2_h) - retained_sources):
        parent = v2_h[source_id]
        newly_excluded.append({
            "source_id": source_id,
            "group_id": parent["group_id"],
            "group_split": parent["group_split"],
            "source_title_clean": parent["source_title_clean"],
            "decision": ("excluded_after_manual_truncation_review" if source_id in TRUNCATION_EXCLUSIONS
                         else "excluded_in_latest_reviewed_csv"),
        })
    out.mkdir(parents=True)
    evaluation = out / "evaluation"
    evaluation.mkdir()
    write_jsonl(evaluation / "benchmark_records.jsonl", records)
    write_jsonl(evaluation / "analysis_records.jsonl", analysis_records)
    write_jsonl(evaluation / "content_review_flags.jsonl", flags)
    write_jsonl(out / "normalization_log.jsonl", normalization_log)
    (out / "excluded_sources.json").write_text(
        json.dumps(prior_excluded + newly_excluded, ensure_ascii=False, indent=2), encoding="utf-8")
    (evaluation / "provenance_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "dataset": "CLAW-Edit", "version": "3.0", "parent_version": "2.0",
        "status": "formal_source_screened_benchmark",
        "reviewed_csv_sha256": selection["reviewed_csv_sha256"],
        "parent_v2_benchmark_sha256": audit["parent_v2_benchmark_sha256"],
        "source_of_exclusion_decisions": str(reviewed_csv),
        "source_of_record_text_and_evidence": str(v2_path),
        "manual_csv_text_overrides_used": False,
        "human_editing_conditions": False,
        "exclusion_unit": "source_id",
        "accepted_missing_terminal_punctuation_count": len(accepted_no_punctuation),
        "counts": {"source_groups": len(groups), "H": len(h_by_source), "E": raw_checked},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, default=Path("research_benchmark_legal_edit_v1"))
    parser.add_argument("--v2", type=Path, default=Path("research_benchmark_legal_edit_v2"))
    parser.add_argument("--csv", type=Path, default=Path("research_benchmark_legal_edit_v1/processed/records.csv"))
    parser.add_argument("--out", type=Path, default=Path("research_benchmark_legal_edit_v3"))
    args = parser.parse_args()
    audit = build(args.v1, args.v2, args.csv, args.out)
    print(json.dumps({key: audit[key] for key in (
        "source_count", "benchmark_count", "benchmark_h_count", "benchmark_e_count",
        "raw_responses_rechecked", "accepted_h_without_terminal_punctuation_count")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
