"""Audit CLAW-Edit provenance and export the fixed, pilot-free benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from process_claw_edit_outputs import clean_text


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def response_content(payload: dict) -> str:
    value = payload.get("generated_abstract", payload.get("abstract"))
    if value is None:
        response = payload.get("response", {})
        choices = response.get("choices", []) if isinstance(response, dict) else []
        value = choices[0].get("message", {}).get("content", "") if choices else ""
    return str(value or "")


def add_issue(issues: list[dict], severity: str, code: str, record_id: str, detail: str = "") -> None:
    issues.append({"severity": severity, "code": code, "record_id": record_id, "detail": detail})


def audit(root: Path) -> dict:
    sources = read_csv(root / "source_registry/sources.csv")
    queue = read_csv(root / "source_registry/generation_queue.csv")
    processed = read_jsonl(root / "processed/records.jsonl")
    source_by_id = {row["source_id"]: row for row in sources}
    processed_by_id = {row["record_id"]: row for row in processed}
    issues: list[dict] = []
    warnings: list[dict] = []
    groups_by_split: defaultdict[str, set[str]] = defaultdict(set)

    for source in sources:
        sid = source["source_id"]
        groups_by_split[source["group_id"]].add(source["group_split"])
        if digest(source["source_title_clean"]) != source["title_sha256"]:
            add_issue(issues, "error", "source_title_hash", sid)
        if digest(source["abstract_clean"]) != source["abstract_sha256"]:
            add_issue(issues, "error", "source_abstract_hash", sid)
        h = processed_by_id.get(f"{sid}-H")
        if not h:
            add_issue(issues, "error", "missing_h_record", sid)
        elif (h.get("abstract_clean") != source["abstract_clean"] or
              h.get("source_title_clean") != source["source_title_clean"] or
              h.get("group_split") != source["group_split"]):
            add_issue(issues, "error", "h_source_mismatch", sid)
    for group, splits in groups_by_split.items():
        if len(splits) != 1:
            add_issue(issues, "error", "group_crosses_splits", group, ",".join(sorted(splits)))

    if len(source_by_id) != len(sources):
        add_issue(issues, "error", "duplicate_source_id", "dataset")
    if len(processed_by_id) != len(processed):
        add_issue(issues, "error", "duplicate_processed_id", "dataset")
    if len({row["record_id"] for row in queue}) != len(queue):
        add_issue(issues, "error", "duplicate_queue_id", "dataset")

    for task in queue:
        rid = task["record_id"]
        source = source_by_id.get(task["source_id"])
        record = processed_by_id.get(rid)
        if source is None:
            add_issue(issues, "error", "missing_source", rid)
            continue
        if record is None:
            add_issue(issues, "error", "missing_processed_e", rid)
            continue
        if task["status"] != "accepted":
            add_issue(issues, "error", "queue_not_accepted", rid, task["status"])
        expected_fields = {
            "group_id": source["group_id"],
            "group_split": source["group_split"],
            "source_title_clean": source["source_title_clean"],
            "title_sha256": source["title_sha256"],
            "input_sha256": source["abstract_sha256"],
            "parent_record_id": f"{source['source_id']}-H",
        }
        for field, expected in expected_fields.items():
            if task.get(field) != expected or record.get(field) != expected:
                add_issue(issues, "error", f"{field}_mismatch", rid)
        for field in ("provider", "prompt_id", "operation"):
            if record.get(field) != task.get(field):
                add_issue(issues, "error", f"{field}_mismatch", rid)
        if str(record.get("intervention_level")) != task["intervention_level"]:
            add_issue(issues, "error", "level_mismatch", rid)
        if record.get("corpus_id") != "E" or record.get("quality_status") != "accepted":
            add_issue(issues, "error", "processed_not_accepted_e", rid)

        prompt_path = root / "protocol/edit_prompts" / f"{task['prompt_id']}.txt"
        if not prompt_path.exists() or digest(prompt_path.read_text(encoding="utf-8")) != task["prompt_sha256"]:
            add_issue(issues, "error", "prompt_template_hash", rid)
        raw_relative = task.get("raw_path", "")
        if not raw_relative:
            add_issue(issues, "error", "missing_raw_path", rid)
            continue
        raw_path = (root / raw_relative).resolve()
        if not raw_path.is_relative_to(root.resolve()) or not raw_path.is_file():
            add_issue(issues, "error", "invalid_raw_path", rid, raw_relative)
            continue
        raw_bytes = raw_path.read_bytes()
        if digest(raw_bytes) != record.get("raw_evidence_sha256"):
            add_issue(issues, "error", "raw_evidence_hash", rid)
        try:
            payload = json.loads(raw_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            add_issue(issues, "error", "invalid_raw_json", rid)
            continue
        for field in ("record_id", "provider", "prompt_sha256", "input_sha256"):
            if payload.get(field) != task.get(field):
                add_issue(issues, "error", f"raw_{field}_mismatch", rid)
        response = payload.get("response", {})
        if isinstance(response, dict) and response.get("model") and response["model"] != task.get("model_snapshot"):
            add_issue(issues, "error", "returned_model_mismatch", rid)
        content = response_content(payload)
        if digest(content.strip()) != task.get("output_sha256"):
            add_issue(issues, "error", "queue_output_hash", rid)
        if content != record.get("abstract_raw") or clean_text(content) != record.get("abstract_clean"):
            add_issue(issues, "error", "processed_text_mismatch", rid)
        if digest(record.get("abstract_clean", "")) != record.get("output_sha256"):
            add_issue(issues, "error", "processed_output_hash", rid)
        request = payload.get("request", {})
        messages = request.get("messages", []) if isinstance(request, dict) else []
        user_text = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
        if source["abstract_clean"] not in user_text:
            add_issue(issues, "error", "source_not_in_prompt", rid)

    benchmark = [row for row in processed if row.get("evaluation_role") == "benchmark"]
    pilot = [row for row in processed if row.get("evaluation_role") == "pilot_only"]
    if len(benchmark) + len(pilot) != len(processed):
        add_issue(issues, "error", "unknown_evaluation_role", "dataset")
    for row in benchmark:
        if row.get("quality_status") != "accepted":
            add_issue(issues, "error", "benchmark_contains_rejected_record", row["record_id"])

    # These are review flags, never automatic exclusion rules.
    flags: list[dict] = []
    edits_by_source_provider: defaultdict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for row in benchmark:
        if row["corpus_id"] != "E":
            continue
        rid = row["record_id"]
        change = float(row["normalized_edit_distance"])
        length_ratio = len(row["abstract_clean"]) / max(len(source_by_id[row["source_id"]]["abstract_clean"]), 1)
        edits_by_source_provider[(row["source_id"], row["provider"])][int(row["intervention_level"])] = row
        if change == 0:
            flags.append({"record_id": rid, "flag": "unchanged_text", "value": change})
        if change > 1.0:
            flags.append({"record_id": rid, "flag": "large_edit_distance", "value": change})
        if length_ratio < 0.5 or length_ratio > 2.0:
            flags.append({"record_id": rid, "flag": "length_ratio_outside_0.5_2.0", "value": round(length_ratio, 4)})
        if float(row["numeric_retention"]) < 1.0:
            flags.append({"record_id": rid, "flag": "numeric_content_changed", "value": row["numeric_retention"]})
    for (sid, provider), levels in edits_by_source_provider.items():
        if len(levels) != 4:
            add_issue(issues, "error", "incomplete_source_provider_levels", f"{provider}-{sid}")
            continue
        changes = [float(levels[level]["normalized_edit_distance"]) for level in (1, 2, 3, 4)]
        if any(left > right for left, right in zip(changes, changes[1:])):
            flags.append({"record_id": f"{provider}-{sid}", "flag": "edit_distance_nonmonotonic", "value": changes})
        hashes = [levels[level]["output_sha256"] for level in (1, 2, 3, 4)]
        if len(set(hashes)) < len(hashes):
            flags.append({"record_id": f"{provider}-{sid}", "flag": "duplicate_output_across_levels", "value": 4 - len(set(hashes))})

    text_to_groups: defaultdict[str, set[str]] = defaultdict(set)
    for row in benchmark:
        text_to_groups[row["output_sha256"]].add(row["group_id"])
    cross_source_duplicates = sum(len(groups) > 1 for groups in text_to_groups.values())
    if cross_source_duplicates:
        add_issue(warnings, "warning", "exact_text_shared_by_multiple_sources", "dataset", str(cross_source_duplicates))

    edit_metrics = {}
    for provider in sorted({row["provider"] for row in benchmark if row["corpus_id"] == "E"}):
        for level in range(1, 5):
            subset = [row for row in benchmark if row["corpus_id"] == "E" and
                      row["provider"] == provider and int(row["intervention_level"]) == level]
            distances = [float(row["normalized_edit_distance"]) for row in subset]
            edit_metrics[f"{provider}:E{level}"] = {
                "n": len(subset),
                "mean_edit_distance": round(mean(distances), 4) if distances else None,
                "median_edit_distance": round(median(distances), 4) if distances else None,
                "unchanged": sum(distance == 0 for distance in distances),
                "numeric_retention_below_1": sum(float(row["numeric_retention"]) < 1.0 for row in subset),
            }

    errors_by_code = Counter(item["code"] for item in issues)
    flags_by_code = Counter(item["flag"] for item in flags)
    report = {
        "source_count": len(sources),
        "queue_count": len(queue),
        "processed_count": len(processed),
        "benchmark_count": len(benchmark),
        "pilot_count": len(pilot),
        "benchmark_by_split_and_type": dict(Counter(f"{r['group_split']}:{r['corpus_id']}" for r in benchmark)),
        "benchmark_edit_by_provider_level": dict(Counter(f"{r['provider']}:E{r['intervention_level']}" for r in benchmark if r["corpus_id"] == "E")),
        "error_count": len(issues),
        "errors_by_code": dict(errors_by_code),
        "warning_count": len(warnings),
        "warnings_by_code": dict(Counter(item["code"] for item in warnings)),
        "queue_quality_status": dict(Counter(task.get("quality_status", "") for task in queue)),
        "review_flag_count": len(flags),
        "review_flags_by_code": dict(flags_by_code),
        "edit_metrics_by_provider_level": edit_metrics,
        "source_provider_groups": len(edits_by_source_provider),
        "cross_source_exact_duplicate_texts": cross_source_duplicates,
        "quality_scope": "Automated provenance and format audit; legal meaning preservation not verified.",
    }
    evaluation = root / "evaluation"
    write_jsonl(evaluation / "audit_errors.jsonl", issues)
    write_jsonl(evaluation / "audit_warnings.jsonl", warnings)
    write_jsonl(evaluation / "content_review_flags.jsonl", flags)
    (evaluation / "provenance_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not issues:
        write_jsonl(evaluation / "benchmark_records.jsonl", benchmark)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research_benchmark_legal_edit_v1"))
    args = parser.parse_args()
    report = audit(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report["error_count"] else 0)


if __name__ == "__main__":
    main()
