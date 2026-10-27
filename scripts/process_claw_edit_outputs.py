"""Import saved model responses into CLAW-Edit and compute paired metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


ZH_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
SPECIAL_RE = re.compile(r"AbstractFilter\([^)]*\);?|更多\s*还原|附视频", re.I)
PROMPT_LEAK_RE = re.compile(r"(以下是修改后的摘要|只输出|原摘要：|作为AI|作为人工智能)", re.I)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?|第\s*[一二三四五六七八九十百千万0-9]+\s*[条章节款项]?")


def clean_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = SPECIAL_RE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chinese_ratio(text: str) -> float:
    zh = len(ZH_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    return zh / (zh + latin) if zh + latin else 0.0


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def retained_numbers(source: str, output: str) -> float:
    expected = NUMBER_RE.findall(source)
    if not expected:
        return 1.0
    remaining = list(NUMBER_RE.findall(output))
    return sum(remaining.count(item) >= expected.count(item) for item in set(expected)) / len(set(expected))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_flat_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research_benchmark_legal_edit_v1"))
    args = parser.parse_args()
    registry = args.root / "source_registry"
    sources = {r["source_id"]: r for r in load_csv(registry / "sources.csv")}
    queue = load_csv(registry / "generation_queue.csv")
    pilot_source_ids = set()
    pilot_path = args.root / "evaluation/wenxin_test_run.json"
    if pilot_path.exists():
        pilot_run = json.loads(pilot_path.read_text(encoding="utf-8"))
        if pilot_run.get("evaluation_role") == "pilot_only":
            pilot_source_ids.add(pilot_run.get("source_id", ""))
    records: list[dict[str, object]] = []
    for source in sources.values():
        records.append({
            "record_id": f"{source['source_id']}-H",
            "source_id": source["source_id"], "group_id": source["group_id"], "parent_record_id": "",
            "corpus_id": "H", "intervention_present": 0, "intervention_level": 0, "operation": "original",
            "provider": "", "model_name": "", "model_snapshot": "", "prompt_id": "",
            "group_split": source["group_split"], "source_year": source["source_year"],
            "evaluation_role": "pilot_only" if source["source_id"] in pilot_source_ids else "benchmark",
            "source_title_raw": source["source_title_raw"], "source_title_clean": source["source_title_clean"],
            "title_sha256": source["title_sha256"],
            "abstract_raw": source["abstract_raw"],
            "abstract_clean": source["abstract_clean"], "input_sha256": source["abstract_sha256"],
            "output_sha256": source["abstract_sha256"], "quality_status": "accepted",
            "char_count": len(source["abstract_clean"]), "chinese_ratio": source["chinese_ratio"],
            "normalized_edit_distance": 0.0, "numeric_retention": 1.0,
            "quality_reason": "", "raw_evidence_sha256": "",
        })
    for task in queue:
        raw_path = args.root / task["raw_path"] if task.get("raw_path") else None
        if not raw_path or not raw_path.exists() or task.get("status") != "accepted":
            continue
        raw_bytes = raw_path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
        output_value = payload.get("generated_abstract", payload.get("abstract"))
        if output_value is None and isinstance(payload.get("response"), dict):
            try:
                output_value = payload["response"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                output_value = ""
        output_raw = str(output_value or "")
        source = sources[task["source_id"]]
        output = clean_text(output_raw)
        distance = edit_distance(source["abstract_clean"], output)
        quality = "accepted"
        reason = ""
        if not output:
            quality, reason = "rejected_empty", "empty_output"
        elif PROMPT_LEAK_RE.search(output):
            quality, reason = "rejected_prompt_leakage", "prompt_or_explanation_in_output"
        elif chinese_ratio(output) < 0.60:
            quality, reason = "rejected_language", "low_chinese_ratio"
        elif len(output) < 50:
            quality, reason = "rejected_truncated", "too_short"
        records.append({
            "record_id": task["record_id"], "source_id": task["source_id"], "group_id": task["group_id"],
            "parent_record_id": task["parent_record_id"], "corpus_id": "E", "intervention_present": 1,
            "intervention_level": int(task["intervention_level"]), "operation": task["operation"],
            "provider": task["provider"], "model_name": task.get("model_name", ""),
            "model_snapshot": task.get("model_snapshot", ""), "prompt_id": task["prompt_id"],
            "group_split": task["group_split"], "source_year": source["source_year"],
            "evaluation_role": "pilot_only" if task["source_id"] in pilot_source_ids else "benchmark",
            "source_title_raw": source["source_title_raw"], "source_title_clean": source["source_title_clean"],
            "title_sha256": source["title_sha256"],
            "abstract_raw": output_raw, "abstract_clean": output,
            "input_sha256": task["input_sha256"], "output_sha256": sha256_bytes(output.encode("utf-8")),
            "raw_evidence_sha256": sha256_bytes(raw_bytes),
            "quality_status": quality, "quality_reason": reason, "char_count": len(output),
            "chinese_ratio": chinese_ratio(output),
            "normalized_edit_distance": distance / max(len(source["abstract_clean"]), 1),
            "numeric_retention": retained_numbers(source["abstract_clean"], output),
        })
    write_jsonl(args.root / "processed/records.jsonl", records)
    flat_path = args.root / "processed/records.csv"
    try:
        write_flat_csv(flat_path, records)
    except PermissionError:
        # A spreadsheet viewer may hold the previous flat export open.
        # Preserve the new result under a deterministic alternate name.
        flat_path = args.root / "processed/records_with_title.csv"
        write_flat_csv(flat_path, records)
    stats = {
        "records": len(records),
        "flat_csv": str(flat_path),
        "by_corpus": dict(Counter(str(x.get("corpus_id", "")) for x in records)),
        "accepted_by_provider_level": dict(Counter(
            f"{x.get('provider', '')}:{x.get('intervention_level', '')}"
            for x in records if x.get("corpus_id") == "E" and x.get("quality_status") == "accepted"
        )),
        "quality_status": dict(Counter(str(x.get("quality_status", "")) for x in records)),
        "mean_edit_distance_accepted_e": (
            sum(float(x.get("normalized_edit_distance", 0.0)) for x in records
                if x.get("corpus_id") == "E" and x.get("quality_status") == "accepted") /
            max(sum(x.get("corpus_id") == "E" and x.get("quality_status") == "accepted" for x in records), 1)
        ),
    }
    (args.root / "evaluation/dataset_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"records_written": len(records), "h_records": len(sources), "accepted_edit_records": sum(x.get("quality_status") == "accepted" and x.get("corpus_id") == "E" for x in records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
