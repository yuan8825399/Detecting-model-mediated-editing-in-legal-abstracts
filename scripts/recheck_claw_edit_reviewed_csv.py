"""Recheck the manually reviewed CSV without changing it or the v2 benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from export_claw_edit_analysis import LIKELY_TRUNCATED, make_report, missing_terminal_punctuation
from process_claw_edit_outputs import clean_text
from review_claw_edit_content import REFUSAL_RE


def sha256(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def recheck(reviewed_csv: Path, v2_records: Path, out: Path) -> dict:
    csv_bytes = reviewed_csv.read_bytes()
    records = list(csv.DictReader(io.StringIO(csv_bytes.decode("gbk"), newline="")))
    v2 = [json.loads(line) for line in v2_records.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["record_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate record_id in reviewed CSV")
    by_source = Counter(row["source_id"] for row in records)
    if set(by_source.values()) != {17}:
        raise ValueError("Reviewed CSV contains incomplete source groups")
    originals = {row["source_id"]: row for row in records if row["corpus_id"] == "H"}
    if len(originals) != len(by_source):
        raise ValueError("A source group does not contain exactly one H record")

    missing_end = [row for row in originals.values() if missing_terminal_punctuation(row["abstract_clean"])]
    raw_clean_mismatch = [row for row in records if clean_text(row["abstract_raw"]) != row["abstract_clean"]]
    output_hash_mismatch = [row for row in records if sha256(row["abstract_clean"]) != row["output_sha256"]]
    input_hash_mismatch = [row for row in records
                           if sha256(originals[row["source_id"]]["abstract_clean"]) != row["input_sha256"]]
    refusal_rows = [row for row in records if REFUSAL_RE.search(row["abstract_clean"])]
    v2_sources = {row["source_id"] for row in v2 if row["corpus_id"] == "H"}
    formal_sources = {row["source_id"] for row in records if row["evaluation_role"] != "pilot_only"}
    removed_sources = sorted(v2_sources - formal_sources)
    summary = {
        "reviewed_csv_sha256": sha256(csv_bytes),
        "records": len(records),
        "source_groups": len(by_source),
        "formal_source_groups": len(formal_sources),
        "pilot_records": sum(row["evaluation_role"] == "pilot_only" for row in records),
        "removed_v2_source_ids": removed_sources,
        "refusal_clean_count": len(refusal_rows),
        "marker_clean_count": sum("<正>" in row["abstract_clean"] for row in records),
        "marker_raw_count": sum("<正>" in row["abstract_raw"] for row in records),
        "h_missing_terminal_count": len(missing_end),
        "h_likely_truncated_count": sum(row["source_id"] in LIKELY_TRUNCATED for row in missing_end),
        "raw_clean_mismatch_count": len(raw_clean_mismatch),
        "raw_clean_mismatch_by_source": dict(sorted(Counter(row["source_id"] for row in raw_clean_mismatch).items())),
        "output_hash_mismatch_count": len(output_hash_mismatch),
        "input_hash_mismatch_count": len(input_hash_mismatch),
    }
    lines = [
        "# 人工复核 CSV 再检出", "",
        f"输入 CSV SHA-256：`{summary['reviewed_csv_sha256']}`。", "",
        f"记录 {len(records)} 条，来源组 {len(by_source)} 个（正式 {len(formal_sources)} 个，"
        f"试运行 {summary['pilot_records']} 条）；每组 17 条，记录 ID 无重复。", "",
        f"- 当前文本拒答提示：{len(refusal_rows)} 条。",
        f"- `<正>`：`abstract_clean` {summary['marker_clean_count']} 条，"
        f"`abstract_raw` {summary['marker_raw_count']} 条。",
        f"- 原始摘要末尾缺句末标点：{len(missing_end)} 条；"
        f"其中疑似截断 {summary['h_likely_truncated_count']} 条。",
        f"- `clean_text(abstract_raw)` 与 `abstract_clean` 不一致：{len(raw_clean_mismatch)} 条。",
        f"- 输出哈希不匹配：{len(output_hash_mismatch)} 条；输入哈希不匹配：{len(input_hash_mismatch)} 条。",
        "",
        "## 与现有 v2 的差别", "",
        "最新 CSV 少了以下 8 个 v2 正式来源组：", "",
    ]
    lines.extend(f"- `{source_id}`" for source_id in removed_sources)
    lines.extend(["", "现有 v2 JSONL 及其检测结果未由本次检查改写。", "",
                  "## 原始字段与分析字段不一致", "",
                  "以下来源组的原始文本与分析文本不一致；人工修订的 `abstract_raw` "
                  "尚未同步为可用于检测的 `abstract_clean`：", ""])
    lines.extend(f"- `{source_id}`：{count} 条" for source_id, count in
                 summary["raw_clean_mismatch_by_source"].items())
    lines.extend(["", make_report(records).replace("# 原始摘要句末复核清单", "## 原始摘要句末复核清单", 1)])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("research_benchmark_legal_edit_v1/processed/records.csv"))
    parser.add_argument("--v2", type=Path, default=Path("research_benchmark_legal_edit_v2/evaluation/benchmark_records.jsonl"))
    parser.add_argument("--out", type=Path,
                        default=Path("research_benchmark_legal_edit_v1/evaluation/reviewed_csv_recheck.md"))
    args = parser.parse_args()
    print(json.dumps(recheck(args.csv, args.v2, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
