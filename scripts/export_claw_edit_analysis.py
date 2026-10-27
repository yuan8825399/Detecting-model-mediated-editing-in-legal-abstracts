"""Export marker-free analysis records and review original abstract endings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TERMINAL_PUNCTUATION = frozenset("。！？；.!?;…")
LIKELY_TRUNCATED = frozenset({
    "china-law-2004-00042", "china-law-2004-00043", "china-law-2004-00047",
    "china-law-2004-00094", "china-law-2004-00115", "china-law-2004-00138",
    "china-law-2005-00093", "china-law-2005-00094", "china-law-2008-00017",
    "china-law-2011-00036", "china-law-2012-00094", "china-law-2013-00002",
    "china-law-2014-00002", "china-law-2014-00082",
})


def missing_terminal_punctuation(text: str) -> bool:
    ending = text.rstrip().rstrip('"”’\'）)】]')
    return not ending or ending[-1] not in TERMINAL_PUNCTUATION


def markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def make_report(records: list[dict]) -> str:
    originals = [row for row in records if row["corpus_id"] == "H"]
    flagged = [row for row in originals if missing_terminal_punctuation(row["abstract_clean"])]
    flagged_ids = {row["source_id"] for row in flagged}
    visible_ids = {row["source_id"] for row in originals}
    if not (LIKELY_TRUNCATED & visible_ids) <= flagged_ids:
        raise ValueError("A reviewed truncation ID is no longer missing terminal punctuation")
    lines = [
        "# 原始摘要句末复核清单", "",
        f"原始摘要 {len(originals)} 条；末尾缺少句末标点 {len(flagged)} 条；"
        f"其中疑似截断 {len(flagged_ids & LIKELY_TRUNCATED)} 条，其余需核对原刊 {len(flagged_ids - LIKELY_TRUNCATED)} 条。",
        "",
        "判定只依据当前原始摘要的末尾文字。疑似截断不等于已核实原刊缺文；"
        "其余条目也不能仅凭语句通顺就确认原刊完整。未自动补写、加标点或删样本。",
        "",
    ]
    for title, rows in (
        ("疑似截断", [row for row in flagged if row["source_id"] in LIKELY_TRUNCATED]),
        ("缺句末标点，尚无明显截断证据", [row for row in flagged if row["source_id"] not in LIKELY_TRUNCATED]),
    ):
        lines.extend([f"## {title}（{len(rows)} 条）", "",
                      "| source_id | 标题 | 分区 | 字数 | 原文末尾 |",
                      "| --- | --- | --- | ---: | --- |"])
        for row in sorted(rows, key=lambda item: item["source_id"]):
            cells = (row["source_id"], row["source_title_clean"], row["group_split"],
                     len(row["abstract_clean"]), "…" + row["abstract_clean"][-42:])
            lines.append("| " + " | ".join(map(markdown_cell, cells)) + " |")
        lines.append("")
    return "\n".join(lines)


def export(root: Path) -> tuple[int, int]:
    input_path = root / "evaluation/benchmark_records.jsonl"
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any("<正>" in row["abstract_clean"] for row in records):
        raise ValueError("abstract_clean still contains <正>; rebuild the derived benchmark first")
    report = make_report(records)
    analysis_path = root / "evaluation/analysis_records.jsonl"
    with analysis_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            analysis_row = {key: value for key, value in row.items() if key != "abstract_raw"}
            if "<正>" in json.dumps(analysis_row, ensure_ascii=False):
                raise ValueError(f"Marker remains in analysis row {row['record_id']}")
            handle.write(json.dumps(analysis_row, ensure_ascii=False) + "\n")
    (root / "evaluation/original_abstract_endings.md").write_text(report, encoding="utf-8")
    flagged_count = sum(row["corpus_id"] == "H" and
                        missing_terminal_punctuation(row["abstract_clean"]) for row in records)
    return len(records), flagged_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research_benchmark_legal_edit_v2"))
    args = parser.parse_args()
    record_count, flagged_count = export(args.root)
    print(f"Exported {record_count} analysis records and {flagged_count} ending-review entries")


if __name__ == "__main__":
    main()
