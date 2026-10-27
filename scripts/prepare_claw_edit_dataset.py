"""Build the CLAW-Edit source registry and reproducible LLM edit queue.

This package models controlled LLM edits of the same Chinese legal abstract.
It deliberately does not create human-polished or human-post-edited records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_PROVIDERS = ("deepseek", "doubao", "kimi", "wenxin")
EDIT_LEVELS = (
    ("E1", 1, "grammar_correction"),
    ("E2", 2, "academic_polish"),
    ("E3", 3, "sentence_restructure"),
    ("E4", 4, "abstract_rewrite"),
)
SPECIAL_RE = re.compile(r"AbstractFilter\([^)]*\);?|更多\s*还原|附视频", re.I)
PROMPT_RE = re.compile(r"^(?:摘要|abstract)\s*[:：]\s*", re.I)
ZH_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def clean_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = SPECIAL_RE.sub("", text)
    text = PROMPT_RE.sub("", text.strip())
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def clean_title(value: str) -> str:
    text = clean_text(value)
    return re.sub(r"\s*(?:下载|附视频)\s*$", "", text, flags=re.I).strip()


def chinese_ratio(text: str) -> float:
    zh = len(ZH_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    return zh / (zh + latin) if zh + latin else 0.0


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def group_id(source_id: str) -> str:
    return "group-" + hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prompt_text(level: str) -> str:
    common = (
        "只输出修改后的摘要，不要输出标题、解释或修改说明。\n"
        "不得新增事实、数据、案例、法条、参考文献或法律结论。\n"
        "必须保留原摘要的研究问题、法律观点和主要结论。\n\n"
        "原摘要：\n{abstract}"
    )
    instructions = {
        "E1": "请对以下中文法学论文摘要做最小范围的错别字、标点和明显语法纠正。不得改变论证顺序或句子结构。",
        "E2": "请对以下中文法学论文摘要进行学术语言润色。可以改善流畅性、术语表达和句间衔接，但不得进行段落或论证结构重组。",
        "E3": "请对以下中文法学论文摘要进行句子层面的结构重组。可以拆分、合并或调整句法，但不得改变核心法律命题。",
        "E4": "请对以下中文法学论文摘要进行整体语言改写。可以重新组织句子和段落，使摘要更加凝练、正式和连贯。",
    }
    return instructions[level] + "\n" + common + "\n"


def make_dirs(root: Path) -> None:
    for name in ("protocol/edit_prompts", "source_registry", "raw", "processed", "splits", "evaluation"):
        (root / name).mkdir(parents=True, exist_ok=True)


def load_original_xlsx_titles(source_dir: Path) -> dict[str, str]:
    """Map the first XLSX column to source ids using year and data-row order."""
    titles: dict[str, str] = {}
    for path in sorted(source_dir.glob("*.xlsx")):
        match = re.search(r"(?<!\d)(20\d{2})(?!\d)", path.name)
        if not match:
            continue
        year = int(match.group(1))
        if not 2000 <= year <= 2025:
            continue
        frame = pd.read_excel(path, dtype=str).fillna("")
        if frame.shape[1] == 0:
            continue
        for index, value in enumerate(frame.iloc[:, 0].tolist(), 1):
            title = unicodedata.normalize("NFKC", str(value))
            title = re.sub(r"\s+", " ", title).strip()
            title = re.sub(r"\s*下载\s*$", "", title).strip()
            if title:
                titles[f"china-law-{year}-{index:05d}"] = title
    return titles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("research_benchmark_v1/source_registry/history_selected.csv"))
    parser.add_argument("--out", type=Path, default=Path("research_benchmark_legal_edit_v1"))
    parser.add_argument("--source-dir", type=Path, default=Path("中国法学"))
    parser.add_argument("--min-chinese-ratio", type=float, default=0.60)
    parser.add_argument("--min-chinese-chars", type=int, default=100)
    parser.add_argument("--providers", nargs="+", default=list(DEFAULT_PROVIDERS))
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    existing_queue = args.out / "source_registry/generation_queue.csv"
    if existing_queue.exists():
        existing_rows = read_csv(existing_queue)
        if any(row.get("status") == "accepted" or row.get("raw_path") for row in existing_rows):
            raise SystemExit(
                f"Refusing to overwrite collected records in {existing_queue}. "
                "Use a new --out directory for another dataset version."
            )
    make_dirs(args.out)

    input_rows = read_csv(args.input)
    original_titles = load_original_xlsx_titles(args.source_dir)
    sources: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    seen_hashes: dict[str, str] = {}
    required = ("source_id", "source_abstract_clean")
    for row in input_rows:
        if any(not row.get(key, "").strip() for key in required):
            exclusions.append({"source_id": row.get("source_id", ""), "reason": "missing_source_field", "rule_version": "claw-edit-v1"})
            continue
        source_id = row["source_id"].strip()
        abstract = clean_text(row["source_abstract_clean"])
        ratio = chinese_ratio(abstract)
        zh_chars = len(ZH_RE.findall(abstract))
        if zh_chars < args.min_chinese_chars:
            exclusions.append({"source_id": source_id, "reason": "too_few_chinese_characters", "rule_version": "claw-edit-v1"})
            continue
        if ratio < args.min_chinese_ratio:
            exclusions.append({"source_id": source_id, "reason": "low_chinese_ratio", "rule_version": "claw-edit-v1"})
            continue
        digest = sha256_text(abstract)
        if digest in seen_hashes:
            exclusions.append({"source_id": source_id, "reason": f"duplicate_abstract_of_{seen_hashes[digest]}", "rule_version": "claw-edit-v1"})
            continue
        seen_hashes[digest] = source_id
        title_raw = original_titles.get(source_id, row.get("source_title_clean", ""))
        title_clean = clean_title(title_raw)
        if not title_clean:
            exclusions.append({"source_id": source_id, "reason": "missing_original_xlsx_title", "rule_version": "claw-edit-v1"})
            continue
        sources.append({
            "source_id": source_id,
            "group_id": row.get("group_id") or group_id(source_id),
            "source_year": row.get("source_year", ""),
            "source_time_raw": row.get("source_time_raw", ""),
            "source_title_raw": title_raw,
            "source_title_clean": title_clean,
            "title_sha256": sha256_text(title_clean),
            "abstract_raw": row["source_abstract_clean"],
            "abstract_clean": abstract,
            "abstract_sha256": digest,
            "chinese_ratio": f"{ratio:.6f}",
            "chinese_char_count": zh_chars,
            "group_split": row.get("group_split", ""),
            "corpus_id": "H",
            "intervention_level": 0,
            "operation": "original",
            "label": "archival_reference",
            "label_basis": "archived_source_assumption; not authorship proof",
        })

    if not sources:
        raise SystemExit("No source abstracts passed the fixed Chinese-language filters.")

    source_fields = list(sources[0])
    write_csv(args.out / "source_registry/sources.csv", sources, source_fields)
    write_csv(args.out / "source_registry/exclusion_log.csv", exclusions, ["source_id", "reason", "rule_version"])
    write_csv(args.out / "splits/group_split.csv", [{"group_id": x["group_id"], "source_id": x["source_id"], "group_split": x["group_split"]} for x in sources], ["group_id", "source_id", "group_split"])

    queue: list[dict[str, object]] = []
    for source in sources:
        for provider in args.providers:
            for prompt_id, level, operation in EDIT_LEVELS:
                queue.append({
                    "record_id": f"{provider}-{source['source_id']}-{prompt_id}",
                    "source_id": source["source_id"],
                    "group_id": source["group_id"],
                    "parent_record_id": f"{source['source_id']}-H",
                    "source_title_raw": source["source_title_raw"],
                    "source_title_clean": source["source_title_clean"],
                    "title_sha256": source["title_sha256"],
                    "group_split": source["group_split"],
                    "provider": provider,
                    "prompt_id": prompt_id,
                    "intervention_level": level,
                    "operation": operation,
                    "model_name": "",
                    "model_snapshot": "",
                    "access_channel": "web_or_api",
                    "status": "pending",
                    "quality_status": "",
                    "raw_path": "",
                    "input_sha256": source["abstract_sha256"],
                    "output_sha256": "",
                    "prompt_sha256": sha256_text(prompt_text(prompt_id)),
                    "generation_time_utc": "",
                    "failure_reason": "",
                })
    queue_fields = list(queue[0])
    write_csv(args.out / "source_registry/generation_queue.csv", queue, queue_fields)

    for prompt_id, _, _ in EDIT_LEVELS:
        (args.out / "protocol/edit_prompts" / f"{prompt_id}.txt").write_text(prompt_text(prompt_id), encoding="utf-8")

    manifest = {
        "dataset": "CLAW-Edit",
        "version": "1.0",
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source_input": str(args.input),
        "title_source_dir": str(args.source_dir),
        "titles_loaded_from_original_xlsx": len(original_titles),
        "source_rows_input": len(input_rows),
        "source_rows_accepted": len(sources),
        "source_rows_excluded": len(exclusions),
        "providers": args.providers,
        "edit_levels": [{"prompt_id": p, "level": n, "operation": op} for p, n, op in EDIT_LEVELS],
        "queue_rows": len(queue),
        "seed": args.seed,
        "filters": {"min_chinese_ratio": args.min_chinese_ratio, "min_chinese_chars": args.min_chinese_chars},
        "label_policy": {"H": "archival_reference", "E": "controlled_llm_edit", "unknown": "not_included"},
        "human_editing_conditions": False,
        "note": "E1-E4 are controlled operations, not estimates of real author AI contribution.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
