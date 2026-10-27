"""Prepare reproducible, non-exclusionary content review for CLAW-Edit."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DIGIT_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?%?(?![A-Za-z0-9])")
ARTICLE_RE = re.compile(r"第\s*[一二三四五六七八九十百千万零〇两0-9]+\s*[条款项章节]")
QUOTED_TITLE_RE = re.compile(r"《[^》\n]{1,40}》")
NEGATION_RE = re.compile(r"不得|不能|不应|不宜|禁止|必须|应当|可以|无需|不必")
REFUSAL_RE = re.compile(
    r"作为(?:一个)?人工智能语言模型|作为AI(?:语言)?模型|我还没学习如何回答|"
    r"我无法(?:回答|处理|完成)|抱歉.{0,30}(?:无法|不能)(?:回答|处理|完成)"
)
SOURCE_GENRE_RE = re.compile(r"纪事|简讯|年会综述|会议综述|大会综述|座谈会上的讲话|征稿|目录|公告")
CRITERIA = {
    "substantive_change": "legal proposition, conclusion, scope, or legal basis changes",
    "meaning_preserved": "legal proposition, conclusion, scope, and legal basis preserved",
    "uncertain": "requires specialist adjudication or fuller source context",
    "not_assessable": "source or output is not a substantive abstract",
}
REVIEW_FIELDS = ("review_verdict", "review_reason", "reviewer_id", "review_date")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def token_delta(pattern: re.Pattern[str], source: str, edited: str) -> tuple[list[str], list[str]]:
    before = Counter(match.group().replace(" ", "") for match in pattern.finditer(source))
    after = Counter(match.group().replace(" ", "") for match in pattern.finditer(edited))
    return sorted((before - after).elements()), sorted((after - before).elements())


def prepare(root: Path, out: Path, seed: int, random_per_cell: int) -> dict:
    records = read_jsonl(root / "evaluation/benchmark_records.jsonl")
    flags = read_jsonl(root / "evaluation/content_review_flags.jsonl")
    by_id = {record["record_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate benchmark record IDs")
    flags_by_id: defaultdict[str, list[str]] = defaultdict(list)
    for item in flags:
        if item["record_id"] in by_id:
            flags_by_id[item["record_id"]].append(item["flag"])

    rows: list[dict] = []
    for record in records:
        if record["corpus_id"] != "E":
            continue
        source = by_id.get(record["parent_record_id"])
        if not source or source["corpus_id"] != "H" or source["source_id"] != record["source_id"]:
            raise ValueError(f"Missing or invalid parent for {record['record_id']}")
        original = source["abstract_clean"]
        edited = record["abstract_clean"]
        digits_removed, digits_added = token_delta(DIGIT_RE, original, edited)
        articles_removed, articles_added = token_delta(ARTICLE_RE, original, edited)
        quoted_titles_removed, quoted_titles_added = token_delta(QUOTED_TITLE_RE, original, edited)
        negations_removed, negations_added = token_delta(NEGATION_RE, original, edited)
        audit_flags = sorted(flags_by_id[record["record_id"]])
        length_ratio = len(edited) / max(len(original), 1)
        distance = float(record["normalized_edit_distance"])
        refusal_alert = bool(REFUSAL_RE.search(edited))
        source_genre_alert = bool(SOURCE_GENRE_RE.search(source["source_title_clean"]))
        source_truncation_alert = original.startswith("<正>") or bool(
            original and original[-1] not in "。！？!?；;." and len(original) < 250
        )
        if (refusal_alert or digits_removed or articles_removed or quoted_titles_removed or
                "large_edit_distance" in audit_flags or "length_ratio_outside_0.5_2.0" in audit_flags):
            priority = "P1"
        elif (source_genre_alert or source_truncation_alert or "numeric_content_changed" in audit_flags
              or negations_removed or negations_added):
            priority = "P2"
        else:
            priority = "P3"
        rows.append({
            "record_id": record["record_id"],
            "source_id": record["source_id"],
            "group_id": record["group_id"],
            "group_split": record["group_split"],
            "source_year": record["source_year"],
            "provider": record["provider"],
            "level": f"E{record['intervention_level']}",
            "source_title": source["source_title_clean"],
            "source_sha256": source["output_sha256"],
            "edited_sha256": record["output_sha256"],
            "priority": priority,
            "audit_flags": audit_flags,
            "refusal_output_alert": refusal_alert,
            "source_genre_alert": source_genre_alert,
            "source_truncation_alert": source_truncation_alert,
            "edit_distance": distance,
            "length_ratio": round(length_ratio, 4),
            "digits_removed": digits_removed,
            "digits_added": digits_added,
            "articles_removed": articles_removed,
            "articles_added": articles_added,
            "quoted_titles_removed": quoted_titles_removed,
            "quoted_titles_added": quoted_titles_added,
            "modal_terms_removed": negations_removed,
            "modal_terms_added": negations_added,
            "original_abstract": original,
            "edited_abstract": edited,
            "review_verdict": "",
            "review_reason": "",
            "reviewer_id": "",
            "review_date": "",
        })
    expected = sum(record["corpus_id"] == "H" for record in records) * 16
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} benchmark edits, got {len(rows)}")

    rng = random.Random(seed)
    by_cell: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[(row["provider"], row["level"])].append(row)
    selected: defaultdict[str, set[str]] = defaultdict(set)
    for cell in sorted(by_cell):
        pool = sorted(by_cell[cell], key=lambda row: row["record_id"])
        for row in rng.sample(pool, random_per_cell):
            selected[row["record_id"]].add("probability_sample")

    # Keep every pre-existing extreme flag, then sample factual-anchor flags.
    for row in rows:
        if {"large_edit_distance", "length_ratio_outside_0.5_2.0"} & set(row["audit_flags"]):
            selected[row["record_id"]].add("extreme_flag_census")
        if row["refusal_output_alert"]:
            selected[row["record_id"]].add("refusal_alert_census")
        if row["articles_removed"] or row["articles_added"]:
            selected[row["record_id"]].add("article_reference_census")
        if row["quoted_titles_removed"] or row["quoted_titles_added"]:
            selected[row["record_id"]].add("quoted_title_census")
    for cell in sorted(by_cell):
        pool = [row for row in sorted(by_cell[cell], key=lambda row: row["record_id"])
                if row["priority"] == "P1" and row["record_id"] not in selected]
        for row in rng.sample(pool, min(2, len(pool))):
            selected[row["record_id"]].add("anchor_risk_sample")

    review_sample = [{**row, "selection_reasons": sorted(selected[row["record_id"]])}
                     for row in rows if row["record_id"] in selected]
    counts = Counter(row["priority"] for row in rows)
    summary = {
        "scope": "formal benchmark E records only; no pilot records or H records",
        "edit_count": len(rows),
        "source_count": len({row["source_id"] for row in rows}),
        "seed": seed,
        "random_per_provider_level": random_per_cell,
        "probability_sample_count": sum("probability_sample" in reasons for reasons in selected.values()),
        "extreme_flag_census_count": sum("extreme_flag_census" in reasons for reasons in selected.values()),
        "refusal_alert_census_count": sum("refusal_alert_census" in reasons for reasons in selected.values()),
        "article_reference_census_count": sum("article_reference_census" in reasons for reasons in selected.values()),
        "quoted_title_census_count": sum("quoted_title_census" in reasons for reasons in selected.values()),
        "anchor_risk_sample_count": sum("anchor_risk_sample" in reasons for reasons in selected.values()),
        "selected_unique_count": len(review_sample),
        "priority_counts": dict(sorted(counts.items())),
        "priority_by_provider_level": {
            f"{provider}:{level}": dict(sorted(Counter(row["priority"] for row in cell_rows).items()))
            for (provider, level), cell_rows in sorted(by_cell.items())
        },
        "token_alert_counts": {
            "digits_removed": sum(bool(row["digits_removed"]) for row in rows),
            "articles_removed": sum(bool(row["articles_removed"]) for row in rows),
            "quoted_titles_removed": sum(bool(row["quoted_titles_removed"]) for row in rows),
            "modal_terms_changed": sum(bool(row["modal_terms_removed"] or row["modal_terms_added"]) for row in rows),
            "refusal_outputs": sum(row["refusal_output_alert"] for row in rows),
            "source_genre_alert_groups": len({row["source_id"] for row in rows if row["source_genre_alert"]}),
            "source_truncation_alert_groups": len({row["source_id"] for row in rows if row["source_truncation_alert"]}),
        },
        "review_verdicts": CRITERIA,
        "interpretation": "All flags are screening cues, not semantic-fidelity judgments or exclusion rules.",
    }
    for filename in ("review_index.jsonl", "review_sample.jsonl"):
        existing_path = out / filename
        if existing_path.exists():
            existing = read_jsonl(existing_path)
            if any(any(row.get(field) for field in REVIEW_FIELDS) for row in existing):
                raise ValueError(f"Refusing to overwrite completed review fields in {existing_path}")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "review_index.jsonl", rows)
    write_jsonl(out / "review_sample.jsonl", review_sample)
    (out / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research_benchmark_legal_edit_v1"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--random-per-cell", type=int, default=8)
    args = parser.parse_args()
    out = args.out or args.root / "evaluation/content_review_v1"
    print(json.dumps(prepare(args.root, out, args.seed, args.random_per_cell), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
