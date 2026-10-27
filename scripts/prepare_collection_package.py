"""Create an execution-ready, provenance-first collection package."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROVIDERS = ["deepseek", "doubao", "kimi", "wenxin"]
PROMPTS = ["P0"]
SPECIAL = re.compile(r"AbstractFilter\([^)]*\);?|更多\s*还原|附视频", re.I)


def clean(value):
    text = SPECIAL.sub("", str(value or ""))
    text = re.sub(r"^\s*摘要\s*[:：]\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def allocate(total, counts):
    years = sorted(counts)
    raw = {year: total * counts[year] / sum(counts.values()) for year in years}
    n = {year: int(raw[year]) for year in years}
    remainder = total - sum(n.values())
    for year in sorted(years, key=lambda y: (raw[y] - n[y], -y), reverse=True)[:remainder]:
        n[year] += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=Path("中国法学/DATA/ai_rewrite/generation_tasks.csv"))
    parser.add_argument("--out", type=Path, default=Path("research_benchmark_v1"))
    parser.add_argument("--n-history", type=int, default=600)
    parser.add_argument("--n-challenge", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    tasks = pd.read_csv(args.tasks, dtype=str, keep_default_na=False)
    tasks = tasks[tasks.year.str.fullmatch(r"20\d{2}")].copy()
    tasks["source_year"] = tasks.year.astype(int)
    tasks["source_title_clean"] = tasks.source_title.map(clean)
    tasks["source_abstract_clean"] = tasks.source_abstract.map(clean)
    tasks = tasks[(tasks.source_title_clean != "") & (tasks.source_abstract_clean != "")]
    history = tasks[tasks.source_year.between(2000, 2020)].copy()
    unknown = tasks[tasks.source_year.between(2021, 2025)].copy()
    counts = history.groupby("source_year").size().to_dict()
    allocation = allocate(args.n_history, counts)
    selected = []
    for year, n in allocation.items():
        pool = history[history.source_year == year]
        indices = rng.choice(pool.index.to_numpy(), size=min(n, len(pool)), replace=False)
        selected.append(pool.loc[indices])
    selected = pd.concat(selected, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    selected["group_id"] = selected.source_id.map(lambda x: "group-" + hashlib.sha256(x.encode()).hexdigest()[:16])
    selected["group_split"] = ""
    for year in sorted(selected.source_year.unique()):
        idx = selected.index[selected.source_year == year].to_numpy()
        rng.shuffle(idx)
        cuts = [round(len(idx) * 0.50), round(len(idx) * 0.667), round(len(idx) * 0.834)]
        selected.loc[idx[:cuts[0]], "group_split"] = "train"
        selected.loc[idx[cuts[0]:cuts[1]], "group_split"] = "dev"
        selected.loc[idx[cuts[1]:cuts[2]], "group_split"] = "calibration"
        selected.loc[idx[cuts[2]:], "group_split"] = "test"
    selected["label"] = "archival_reference"
    selected["label_basis"] = "archived_source_assumption; not authorship proof"
    selected["corpus_id"] = "H"

    unknown = unknown.sort_values(["source_year", "source_id"]).copy()
    unknown["group_id"] = unknown.source_id.map(lambda x: "group-" + hashlib.sha256(x.encode()).hexdigest()[:16])
    unknown["corpus_id"] = "U"
    unknown["label"] = ""
    unknown["label_basis"] = "unknown_source; inference_only"
    unknown["group_split"] = "inference"

    for name in ["protocol", "source_registry", "raw", "processed", "splits", "evaluation", "inference"]:
        (args.out / name).mkdir(parents=True, exist_ok=True)

    registry_cols = ["source_id", "group_id", "corpus_id", "source_year", "source_time_raw", "source_title_clean", "source_abstract_clean", "group_split", "label", "label_basis"]
    selected.rename(columns={"source_time": "source_time_raw"})[registry_cols].to_csv(args.out / "source_registry/history_selected.csv", index=False, encoding="utf-8-sig")
    unknown.rename(columns={"source_time": "source_time_raw"})[registry_cols].to_csv(args.out / "source_registry/unknown_2021_2025.csv", index=False, encoding="utf-8-sig")

    queue = []
    for row in selected.itertuples(index=False):
        for provider in PROVIDERS:
            for prompt_id in PROMPTS:
                queue.append({
                    "record_id": f"{provider}-{row.source_id}-{prompt_id}", "source_id": row.source_id,
                    "group_id": row.group_id, "source_year": row.source_year, "group_split": row.group_split,
                    "provider": provider, "prompt_id": prompt_id, "model_name": "",
                    "model_snapshot": "", "access_channel": "web", "collection_batch": "",
                    "status": "pending", "raw_evidence_path": "", "evidence_sha256": "",
                    "generated_title": "", "generated_abstract": "", "quality_status": "",
                    "failure_reason": "", "generation_time_utc": "", "collector_note": ""
                })
    queue_df = pd.DataFrame(queue)
    queue_df.to_csv(args.out / "source_registry/generation_queue_p0.csv", index=False, encoding="utf-8-sig")

    challenge_sources = selected[selected.group_split == "test"].sample(n=min(args.n_challenge, int((selected.group_split == "test").sum())), random_state=args.seed)
    challenge = []
    for row in challenge_sources.itertuples(index=False):
        for provider in PROVIDERS:
            for prompt_id in ["P1", "P2"]:
                challenge.append({"record_id": f"{provider}-{row.source_id}-{prompt_id}", "source_id": row.source_id,
                                  "group_id": row.group_id, "source_year": row.source_year, "group_split": "test",
                                  "provider": provider, "prompt_id": prompt_id, "model_name": "", "model_snapshot": "",
                                  "access_channel": "web", "collection_batch": "", "status": "pending",
                                  "raw_evidence_path": "", "evidence_sha256": "", "generated_title": "",
                                  "generated_abstract": "", "quality_status": "", "failure_reason": "",
                                  "generation_time_utc": "", "collector_note": ""})
    pd.DataFrame(challenge).to_csv(args.out / "source_registry/generation_queue_challenge_p1_p2.csv", index=False, encoding="utf-8-sig")

    prompts = {
        "P0.txt": "以下标题仅作为法律研究主题线索：{source_title}\n请围绕该主题独立拟定一篇中文法学论文的标题与摘要。摘要长度参照 length_band。不要虚构实证数据、案例编号或文献引用。请分别输出标题和摘要。",
        "P1.txt": "请围绕以下法律研究主题独立写作，提出明确研究问题、规范分析路径和制度含义。仅输出新的论文标题和摘要，不复述原摘要：{source_title}",
        "P2.txt": "请将以下法律主题改写为一篇具有明确争点的中文法学论文摘要，并拟定新标题。不要捏造数据、案例编号或参考文献。仅输出标题和摘要：{source_title}"
    }
    for name, content in prompts.items():
        (args.out / "protocol" / name).write_text(content + "\n", encoding="utf-8")

    manifest = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(), "seed": args.seed,
        "history_pool_rows": int(len(history)), "selected_history_rows": int(len(selected)),
        "unknown_2021_2025_rows": int(len(unknown)), "allocation_by_year": {str(k): int(v) for k, v in allocation.items()},
        "provider_count": len(PROVIDERS), "providers": PROVIDERS, "p0_queue_rows": len(queue_df),
        "challenge_queue_rows": len(challenge), "source_registry_sha256": hashlib.sha256((args.out / "source_registry/history_selected.csv").read_bytes()).hexdigest(),
        "label_policy": {"H": "archival_reference", "G": "controlled_generated", "U": "unknown_source"}
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
