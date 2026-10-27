"""Analyze paired detector scores and source-group uncertainty."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def interval(values: list[float]) -> list[float]:
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research_benchmark_legal_edit_v1"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    predictions = [row for row in read_jsonl(args.root / "evaluation/char_ngram_test_predictions.jsonl")
                   if row["experiment"] == "pooled"]
    by_source: defaultdict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        by_source[row["source_id"]].append(row)
    source_ids = sorted(by_source)
    expected = {"deepseek", "doubao", "kimi", "wenxin"}
    for source_id, records in by_source.items():
        h = [record for record in records if record["corpus_id"] == "H"]
        edits = [record for record in records if record["corpus_id"] == "E"]
        if len(h) != 1 or len(edits) != 16 or {record["provider"] for record in edits} != expected:
            raise ValueError(f"Incomplete pooled test group: {source_id}")

    rng = np.random.default_rng(20260921)
    samples = [rng.choice(source_ids, size=len(source_ids), replace=True).tolist()
               for _ in range(args.bootstrap)]
    result = {"test_source_groups": len(source_ids), "bootstrap_replicates": args.bootstrap,
              "unit_of_resampling": "source_id", "by_provider_level": {}}

    h_by_source = {sid: next(row for row in records if row["corpus_id"] == "H")
                   for sid, records in by_source.items()}
    test_fpr = np.mean([h_by_source[sid]["detected"] for sid in source_ids])
    boot_fpr = [np.mean([h_by_source[sid]["detected"] for sid in sample]) for sample in samples]
    result["test_fpr"] = round(float(test_fpr), 4)
    result["test_fpr_95ci"] = interval(boot_fpr)

    for provider in sorted(expected):
        sequences = {}
        for sid, records in by_source.items():
            edits = {int(row["intervention_level"]): row for row in records
                     if row["corpus_id"] == "E" and row["provider"] == provider}
            if set(edits) != {1, 2, 3, 4}:
                raise ValueError(f"Missing edit level: {sid}, {provider}")
            sequences[sid] = [edits[level] for level in (1, 2, 3, 4)]
        monotonic = [all(a["score"] < b["score"] for a, b in zip(sequences[sid], sequences[sid][1:]))
                     for sid in source_ids]
        correlations = []
        undefined_correlations = 0
        for sid in source_ids:
            scores = [r["score"] for r in sequences[sid]]
            if len(set(scores)) < 2:
                undefined_correlations += 1
            else:
                correlations.append(float(spearmanr([1, 2, 3, 4], scores).statistic))
        result[provider] = {
            "strictly_increasing_score_fraction": round(float(np.mean(monotonic)), 4),
            "mean_within_source_spearman": round(float(np.mean(correlations)), 4) if correlations else None,
            "undefined_spearman_count": undefined_correlations,
        }
        for level in range(1, 5):
            key = f"{provider}:E{level}"
            tpr_by_source = {sid: float(sequences[sid][level - 1]["detected"]) for sid in source_ids}
            delta_by_source = {sid: sequences[sid][level - 1]["score"] - h_by_source[sid]["score"]
                               for sid in source_ids}
            boot_tpr = [np.mean([tpr_by_source[sid] for sid in sample]) for sample in samples]
            boot_delta = [np.mean([delta_by_source[sid] for sid in sample]) for sample in samples]
            result["by_provider_level"][key] = {
                "tpr": round(float(np.mean(list(tpr_by_source.values()))), 4),
                "tpr_95ci": interval(boot_tpr),
                "mean_score_change_from_h": round(float(np.mean(list(delta_by_source.values()))), 4),
                "score_change_95ci": interval(boot_delta),
                "fraction_scored_above_h": round(float(np.mean([value > 0 for value in delta_by_source.values()])), 4),
            }
    output = args.root / "evaluation/paired_score_analysis.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"test_source_groups": result["test_source_groups"],
                      "test_fpr": result["test_fpr"], "test_fpr_95ci": result["test_fpr_95ci"],
                      "score_monotonic_fraction_by_provider": {
                          p: result[p]["strictly_increasing_score_fraction"] for p in sorted(expected)}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
