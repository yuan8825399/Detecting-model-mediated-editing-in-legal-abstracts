"""Run a fixed character n-gram baseline on the audited CLAW-Edit benchmark."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected(rows: list[dict], split: str, providers: set[str]) -> list[dict]:
    return [r for r in rows if r["group_split"] == split and
            (r["corpus_id"] == "H" or r["provider"] in providers)]


def text_of(row: dict, input_mode: str) -> str:
    if input_mode == "title_abstract":
        return row["source_title_clean"] + "\n" + row["abstract_clean"]
    return row["abstract_clean"]


def label_of(row: dict) -> int:
    return int(row["corpus_id"] == "E")


def threshold_at_fpr(negative_scores: np.ndarray, max_fpr: float) -> float:
    """Strict > threshold allows at most floor(n * max_fpr) calibration FPs."""
    ranked = np.sort(negative_scores)[::-1]
    allowed = int(np.floor(max_fpr * len(ranked)))
    return float(ranked[allowed]) if allowed < len(ranked) else float("-inf")


def score_test(model, rows: list[dict], threshold: float, input_mode: str) -> tuple[list[dict], dict]:
    scores = model.decision_function([text_of(row, input_mode) for row in rows])
    predictions = []
    for row, score in zip(rows, scores):
        predictions.append({
            "record_id": row["record_id"],
            "source_id": row["source_id"],
            "group_id": row["group_id"],
            "provider": row["provider"],
            "corpus_id": row["corpus_id"],
            "intervention_level": row["intervention_level"],
            "score": float(score),
            "detected": bool(score > threshold),
        })
    negatives = [p for p in predictions if p["corpus_id"] == "H"]
    result = {
        "test_h": len(negatives),
        "test_e": len(predictions) - len(negatives),
        "actual_fpr": sum(p["detected"] for p in negatives) / len(negatives),
        "by_provider_level": {},
    }
    for provider in sorted({p["provider"] for p in predictions if p["corpus_id"] == "E"}):
        for level in range(1, 5):
            positives = [p for p in predictions if p["provider"] == provider and
                         p["corpus_id"] == "E" and p["intervention_level"] == level]
            negative_scores = [p["score"] for p in negatives]
            positive_scores = [p["score"] for p in positives]
            result["by_provider_level"][f"{provider}:E{level}"] = {
                "n": len(positives),
                "auroc": float(roc_auc_score(
                    [0] * len(negative_scores) + [1] * len(positive_scores),
                    negative_scores + positive_scores,
                )),
                "tpr_at_calibrated_threshold": sum(p["detected"] for p in positives) / len(positives),
                "mean_score": float(np.mean(positive_scores)),
            }
    return predictions, result


def run_condition(rows: list[dict], train_providers: set[str], test_providers: set[str],
                  max_fpr: float, input_mode: str) -> tuple[list[dict], dict]:
    train = selected(rows, "train", train_providers)
    calibration = selected(rows, "calibration", train_providers)
    test = selected(rows, "test", test_providers)
    train_groups = {r["group_id"] for r in train}
    calibration_groups = {r["group_id"] for r in calibration}
    test_groups = {r["group_id"] for r in test}
    assert not train_groups & calibration_groups
    assert not train_groups & test_groups
    assert not calibration_groups & test_groups
    y_train = np.array([label_of(row) for row in train])
    counts = Counter(y_train)
    weights = np.array([len(train) / (2 * counts[label]) for label in y_train])
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3,
                        max_features=100000, sublinear_tf=True),
        LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=20260921),
    )
    model.fit([text_of(row, input_mode) for row in train], y_train, logisticregression__sample_weight=weights)
    calibration_h = [row for row in calibration if row["corpus_id"] == "H"]
    negative_scores = model.decision_function([text_of(row, input_mode) for row in calibration_h])
    threshold = threshold_at_fpr(negative_scores, max_fpr)
    predictions, result = score_test(model, test, threshold, input_mode)
    result.update({
        "train_providers": sorted(train_providers),
        "test_providers": sorted(test_providers),
        "train_h": int(counts[0]),
        "train_e": int(counts[1]),
        "calibration_h": len(calibration_h),
        "calibration_fpr": float(np.mean(negative_scores > threshold)),
        "threshold": threshold,
        "target_fpr": max_fpr,
        "model": "TF-IDF character 2-4 grams + balanced logistic regression",
        "input": input_mode,
    })
    return predictions, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research_benchmark_legal_edit_v1"))
    parser.add_argument("--input-mode", choices=("abstract", "title_abstract"), default="abstract")
    args = parser.parse_args()
    audit = json.loads((args.root / "evaluation/provenance_audit.json").read_text(encoding="utf-8"))
    if audit["error_count"]:
        raise SystemExit("Provenance audit has errors; baseline is blocked.")
    rows = read_jsonl(args.root / "evaluation/benchmark_records.jsonl")
    providers = {row["provider"] for row in rows if row["corpus_id"] == "E"}
    results = {}
    all_predictions = []
    for held_out in [None, *sorted(providers)]:
        name = "pooled" if held_out is None else f"leave_out_{held_out}"
        train_providers = providers if held_out is None else providers - {held_out}
        test_providers = providers if held_out is None else {held_out}
        predictions, result = run_condition(rows, train_providers, test_providers, 0.05, args.input_mode)
        results[name] = result
        for prediction in predictions:
            prediction["experiment"] = name
        all_predictions.extend(predictions)
    output = args.root / "evaluation"
    suffix = "" if args.input_mode == "abstract" else "_title_abstract"
    (output / f"char_ngram_baseline{suffix}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / f"char_ngram_test_predictions{suffix}.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in all_predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    summary = {name: {"actual_fpr": item["actual_fpr"], "test_h": item["test_h"],
                      "test_e": item["test_e"], "mean_tpr": float(np.mean([
                          score["tpr_at_calibrated_threshold"] for score in item["by_provider_level"].values()]))}
               for name, item in results.items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
