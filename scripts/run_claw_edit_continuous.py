"""Train and test source-group-separated continuous z prediction for CLAW-Edit."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge


Z_BY_LEVEL = {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}
ALPHA_GRID = (0.1, 1.0, 10.0)
BOOTSTRAP_REPLICATES = 1000


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def z_of(row: dict) -> float:
    level = int(row["intervention_level"])
    if row["corpus_id"] == "H" and level != 0:
        raise ValueError(f"H record has nonzero intervention level: {row['record_id']}")
    if row["corpus_id"] == "E" and level not in (1, 2, 3, 4):
        raise ValueError(f"E record has invalid intervention level: {row['record_id']}")
    return Z_BY_LEVEL[level]


def text_of(row: dict, input_mode: str) -> str:
    if input_mode == "title_abstract":
        return row["source_title_clean"] + "\n" + row["abstract_clean"]
    return row["abstract_clean"]


def clip(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def metric_dict(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    difference = prediction - target
    has_variation = bool(np.std(target) and np.std(prediction))
    pearson = float(np.corrcoef(target, prediction)[0, 1]) if has_variation else None
    spearman = spearmanr(target, prediction).statistic if has_variation else None
    return {
        "n": int(len(target)),
        "mae": round(float(np.mean(np.abs(difference))), 6),
        "rmse": round(float(np.sqrt(np.mean(np.square(difference)))), 6),
        "bias": round(float(np.mean(difference)), 6),
        "pearson_r": round(pearson, 6) if pearson is not None and np.isfinite(pearson) else None,
        "spearman_rho": round(float(spearman), 6) if spearman is not None and np.isfinite(spearman) else None,
    }


def bootstrap_source_ci(rows: list[dict], prediction_key: str, replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    """Bootstrap aggregate continuous-prediction metrics at source-group level."""
    by_source: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
    source_ids = sorted(by_source)
    if any(len(by_source[source_id]) != 17 for source_id in source_ids):
        raise ValueError("Bootstrap requires complete source groups")
    rng = np.random.default_rng(20261018)
    samples = [rng.choice(source_ids, size=len(source_ids), replace=True) for _ in range(replicates)]
    results: defaultdict[str, list[float]] = defaultdict(list)
    for sample in samples:
        sampled = [row for source_id in sample for row in by_source[source_id]]
        metrics = metric_dict(np.array([row["z"] for row in sampled]),
                              np.array([row[prediction_key] for row in sampled]))
        for key in ("mae", "rmse", "bias", "pearson_r", "spearman_rho"):
            if metrics[key] is not None:
                results[key].append(metrics[key])
    return {
        f"{key}_95ci": [round(float(np.percentile(values, 2.5)), 6),
                             round(float(np.percentile(values, 97.5)), 6)]
        for key, values in sorted(results.items())
    }


def group_metrics(rows: list[dict], prediction_key: str) -> dict:
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = "H" if row["corpus_id"] == "H" else f"{row['provider']}:E{row['intervention_level']}"
        grouped[key].append(row)
    return {
        key: metric_dict(np.array([item["z"] for item in values]),
                         np.array([item[prediction_key] for item in values]))
        for key, values in sorted(grouped.items())
    }


def balanced_weights(labels: np.ndarray) -> np.ndarray:
    counts = Counter(labels.tolist())
    return np.array([len(labels) / (len(counts) * counts[value]) for value in labels], dtype=float)


def assert_group_isolation(splits: dict[str, list[dict]]) -> dict[str, int]:
    groups = {name: {row["group_id"] for row in rows} for name, rows in splits.items()}
    names = sorted(groups)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if groups[left] & groups[right]:
                raise ValueError(f"Source-group leakage between {left} and {right}")
    for name, rows in splits.items():
        counts = Counter(row["source_id"] for row in rows)
        if set(counts.values()) != {17}:
            raise ValueError(f"Incomplete source group in {name}")
    return {name: len(value) for name, value in groups.items()}


def run(root: Path, input_mode: str) -> dict:
    records = read_jsonl(root / "evaluation/benchmark_records.jsonl")
    records = [{**record, "z": z_of(record), "z_label_definition": "operation_intensity_v1"}
               for record in records]
    if len(records) != len({row["record_id"] for row in records}):
        raise ValueError("Duplicate record IDs")
    splits = {name: [row for row in records if row["group_split"] == name]
              for name in ("train", "dev", "calibration", "test")}
    source_counts = assert_group_isolation(splits)
    train, dev, calibration, test = (splits[name] for name in ("train", "dev", "calibration", "test"))

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3,
                                 max_features=100000, sublinear_tf=True)
    train_x = vectorizer.fit_transform([text_of(row, input_mode) for row in train])
    dev_x = vectorizer.transform([text_of(row, input_mode) for row in dev])
    calibration_x = vectorizer.transform([text_of(row, input_mode) for row in calibration])
    test_x = vectorizer.transform([text_of(row, input_mode) for row in test])
    train_y = np.array([row["z"] for row in train])
    dev_y = np.array([row["z"] for row in dev])
    calibration_y = np.array([row["z"] for row in calibration])
    test_y = np.array([row["z"] for row in test])
    weights = balanced_weights(train_y)

    dev_results = {}
    candidates: dict[float, Ridge] = {}
    for alpha in ALPHA_GRID:
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(train_x, train_y, sample_weight=weights)
        candidates[alpha] = model
        dev_results[str(alpha)] = metric_dict(dev_y, clip(model.predict(dev_x)))
    selected_alpha = min(ALPHA_GRID, key=lambda value: dev_results[str(value)]["mae"])
    model = candidates[selected_alpha]
    calibration_raw = clip(model.predict(calibration_x))
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(calibration_raw, calibration_y, sample_weight=balanced_weights(calibration_y))
    test_raw = clip(model.predict(test_x))
    test_calibrated = clip(calibrator.predict(test_raw))

    predictions = []
    for row, raw, calibrated in zip(test, test_raw, test_calibrated):
        predictions.append({
            "record_id": row["record_id"], "source_id": row["source_id"],
            "group_id": row["group_id"], "group_split": row["group_split"],
            "corpus_id": row["corpus_id"], "provider": row["provider"],
            "intervention_level": row["intervention_level"], "z": row["z"],
            "z_prediction_raw": round(float(raw), 6),
            "z_prediction_calibrated": round(float(calibrated), 6),
        })
    output = root / "evaluation"
    analysis_records = [{key: value for key, value in row.items() if key != "abstract_raw"} for row in records]
    if any("<正>" in json.dumps(row, ensure_ascii=False) for row in analysis_records):
        raise ValueError("A continuous-label analysis record contains <正>")
    write_jsonl(output / "continuous_z_records.jsonl", analysis_records)
    write_jsonl(output / f"continuous_z_test_predictions_{input_mode}.jsonl", predictions)
    result = {
        "task": "continuous LLM intervention operation-intensity prediction",
        "label": {
            "name": "z", "range": [0.0, 1.0],
            "mapping": {"H": 0.0, "E1": 0.25, "E2": 0.50, "E3": 0.75, "E4": 1.0},
            "interpretation": "Predefined edit-operation intensity proxy; not a measured proportion of LLM authorship.",
            "relationship": "E1-E4 are parallel edits from the same H source, not a chained trajectory.",
        },
        "model": "TF-IDF character 2-4 grams plus class-balanced ridge regression",
        "input": input_mode,
        "split_protocol": "train fit; dev alpha selection; calibration isotonic mapping; test final evaluation; source groups are disjoint.",
        "source_groups_by_split": source_counts,
        "records_by_split": {name: len(rows) for name, rows in splits.items()},
        "selected_alpha": selected_alpha,
        "dev_by_alpha_raw_clipped": dev_results,
        "calibration_raw_clipped": metric_dict(calibration_y, calibration_raw),
        "calibration_isotonic": metric_dict(calibration_y, clip(calibrator.predict(calibration_raw))),
        "test_raw_clipped": metric_dict(test_y, test_raw),
        "test_isotonic_calibrated": metric_dict(test_y, test_calibrated),
        "test_isotonic_calibrated_source_bootstrap": bootstrap_source_ci(
            predictions, "z_prediction_calibrated"
        ),
        "test_by_provider_level_raw_clipped": group_metrics(predictions, "z_prediction_raw"),
        "test_by_provider_level_isotonic_calibrated": group_metrics(predictions, "z_prediction_calibrated"),
    }
    (output / f"continuous_z_regression_{input_mode}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research_benchmark_legal_edit_v3"))
    parser.add_argument("--input-mode", choices=("abstract", "title_abstract"), default="abstract")
    args = parser.parse_args()
    result = run(args.root, args.input_mode)
    print(json.dumps({
        "selected_alpha": result["selected_alpha"],
        "test_raw_clipped": result["test_raw_clipped"],
        "test_isotonic_calibrated": result["test_isotonic_calibrated"],
        "source_groups_by_split": result["source_groups_by_split"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
