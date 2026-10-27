"""Ordinal-operation and robustness experiments for CLAW-Edit v3.

The five labels are experimental operation levels, not authorship proportions.
All development, calibration, and evaluation partitions are source-group
disjoint.  The cumulative model estimates P(level > k), k=0..3.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


LEVELS = np.arange(5)
Z = LEVELS / 4.0
PROVIDERS = ("deepseek", "doubao", "kimi", "wenxin")
C_GRID = (0.1, 1.0, 10.0)
BOOTSTRAP_REPLICATES = 1000
RNG_SEED = 20261019


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def level_of(row: dict) -> int:
    level = int(row["intervention_level"])
    if row["corpus_id"] == "H" and level != 0:
        raise ValueError(f"H record has a nonzero level: {row['record_id']}")
    if row["corpus_id"] == "E" and level not in (1, 2, 3, 4):
        raise ValueError(f"E record has an invalid level: {row['record_id']}")
    return level


def text_of(row: dict, input_mode: str) -> str:
    if input_mode == "title_abstract":
        return row["source_title_clean"] + "\n" + row["abstract_clean"]
    return row["abstract_clean"]


def balanced_weights(binary: np.ndarray) -> np.ndarray:
    counts = Counter(binary.tolist())
    if set(counts) != {0, 1}:
        raise ValueError("A cumulative threshold has only one class")
    return np.array([len(binary) / (2 * counts[value]) for value in binary], dtype=float)


def ensure_group_isolation(splits: dict[str, list[dict]], expected_group_size: int | None = None) -> dict:
    group_sets = {name: {row["group_id"] for row in rows} for name, rows in splits.items()}
    names = list(group_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise ValueError(f"Source-group leakage between {left} and {right}: {sorted(overlap)[:3]}")
    for name, rows in splits.items():
        counts = Counter(row["group_id"] for row in rows)
        if expected_group_size is not None and set(counts.values()) != {expected_group_size}:
            raise ValueError(f"Unexpected group size in {name}: {set(counts.values())}")
    return {name: len(groups) for name, groups in group_sets.items()}


def monotone_probabilities(cumulative: np.ndarray) -> np.ndarray:
    """Convert four P(Y>k) values to valid ordered five-class probabilities."""
    cumulative = np.clip(np.asarray(cumulative, dtype=float), 0.0, 1.0)
    cumulative = np.minimum.accumulate(cumulative, axis=1)
    probabilities = np.column_stack((1 - cumulative[:, 0],
                                     cumulative[:, 0] - cumulative[:, 1],
                                     cumulative[:, 1] - cumulative[:, 2],
                                     cumulative[:, 2] - cumulative[:, 3],
                                     cumulative[:, 3]))
    if not np.allclose(probabilities.sum(axis=1), 1.0) or np.any(probabilities < -1e-8):
        raise ValueError("Invalid ordinal probability construction")
    return np.clip(probabilities, 0.0, 1.0)


def metric_dict(y: np.ndarray, probabilities: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    prediction = np.argmax(probabilities, axis=1)
    expected_z = probabilities @ Z
    target_z = y / 4.0
    rho = spearmanr(y, expected_z).statistic if np.std(y) and np.std(expected_z) else None
    return {
        "n": int(len(y)),
        "exact_accuracy": round(float(accuracy_score(y, prediction)), 6),
        "macro_f1": round(float(f1_score(y, prediction, labels=LEVELS, average="macro", zero_division=0)), 6),
        "quadratic_weighted_kappa": round(float(cohen_kappa_score(y, prediction, labels=LEVELS, weights="quadratic")), 6),
        "mae_levels": round(float(np.mean(np.abs(y - prediction))), 6),
        "within_one_level": round(float(np.mean(np.abs(y - prediction) <= 1)), 6),
        "expected_z_mae": round(float(np.mean(np.abs(target_z - expected_z))), 6),
        "expected_z_rmse": round(float(np.sqrt(np.mean((target_z - expected_z) ** 2))), 6),
        "expected_z_spearman": round(float(rho), 6) if rho is not None and np.isfinite(rho) else None,
    }


def level_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    prediction = np.argmax(probabilities, axis=1)
    result = {}
    for level in LEVELS:
        mask = y == level
        result[f"L{level}"] = {
            "n": int(mask.sum()),
            "recall": round(float(np.mean(prediction[mask] == level)), 6),
            "mean_expected_z": round(float(np.mean((probabilities @ Z)[mask])), 6),
            "mean_prediction": round(float(np.mean(prediction[mask])), 6),
        }
    return result


class CumulativeOrdinalModel:
    def __init__(self, c: float):
        self.c = c
        self.models: list[LogisticRegression] = []
        self.calibrators: list[IsotonicRegression] = []

    def fit(self, matrix, y: np.ndarray) -> None:
        self.models = []
        for threshold in range(4):
            binary = (y > threshold).astype(int)
            model = LogisticRegression(C=self.c, max_iter=1000, solver="liblinear",
                                       random_state=RNG_SEED + threshold)
            model.fit(matrix, binary, sample_weight=balanced_weights(binary))
            self.models.append(model)

    def raw_cumulative(self, matrix) -> np.ndarray:
        return np.column_stack([model.predict_proba(matrix)[:, 1] for model in self.models])

    def calibrate(self, matrix, y: np.ndarray) -> None:
        raw = self.raw_cumulative(matrix)
        self.calibrators = []
        for threshold in range(4):
            binary = (y > threshold).astype(int)
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            calibrator.fit(raw[:, threshold], binary, sample_weight=balanced_weights(binary))
            self.calibrators.append(calibrator)

    def predict_proba(self, matrix, calibrated: bool = True) -> np.ndarray:
        cumulative = self.raw_cumulative(matrix)
        if calibrated:
            if len(self.calibrators) != 4:
                raise ValueError("Calibration is required before calibrated prediction")
            cumulative = np.column_stack([cal.predict(cumulative[:, index])
                                          for index, cal in enumerate(self.calibrators)])
        return monotone_probabilities(cumulative)


def fit_and_predict(splits: dict[str, list[dict]], input_mode: str) -> tuple[dict, list[dict]]:
    group_counts = ensure_group_isolation(splits)
    train, dev, calibration, test = (splits[key] for key in ("train", "dev", "calibration", "test"))
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3,
                                 max_features=100000, sublinear_tf=True)
    train_x = vectorizer.fit_transform([text_of(row, input_mode) for row in train])
    matrices = {
        "train": train_x,
        "dev": vectorizer.transform([text_of(row, input_mode) for row in dev]),
        "calibration": vectorizer.transform([text_of(row, input_mode) for row in calibration]),
        "test": vectorizer.transform([text_of(row, input_mode) for row in test]),
    }
    labels = {name: np.array([level_of(row) for row in rows])
              for name, rows in (("train", train), ("dev", dev), ("calibration", calibration), ("test", test))}
    dev_by_c = {}
    candidates = {}
    for c in C_GRID:
        model = CumulativeOrdinalModel(c)
        model.fit(matrices["train"], labels["train"])
        probabilities = model.predict_proba(matrices["dev"], calibrated=False)
        candidates[c] = model
        dev_by_c[str(c)] = metric_dict(labels["dev"], probabilities)
    selected_c = min(C_GRID, key=lambda c: (dev_by_c[str(c)]["mae_levels"], -dev_by_c[str(c)]["quadratic_weighted_kappa"]))
    model = candidates[selected_c]
    calibration_raw = model.predict_proba(matrices["calibration"], calibrated=False)
    model.calibrate(matrices["calibration"], labels["calibration"])
    calibration_calibrated = model.predict_proba(matrices["calibration"], calibrated=True)
    test_raw = model.predict_proba(matrices["test"], calibrated=False)
    test_calibrated = model.predict_proba(matrices["test"], calibrated=True)
    predictions = []
    for row, raw, calibrated in zip(test, test_raw, test_calibrated):
        predictions.append({
            "record_id": row["record_id"], "source_id": row["source_id"],
            "group_id": row["group_id"], "source_year": int(row["source_year"]),
            "corpus_id": row["corpus_id"], "provider": row["provider"],
            "intervention_level": level_of(row),
            "ordinal_prediction": int(np.argmax(calibrated)),
            "expected_z": round(float(calibrated @ Z), 6),
            "probabilities_raw": [round(float(value), 6) for value in raw],
            "probabilities_calibrated": [round(float(value), 6) for value in calibrated],
        })
    result = {
        "model": "Cumulative ordinal logistic regression over character 2-4-gram TF-IDF; independently isotonic-calibrated threshold probabilities are monotonized before five-class reconstruction.",
        "input": input_mode,
        "source_groups_by_split": group_counts,
        "records_by_split": {name: len(rows) for name, rows in splits.items()},
        "selected_c": selected_c,
        "dev_by_c_raw": dev_by_c,
        "calibration_raw": metric_dict(labels["calibration"], calibration_raw),
        "calibration_isotonic": metric_dict(labels["calibration"], calibration_calibrated),
        "test_raw": metric_dict(labels["test"], test_raw),
        "test_isotonic_calibrated": metric_dict(labels["test"], test_calibrated),
        "test_by_level_isotonic_calibrated": level_metrics(labels["test"], test_calibrated),
    }
    return result, predictions


def bootstrap_source_ci(predictions: list[dict], replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    by_source: defaultdict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        by_source[row["source_id"]].append(row)
    source_ids = sorted(by_source)
    sizes = {len(by_source[source]) for source in source_ids}
    if not source_ids or len(sizes) != 1:
        raise ValueError("Bootstrap requires equally complete source groups")
    rng = np.random.default_rng(RNG_SEED)
    values: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sample = rng.choice(source_ids, size=len(source_ids), replace=True)
        rows = [row for source in sample for row in by_source[source]]
        y = np.array([row["intervention_level"] for row in rows])
        p = np.array([row["probabilities_calibrated"] for row in rows])
        metrics = metric_dict(y, p)
        for key, value in metrics.items():
            if key != "n" and value is not None:
                values[key].append(value)
    return {f"{key}_95ci": [round(float(np.percentile(items, 2.5)), 6),
                                round(float(np.percentile(items, 97.5)), 6)]
            for key, items in sorted(values.items())}


def fixed_splits(rows: list[dict], providers: set[str] | None = None) -> dict[str, list[dict]]:
    return {
        split: [row for row in rows if row["group_split"] == split and
                (providers is None or row["corpus_id"] == "H" or row["provider"] in providers)]
        for split in ("train", "dev", "calibration", "test")
    }


def temporal_splits(rows: list[dict]) -> dict[str, list[dict]]:
    bounds = {
        "train": lambda year: year <= 2010,
        "dev": lambda year: 2011 <= year <= 2013,
        "calibration": lambda year: 2014 <= year <= 2016,
        "test": lambda year: year >= 2017,
    }
    return {name: [row for row in rows if predicate(int(row["source_year"]))]
            for name, predicate in bounds.items()}


def add_length_strata(predictions: list[dict], train_rows: list[dict]) -> dict:
    source_length = {row["source_id"]: int(row["char_count"]) for row in train_rows if row["corpus_id"] == "H"}
    cut_one, cut_two = np.quantile(np.array(list(source_length.values())), [1 / 3, 2 / 3])
    by_source: defaultdict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        by_source[row["source_id"]].append(row)
    results = {}
    for name, predicate in {
        "short": lambda length: length <= cut_one,
        "medium": lambda length: cut_one < length <= cut_two,
        "long": lambda length: length > cut_two,
    }.items():
        subset = [row for source, records in by_source.items()
                  for row in records if predicate(next(item["source_length"] for item in records if item["corpus_id"] == "H"))]
        # Source length is carried below; this assertion detects malformed groups.
        if not subset:
            raise ValueError(f"Empty length subset: {name}")
        y = np.array([row["intervention_level"] for row in subset])
        p = np.array([row["probabilities_calibrated"] for row in subset])
        results[name] = {
            "source_groups": len({row["source_id"] for row in subset}),
            "records": len(subset),
            "metric": metric_dict(y, p),
            "source_bootstrap_95ci": bootstrap_source_ci(subset),
        }
    return {
        "source_length_reference": "H abstract char_count; cut points are calculated on the fixed training partition only.",
        "cut_points": {"short_at_or_below": round(float(cut_one), 3), "medium_at_or_below": round(float(cut_two), 3)},
        "strata": results,
    }


def attach_source_length(predictions: list[dict], rows: list[dict]) -> list[dict]:
    source_lengths = {row["source_id"]: int(row["char_count"]) for row in rows if row["corpus_id"] == "H"}
    return [{**row, "source_length": source_lengths[row["source_id"]]} for row in predictions]


def brief(metric: dict) -> str:
    return (f"accuracy={metric['exact_accuracy']:.3f}, macro-F1={metric['macro_f1']:.3f}, "
            f"QWK={metric['quadratic_weighted_kappa']:.3f}, MAE-level={metric['mae_levels']:.3f}")


def report_markdown(result: dict) -> str:
    lines = ["# CLAW-Edit v3: 有序操作等级与稳健性实验", "",
             "标签为预设操作等级 H/E1/E2/E3/E4（0--4），不是实际 LLM 作者贡献或文本生成比例。",
             "累计有序逻辑模型估计四个 P(level > k)，在校准集独立做 isotonic 校准后施加单调约束，",
             "再还原为五级概率。所有切分及 bootstrap 均以来源组为单位。", ""]
    for mode, item in result["fixed_split"].items():
        metric = item["test_isotonic_calibrated"]
        ci = item["source_bootstrap_95ci"]
        label = "摘要" if mode == "abstract" else "标题 + 摘要"
        lines += [f"## 固定来源组切分：{label}", "",
                  f"{brief(metric)}。", "",
                  "| Metric | Point estimate | Source-bootstrap 95% CI |",
                  "| --- | ---: | ---: |"]
        for key in ("exact_accuracy", "macro_f1", "quadratic_weighted_kappa", "mae_levels", "expected_z_mae"):
            interval = ci[f"{key}_95ci"]
            lines.append(f"| {key} | {metric[key]:.3f} | [{interval[0]:.3f}, {interval[1]:.3f}] |")
        lines.append("")
    lines += ["## 跨提供方外推（固定切分，摘要）", "",
              "训练、开发和校准均排除目标提供方；测试仅包含该提供方的 4 个编辑等级及共享 H。", "",
              "| Held-out provider | Test source groups | Accuracy | Macro-F1 | QWK | MAE-level |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for provider, item in result["cross_provider"].items():
        metric = item["test_isotonic_calibrated"]
        lines.append(f"| {provider} | {item['source_groups_by_split']['test']} | {metric['exact_accuracy']:.3f} | {metric['macro_f1']:.3f} | {metric['quadratic_weighted_kappa']:.3f} | {metric['mae_levels']:.3f} |")
    temporal = result["temporal"]
    metric = temporal["test_isotonic_calibrated"]
    lines += ["", "## 严格时间外推（摘要）", "",
              "训练年 2000--2010，开发年 2011--2013，校准年 2014--2016，测试年 2017--2020。",
              f"测试结果：{brief(metric)}。", "", "## 摘要长度分层（固定测试分区，摘要）", "",
              "长度以原始 H 摘要的字符数计；三分位切点只在固定训练分区的 H 中计算。", "",
              "| Stratum | Source groups | Accuracy | Macro-F1 | QWK | MAE-level |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in result["length_strata"]["strata"].items():
        item_metric = item["metric"]
        lines.append(f"| {name} | {item['source_groups']} | {item_metric['exact_accuracy']:.3f} | {item_metric['macro_f1']:.3f} | {item_metric['quadratic_weighted_kappa']:.3f} | {item_metric['mae_levels']:.3f} |")
    lines += ["", "## 边界", "",
              "当前基准没有人工核定的法学主题字段，故未把关键词或自动聚类结果误称为跨主题测试。",
              "标准中文预训练模型需要另有可追溯的预训练权重；项目内 Tiny-BERT 不能代替该基线。",
              "v3 在看到先前 v2 结果后进行质量筛选，因此全部结果仅构成质量筛选后的敏感性分析。", ""]
    return "\n".join(lines)


def run(root: Path) -> dict:
    rows = read_jsonl(root / "evaluation/benchmark_records.jsonl")
    if len(rows) != len({row["record_id"] for row in rows}):
        raise ValueError("Duplicate record IDs")
    if any("<正>" in row["abstract_clean"] for row in rows):
        raise ValueError("Analysis input contains <正>")
    result = {
        "benchmark_version": "3.0",
        "label": {
            "levels": {"H": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 4},
            "z_mapping": {"H": 0.0, "E1": 0.25, "E2": 0.5, "E3": 0.75, "E4": 1.0},
            "interpretation": "Pre-specified edit-operation labels, not LLM authorship or generated-text proportions.",
        },
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "source_id",
        "fixed_split": {},
        "cross_provider": {},
    }
    all_predictions = []
    for mode in ("abstract", "title_abstract"):
        fixed = fixed_splits(rows)
        item, predictions = fit_and_predict(fixed, mode)
        predictions = attach_source_length(predictions, rows)
        item["source_bootstrap_95ci"] = bootstrap_source_ci(predictions)
        result["fixed_split"][mode] = item
        all_predictions.extend([{**row, "experiment": f"fixed_{mode}"} for row in predictions])
        if mode == "abstract":
            result["length_strata"] = add_length_strata(predictions, fixed["train"])
    for provider in PROVIDERS:
        splits = fixed_splits(rows, set(PROVIDERS) - {provider})
        # The target provider appears only in test. Replace the test selection.
        splits["test"] = [row for row in rows if row["group_split"] == "test" and
                          (row["corpus_id"] == "H" or row["provider"] == provider)]
        item, predictions = fit_and_predict(splits, "abstract")
        item["source_bootstrap_95ci"] = bootstrap_source_ci(predictions)
        result["cross_provider"][provider] = item
        all_predictions.extend([{**row, "experiment": f"cross_provider_{provider}"} for row in predictions])
    temporal = temporal_splits(rows)
    temporal_item, temporal_predictions = fit_and_predict(temporal, "abstract")
    temporal_item["source_bootstrap_95ci"] = bootstrap_source_ci(temporal_predictions)
    temporal_item["year_protocol"] = {"train": "2000-2010", "dev": "2011-2013",
                                          "calibration": "2014-2016", "test": "2017-2020"}
    result["temporal"] = temporal_item
    all_predictions.extend([{**row, "experiment": "temporal_abstract"} for row in temporal_predictions])
    output = root / "evaluation"
    write_jsonl(output / "ordinal_robustness_test_predictions.jsonl", all_predictions)
    (output / "ordinal_robustness.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "ordinal_robustness.md").write_text(report_markdown(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research_benchmark_legal_edit_v3"))
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps({
        "fixed_split": {mode: item["test_isotonic_calibrated"] for mode, item in result["fixed_split"].items()},
        "cross_provider": {provider: item["test_isotonic_calibrated"] for provider, item in result["cross_provider"].items()},
        "temporal": result["temporal"]["test_isotonic_calibrated"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
