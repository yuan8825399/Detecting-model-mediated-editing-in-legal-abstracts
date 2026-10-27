"""Analyze level-wise detection and within-source score response on v3."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


PROVIDERS = ("deepseek", "doubao", "kimi", "wenxin")
LEVELS = (1, 2, 3, 4)
MODES = {"abstract": "", "title_abstract": "_title_abstract"}
SLOPE_WEIGHTS = np.array((-1.5, -0.5, 0.5, 1.5)) / 5


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def test_matrix(predictions: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    pooled = [row for row in predictions if row["experiment"] == "pooled"]
    by_source: defaultdict[str, list[dict]] = defaultdict(list)
    for row in pooled:
        by_source[row["source_id"]].append(row)
    sources = sorted(by_source)
    if not sources or len(pooled) != len(sources) * 17:
        raise ValueError("Pooled predictions do not contain 17 rows per source")
    h = np.empty(len(sources))
    e = np.empty((len(sources), len(PROVIDERS), len(LEVELS)))
    detected = np.empty_like(e, dtype=bool)
    for i, source_id in enumerate(sources):
        rows = by_source[source_id]
        originals = [row for row in rows if row["corpus_id"] == "H"]
        edits = [row for row in rows if row["corpus_id"] == "E"]
        if len(originals) != 1 or len(edits) != 16:
            raise ValueError(f"Incomplete source group: {source_id}")
        h[i] = originals[0]["score"]
        keyed = {(row["provider"], int(row["intervention_level"])): row for row in edits}
        if len(keyed) != 16 or set(keyed) != {(provider, level)
                                             for provider in PROVIDERS for level in LEVELS}:
            raise ValueError(f"Missing or duplicate provider/level: {source_id}")
        for j, provider in enumerate(PROVIDERS):
            for k, level in enumerate(LEVELS):
                row = keyed[(provider, level)]
                e[i, j, k] = row["score"]
                detected[i, j, k] = row["detected"]
    return sources, h, e, detected


def mean_spearman(scores: np.ndarray) -> tuple[float | None, int]:
    flat = scores.reshape((-1, 4))
    coefficients = []
    for row in flat:
        ranks = rankdata(row)
        centered = ranks - ranks.mean()
        norm = np.linalg.norm(centered)
        if norm:
            coefficients.append(float(np.dot(centered, np.arange(4) - 1.5) /
                                      (norm * np.sqrt(5))))
    return (float(np.mean(coefficients)) if coefficients else None,
            len(flat) - len(coefficients))


def permutation_pvalue(scores: np.ndarray, rng: np.random.Generator,
                       permutations: int) -> float:
    flat = scores.reshape((-1, 4))
    observed = float(np.mean(flat @ SLOPE_WEIGHTS))
    extreme = 0
    for _ in range(permutations):
        order = np.argsort(rng.random(flat.shape), axis=1)
        permuted = np.take_along_axis(flat, order, axis=1)
        extreme += float(np.mean(permuted @ SLOPE_WEIGHTS)) >= observed - 1e-12
    return (extreme + 1) / (permutations + 1)


def response(scores: np.ndarray, h: np.ndarray, bootstrap_indices: np.ndarray,
             rng: np.random.Generator, permutations: int) -> dict:
    # Score dimensions are source x provider x level; H is shared within a source.
    differences = np.diff(scores, axis=2)
    strict = np.all(differences > 0, axis=2)
    h_to_e1 = scores[:, :, 0] - h[:, None]
    strict_with_h = strict & (h_to_e1 > 0)
    slopes = scores @ SLOPE_WEIGHTS
    rho, undefined = mean_spearman(scores)
    boot_slopes = np.mean(slopes[bootstrap_indices], axis=(1, 2))
    boot_strict = np.mean(strict[bootstrap_indices], axis=(1, 2))
    adjacent = {}
    for idx, name in enumerate(("E1_to_E2", "E2_to_E3", "E3_to_E4")):
        delta = differences[:, :, idx]
        adjacent[name] = {
            "mean_score_change": float(delta.mean()),
            "mean_score_change_95ci": interval(np.mean(delta[bootstrap_indices], axis=(1, 2))),
            "decrease_fraction": float(np.mean(delta < 0)),
            "tie_fraction": float(np.mean(delta == 0)),
            "nonincrease_fraction": float(np.mean(delta <= 0)),
        }
    return {
        "sequences": int(strict.size),
        "strict_E1_to_E4_count": int(strict.sum()),
        "strict_E1_to_E4_fraction": float(strict.mean()),
        "strict_E1_to_E4_95ci": interval(boot_strict),
        "non_strict_fraction": float(1 - strict.mean()),
        "strict_H_to_E4_count": int(strict_with_h.sum()),
        "strict_H_to_E4_fraction": float(strict_with_h.mean()),
        "H_to_E1_nonincrease_fraction": float(np.mean(h_to_e1 <= 0)),
        "mean_within_sequence_spearman": rho,
        "undefined_spearman_count": undefined,
        "mean_level_slope": float(slopes.mean()),
        "mean_level_slope_95ci": interval(boot_slopes),
        "one_sided_permutation_p_for_positive_slope": permutation_pvalue(scores, rng, permutations),
        "adjacent": adjacent,
    }


def analyze_mode(root: Path, mode: str, bootstrap: int, permutations: int) -> dict:
    suffix = MODES[mode]
    baseline = json.loads((root / "evaluation" / f"char_ngram_baseline{suffix}.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(root / "evaluation" / f"char_ngram_test_predictions{suffix}.jsonl")
    sources, h, e, detected = test_matrix(predictions)
    pooled_baseline = baseline["pooled"]
    if len(sources) != pooled_baseline["test_h"] or e.size != pooled_baseline["test_e"]:
        raise ValueError("Prediction counts disagree with baseline results")
    threshold = float(pooled_baseline["threshold"])
    if not np.array_equal(detected, e > threshold):
        raise ValueError("Saved predictions disagree with the calibrated threshold")
    test_fpr = float(np.mean(h > threshold))
    if not np.isclose(test_fpr, pooled_baseline["actual_fpr"]):
        raise ValueError("Saved test H scores disagree with baseline FPR")
    rng = np.random.default_rng(20261016)
    boot_idx = rng.integers(0, len(sources), size=(bootstrap, len(sources)))
    levels = {}
    for k, level in enumerate(LEVELS):
        positive = e[:, :, k].reshape(-1)
        labels = np.r_[np.zeros(len(h)), np.ones(len(positive))]
        actual_auroc = float(roc_auc_score(labels, np.r_[h, positive]))
        tpr = float(detected[:, :, k].mean())
        tpr_ci = interval(np.mean(detected[boot_idx, :, k], axis=(1, 2)))
        auroc_boot = []
        for indices in boot_idx:
            negative = h[indices]
            positive_boot = e[indices, :, k].reshape(-1)
            auroc_boot.append(roc_auc_score(
                np.r_[np.zeros(len(negative)), np.ones(len(positive_boot))],
                np.r_[negative, positive_boot]))
        levels[f"E{level}"] = {
            "test_h": len(h), "test_e": len(positive),
            "auroc": actual_auroc, "auroc_95ci": interval(np.asarray(auroc_boot)),
            "recall_at_calibrated_5pct_fpr": tpr,
            "recall_95ci": tpr_ci,
        }
    by_provider_level = {}
    for j, provider in enumerate(PROVIDERS):
        for k, level in enumerate(LEVELS):
            metric = pooled_baseline["by_provider_level"][f"{provider}:E{level}"]
            if (not np.isclose(metric["auroc"], roc_auc_score(
                    np.r_[np.zeros(len(h)), np.ones(len(h))], np.r_[h, e[:, j, k]])) or
                    not np.isclose(metric["tpr_at_calibrated_threshold"], detected[:, j, k].mean())):
                raise ValueError(f"Provider-level baseline mismatch: {provider}:E{level}")
            by_provider_level[f"{provider}:E{level}"] = {
                "test_h": len(h), "test_e": len(h),
                "auroc": float(metric["auroc"]),
                "recall_at_calibrated_5pct_fpr": float(metric["tpr_at_calibrated_threshold"]),
            }
    return {
        "input_mode": mode,
        "test_sources": len(sources),
        "calibration_h": int(pooled_baseline["calibration_h"]),
        "target_calibration_fpr": float(pooled_baseline["target_fpr"]),
        "achieved_calibration_fpr": float(pooled_baseline["calibration_fpr"]),
        "test_fpr_at_calibrated_threshold": test_fpr,
        "threshold": threshold,
        "levels": levels,
        "by_provider_level": by_provider_level,
        "response_pooled": response(e, h, boot_idx, rng, permutations),
        "response_by_provider": {
            provider: response(e[:, j:j+1, :], h, boot_idx, rng, permutations)
            for j, provider in enumerate(PROVIDERS)
        },
    }


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def make_report(results: dict) -> str:
    lines = [
        "# CLAW-Edit v3: 分级检出与分数响应", "",
        "对象：v3 正式测试分区的 86 个来源组。每组 1 条 H，以及四个模型各 1 条 E1-E4。",
        "使用已训练的字符 2-4 gram TF-IDF + 逻辑回归基线；下表来自同一 pooled 模型，",
        "逐级评估时没有重新训练或用测试集选阈值。", "",
        "阈值仅由 calibration 分区的 H 分数设定，目标误报率 5%；测试集实际误报率单独报告。",
        "AUROC 不依赖阈值。95% 区间按来源组重采样 1000 次；一个来源组的 H 与 16 个 E 一起重采样。",
        "", "## 分级检出", "",
    ]
    for mode, result in results["modes"].items():
        label = "摘要" if mode == "abstract" else "标题 + 摘要"
        lines += [f"### {label}", "",
                  f"校准 H={result['calibration_h']}；校准误报率 "
                  f"{percent(result['achieved_calibration_fpr'])}，测试 H={result['test_sources']}；"
                  f"测试误报率 {percent(result['test_fpr_at_calibrated_threshold'])}。", "",
                  "| 对比 | H / E | AUROC [95% CI] | 召回率 [95% CI] |",
                  "| --- | ---: | ---: | ---: |"]
        for level, metric in result["levels"].items():
            lines.append(
                f"| H vs {level} | {metric['test_h']} / {metric['test_e']} | "
                f"{metric['auroc']:.3f} [{metric['auroc_95ci'][0]:.3f}, "
                f"{metric['auroc_95ci'][1]:.3f}] | "
                f"{percent(metric['recall_at_calibrated_5pct_fpr'])} "
                f"[{percent(metric['recall_95ci'][0])}, {percent(metric['recall_95ci'][1])}] |"
            )
        lines += ["", "按生成模型分层（每格 H=86、E=86；使用同一阈值）：", "",
                  "| 模型 | E1 AUROC / 召回 | E2 AUROC / 召回 | E3 AUROC / 召回 | E4 AUROC / 召回 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for provider in PROVIDERS:
            cells = []
            for level in LEVELS:
                metric = result["by_provider_level"][f"{provider}:E{level}"]
                cells.append(f"{metric['auroc']:.3f} / "
                             f"{percent(metric['recall_at_calibrated_5pct_fpr'])}")
            lines.append(f"| {provider} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## 同源分数响应", "",
              "严格递增定义为 E1 < E2 < E3 < E4；任一相邻分数下降或持平都不符合。",
              "另报告 H < E1 < E2 < E3 < E4 的更严格比例。",
              "平均斜率是每条 E1-E4 序列的线性斜率（分数/等级）；单侧置换检验在每条序列内随机置换等级，",
              "原假设为等级与分数无关联。斜率为正不意味着每条序列单调。", "",
              "| 输入 | 模型 | 序列数 | E1-E4 严格递增 | 不符合 | H-E4 严格递增 | 平均 Spearman | 平均斜率 [95% CI] | 单侧 p |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode, result in results["modes"].items():
        label = "摘要" if mode == "abstract" else "标题 + 摘要"
        for provider, metric in (("全部", result["response_pooled"]),
                                 *((p, result["response_by_provider"][p]) for p in PROVIDERS)):
            rho = metric["mean_within_sequence_spearman"]
            lines.append(
                f"| {label} | {provider} | {metric['sequences']} | "
                f"{percent(metric['strict_E1_to_E4_fraction'])} | "
                f"{percent(metric['non_strict_fraction'])} | "
                f"{percent(metric['strict_H_to_E4_fraction'])} | "
                f"{rho:.3f} | {metric['mean_level_slope']:.3f} "
                f"[{metric['mean_level_slope_95ci'][0]:.3f}, "
                f"{metric['mean_level_slope_95ci'][1]:.3f}] | "
                f"{metric['one_sided_permutation_p_for_positive_slope']:.4f} |"
            )
    lines += ["", "相邻等级违背（下降 / 持平；非递增 = 两者之和）：", "",
              "| 输入 | 模型 | E1→E2 | E2→E3 | E3→E4 |",
              "| --- | --- | ---: | ---: | ---: |"]
    for mode, result in results["modes"].items():
        label = "摘要" if mode == "abstract" else "标题 + 摘要"
        for provider, metric in (("全部", result["response_pooled"]),
                                 *((p, result["response_by_provider"][p]) for p in PROVIDERS)):
            entries = [f"{percent(metric['adjacent'][key]['decrease_fraction'])} / "
                       f"{percent(metric['adjacent'][key]['tie_fraction'])}"
                       for key in ("E1_to_E2", "E2_to_E3", "E3_to_E4")]
            lines.append(f"| {label} | {provider} | " + " | ".join(entries) + " |")
    lines += ["", "全部模型的相邻平均分数变化（来源组 bootstrap 95% CI）：", "",
              "| 输入 | E1→E2 | E2→E3 | E3→E4 |",
              "| --- | ---: | ---: | ---: |"]
    for mode, result in results["modes"].items():
        label = "摘要" if mode == "abstract" else "标题 + 摘要"
        adjacent = result["response_pooled"]["adjacent"]
        cells = []
        for key in ("E1_to_E2", "E2_to_E3", "E3_to_E4"):
            metric = adjacent[key]
            cells.append(f"{metric['mean_score_change']:.3f} "
                         f"[{metric['mean_score_change_95ci'][0]:.3f}, "
                         f"{metric['mean_score_change_95ci'][1]:.3f}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## 解释边界", "",
              "测试集实际误报率高于目标 5%，因此上述召回率只能称为“在校准集 5% 目标所选阈值下的测试召回率”，",
              "不能称为“测试集 FPR=5% 时的召回率”。E1-E4 是预设编辑操作，",
              "不是实际人工贡献度或 AI 参与度的真值。v3 的删组决定发生在查看 v2 后，",
              "本分析是清理后敏感性分析，不是独立确认性验证。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research_benchmark_legal_edit_v3"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap < 1 or args.permutations < 1:
        raise ValueError("Bootstrap and permutation counts must be positive")
    audit = json.loads((args.root / "evaluation/provenance_audit.json").read_text(encoding="utf-8"))
    if audit["error_count"]:
        raise ValueError("Benchmark provenance audit has errors")
    results = {
        "benchmark_version": "3.0",
        "bootstrap_replicates": args.bootstrap,
        "permutation_replicates": args.permutations,
        "bootstrap_unit": "source_id",
        "modes": {mode: analyze_mode(args.root, mode, args.bootstrap, args.permutations)
                  for mode in MODES},
    }
    output = args.root / "evaluation"
    (output / "level_detection_and_score_response.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "level_detection_and_score_response.md").write_text(make_report(results), encoding="utf-8")
    print(json.dumps({mode: {
        "test_fpr": result["test_fpr_at_calibrated_threshold"],
        "levels": {level: {"auroc": metrics["auroc"],
                           "recall": metrics["recall_at_calibrated_5pct_fpr"]}
                   for level, metrics in result["levels"].items()},
        "strict_monotonic_fraction": result["response_pooled"]["strict_E1_to_E4_fraction"],
        "slope_p": result["response_pooled"]["one_sided_permutation_p_for_positive_slope"],
    } for mode, result in results["modes"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
