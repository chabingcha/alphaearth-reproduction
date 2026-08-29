"""Generate figures and a Chinese report for the four extended GBA modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.fetch_gba_environment_gee import VARIABLES


ROOT = Path(__file__).resolve().parent.parent


def rf_transformer_figure(analysis: dict, transformer: dict, output: Path) -> None:
    rf = analysis["random_forest"]["per_variable"]
    tf = transformer["per_variable"]
    ordered = sorted(
        VARIABLES, key=lambda name: rf[name]["random_cv_r2_mean"]
    )
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.scatter(
        [rf[name]["random_cv_r2_mean"] for name in ordered], y,
        color="#2878B5", marker="o", s=42, label="RF random CV", zorder=3,
    )
    ax.scatter(
        [rf[name]["spatial_cv_r2_mean"] for name in ordered], y,
        facecolor="white", edgecolor="#2878B5", marker="o", s=42,
        label="RF spatial CV", zorder=3,
    )
    ax.scatter(
        [tf[name]["random_cv_r2_mean"] for name in ordered], y,
        color="#F28E2B", marker="s", s=38, label="Transformer random CV", zorder=3,
    )
    ax.scatter(
        [tf[name]["spatial_cv_r2_mean"] for name in ordered], y,
        facecolor="white", edgecolor="#F28E2B", marker="s", s=38,
        label="Transformer spatial CV", zorder=3,
    )
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(y, labels=ordered)
    ax.set_xlabel("Cross-validated R²")
    ax.set_title("GBA: Random Forest vs compact multi-task Transformer")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def dictionary_figure(dictionary: dict, output: Path) -> None:
    summary = dictionary["summary"]
    correlations = summary["method_matrix_correlations"]
    labels = ["Spearman", "RF", "Transformer"]
    matrix = np.eye(3)
    matrix[0, 1] = matrix[1, 0] = correlations["spearman_vs_rf"]
    matrix[0, 2] = matrix[2, 0] = correlations["spearman_vs_transformer"]
    matrix[1, 2] = matrix[2, 1] = correlations["rf_vs_transformer"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    counts = [
        64 - summary["dimensions_two_or_more_methods_agree"],
        summary["dimensions_two_or_more_methods_agree"]
        - summary["dimensions_all_three_methods_agree"],
        summary["dimensions_all_three_methods_agree"],
        summary["robust_dimensions"],
    ]
    bars = axes[0].bar(
        ["<2 agree", "Exactly 2", "All 3", "Robust"],
        counts,
        color=["#B8B8B8", "#59A14F", "#2878B5", "#9C6ADE"],
    )
    axes[0].bar_label(bars)
    axes[0].set_ylim(0, 64)
    axes[0].set_ylabel("Embedding dimensions")
    axes[0].set_title("Three-method dictionary")
    image = axes[1].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    axes[1].set_xticks(range(3), labels=labels, rotation=20)
    axes[1].set_yticks(range(3), labels=labels)
    axes[1].set_title("Importance-matrix correlation")
    for row in range(3):
        for column in range(3):
            axes[1].text(
                column, row, f"{matrix[row, column]:.3f}",
                ha="center", va="center",
                color="white" if matrix[row, column] > 0.55 else "black",
            )
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluation_figure(evaluation: dict, output: Path) -> None:
    criteria = [
        "grounding", "coherence", "scientific_accuracy", "completeness",
        "practical_utility",
    ]
    labels = ["Grounding", "Coherence", "Accuracy", "Completeness", "Utility"]
    values = [evaluation["scores"][key]["mean"] for key in criteria]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bars = axes[0].bar(labels, values, color="#4E79A7")
    axes[0].bar_label(bars, labels=[f"{value:.2f}" for value in values])
    axes[0].set_ylim(0, 5.4)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Automated evidence-check score (1–5)")
    axes[0].set_title("360 deterministic QA checks")
    retrieval = evaluation["retrieval_benchmark"]
    axes[1].axis("off")
    text = (
        "FAISS retrieval benchmark\n\n"
        f"Recall@10: {retrieval['exact_recall_at_10_mean']:.1%}\n"
        f"Raw search mean: {retrieval['raw_faiss_search_latency_ms_mean']:.3f} ms\n"
        f"Raw search p95: {retrieval['raw_faiss_search_latency_ms_p95']:.3f} ms\n"
        f"Mean analog distance: {retrieval['filtered_analog_distance_km_mean']:.1f} km\n"
        f"Cross-city analogs: {retrieval['filtered_analogs_cross_city_fraction']:.1%}\n\n"
        "Scores at left are not LLM-as-Judge scores.\n"
        "They verify generated claims against source rows."
    )
    axes[1].text(
        0.05, 0.95, text, va="top", fontsize=12,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F4F6F8", "edgecolor": "#AAB2BD"},
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis", type=Path,
        default=ROOT / "results/gba_real/analysis_results_paper_method.json",
    )
    parser.add_argument(
        "--transformer", type=Path,
        default=ROOT / "results/gba_real/transformer_results.json",
    )
    parser.add_argument(
        "--dictionary", type=Path,
        default=ROOT / "results/gba_real/dimension_dictionary.json",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "results/gba_real/lsi/manifest.json",
    )
    parser.add_argument(
        "--evaluation", type=Path,
        default=ROOT / "results/gba_real/lsi/evaluation/evaluation_summary.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/gba_real/EXTENDED_REPORT.md",
    )
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    transformer = json.loads(args.transformer.read_text(encoding="utf-8"))
    dictionary = json.loads(args.dictionary.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rf_transformer_figure(
        analysis, transformer, output_dir / "transformer_rf_comparison.png"
    )
    dictionary_figure(dictionary, output_dir / "method_dictionary_summary.png")
    evaluation_figure(evaluation, output_dir / "lsi_evaluation_summary.png")

    rf_summary = analysis["random_forest"]["summary"]
    tf_summary = transformer["summary"]
    dict_summary = dictionary["summary"]
    robust = [
        (name, info) for name, info in dictionary["dimensions"].items() if info["robust"]
    ]
    robust.sort(key=lambda item: item[1]["combined_support"], reverse=True)
    tf_top = sorted(
        transformer["per_variable"].items(),
        key=lambda item: item[1]["random_cv_r2_mean"], reverse=True,
    )[:10]
    retrieval = evaluation["retrieval_benchmark"]
    robust_rows = "\n".join(
        f"| {name} | {info['primary_variable']} | {info['spearman_rho']:.3f} | "
        f"{info['agreement_count']} | {info['temporal_stability_r']:.3f} |"
        for name, info in robust[:10]
    )
    tf_rows = "\n".join(
        f"| {name} | {info['random_cv_r2_mean']:.3f} | "
        f"{info['spatial_cv_r2_mean']:.3f} | {info['delta_r2']:.3f} |"
        for name, info in tf_top
    )
    report = f"""# AlphaEarth 大湾区扩展复现：四个后续模块

## 总结

四个后续模块已经形成可重复运行的实现：紧凑型多任务 Transformer、Spearman/RF/Transformer 三方法维度字典、54,614 向量的 FAISS/RAG 系统，以及 360 次真实位置问答的证据一致性评测。真正的四模型 LLM-as-Judge 调用器也已实现，但由于当前环境没有模型 API 凭据，尚未产生四模型评分；当前 **4.92/5** 仅代表确定性证据检查，不能与原论文的 **3.74/5** 直接比较。

## 1. 多任务 Transformer

- 数据：严格口径大湾区真实数据，每折从 54,614 行中固定抽样 30,000 行。
- 模型：64 个带维度标识的数值 token，2 层、4 头、`d_model=32`，一次预测 26 个目标。
- 验证：5 折随机 CV 与 0.5° 空间分块 CV。
- 重要性：在留出集逐维置换，而不是把注意力权重直接解释为因果重要性。
- 运行设备：Apple MPS；正式训练耗时 {tf_summary['runtime_seconds'] / 60:.1f} 分钟。

| 指标 | 原论文 | 大湾区 RF | 大湾区 Transformer |
|---|---:|---:|---:|
| 平均随机 CV R² | 未统一报告 | {rf_summary['mean_random_cv_r2']:.3f} | {tf_summary['mean_random_cv_r2']:.3f} |
| 平均空间 CV R² | 未统一报告 | {rf_summary['mean_random_cv_r2'] - rf_summary['mean_spatial_delta_r2']:.3f} | {tf_summary['mean_spatial_cv_r2']:.3f} |
| 平均空间 ΔR² | Transformer 0.017 | {rf_summary['mean_spatial_delta_r2']:.3f} | {tf_summary['mean_spatial_delta_r2']:.3f} |
| 随机 R² > 0.7 | — | {rf_summary['variables_r2_gt_0_7']}/26 | {tf_summary['variables_random_r2_gt_0_7']}/26 |

Transformer 的平均随机 R² 为 **{tf_summary['mean_random_cv_r2']:.3f}**，低于 RF 的 **{rf_summary['mean_random_cv_r2']:.3f}**；平均空间下降 **{tf_summary['mean_spatial_delta_r2']:.3f}**，与 RF 的 **{rf_summary['mean_spatial_delta_r2']:.3f}** 接近。这是缩小数据与模型下的真实结果，不支持“Transformer 必然优于随机森林”。

### Transformer表现最好的变量

| 变量 | 随机R² | 空间R² | ΔR² |
|---|---:|---:|---:|
{tf_rows}

![RF与Transformer比较](transformer_rf_comparison.png)

## 2. 三方法维度字典

- 至少两种方法一致：**{dict_summary['dimensions_two_or_more_methods_agree']}/64**。
- 三种方法完全一致：**{dict_summary['dimensions_all_three_methods_agree']}/64**。
- 同时满足时间稳定性 `r >= 0.90` 的稳健维度：**{dict_summary['robust_dimensions']}/64**。
- 重要性矩阵相关：Spearman–RF **{dict_summary['method_matrix_correlations']['spearman_vs_rf']:.3f}**，Spearman–Transformer **{dict_summary['method_matrix_correlations']['spearman_vs_transformer']:.3f}**，RF–Transformer **{dict_summary['method_matrix_correlations']['rf_vs_transformer']:.3f}**。

| 维度 | 主要变量 | Spearman ρ | 一致方法数 | 时间稳定性 |
|---|---|---:|---:|---:|
{robust_rows}

![三方法字典](method_dictionary_summary.png)

完整字典同时提供 JSON、按维度 CSV 和按变量 CSV。无两方法一致的维度仍会保留，但只标记为探索性解释。

## 3. FAISS 与 RAG 原型

- 索引：`{manifest['index_type']}`，{manifest['n_vectors']:,} 个 64 维年度向量。
- 参数：`nlist={manifest['nlist']}`，`nprobe={manifest['nprobe']}`，余弦相似度。
- 检索时排除同一 `point_id`，防止同地点不同年份制造虚假的相似结果。
- 36 个评测位置上 Recall@10 为 **{retrieval['exact_recall_at_10_mean']:.1%}**，原始 FAISS 搜索平均 **{retrieval['raw_faiss_search_latency_ms_mean']:.3f} ms**。
- 返回相似地点平均距离 **{retrieval['filtered_analog_distance_km_mean']:.1f} km**，其中 **{retrieval['filtered_analogs_cross_city_fraction']:.1%}** 来自其他城市。

RAG回答包含：位置解析、真实环境画像、区域百分位、相关AlphaEarth维度、相似地点及方法限制。当前采用确定性证据渲染，优点是每个数值都可追溯；它不是开放式大语言模型生成。

## 4. 360次问答评测

- 36 个分层抽取的真实位置 × 10 类意图 = **{evaluation['cycles']}** 次。
- 数值证据回查准确率：**{evaluation['numeric_accuracy_fraction']:.1%}**。
- 完整RAG回答平均延迟：**{evaluation['latency_ms']['mean']:.2f} ms**。
- 确定性证据检查总体分：**{evaluation['scores']['overall']['mean']:.2f}/5**。

该评分回答的是“系统有没有忠实呈现检索到的数据”，没有回答“开放式LLM的语言、推理和建议是否达到专家水平”。因此它不能与论文四模型 LLM-as-Judge 的 3.74 分作优劣比较。

![LSI评测](lsi_evaluation_summary.png)

## 与原论文仍不完全相同的地方

1. Transformer 训练数据为 30,000 行、2层4头；原论文约500万行、4层8头、60轮。
2. 大湾区空间验证使用0.5°块；论文在CONUS使用2°块。
3. PRISM和NLCD不覆盖中国，仍由ERA5-Land和WorldCover替代。
4. 当前执行的是确定性证据评测；四模型轮换评审脚本已准备，但没有API凭据，所以未调用外部模型。
5. 没有人工环境领域专家评分。该项也是原论文承认的限制，可作为后续增强。

## 主要文件

- `transformer_results.json`：Transformer逐变量、逐折结果及64×26置换重要性。
- `gba_transformer.pt`：第一随机折最佳模型与标准化参数。
- `dimension_dictionary.json/csv`：三方法维度知识库。
- `lsi/manifest.json`、`lsi/gba_aef_ivfflat.faiss`：检索索引与参数。
- `lsi/demo_answers.md`：三类实际问答示例。
- `lsi/evaluation/evaluation_summary.json`：360次证据评测汇总。
- `lsi/evaluation/answers.jsonl`：全部问题、答案、证据与检索结果。
- `src/run_gba_llm_judge.py`：可恢复、四模型轮换的真正LLM评审接口。
"""
    args.output.write_text(report, encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
