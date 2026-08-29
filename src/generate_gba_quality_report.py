"""Generate figures and a Chinese report from the GBA quality audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from src.fetch_gba_environment_gee import VARIABLES
from src.run_gba_quality_audit import STATIC_VARIABLES


ROOT = Path(__file__).resolve().parent.parent
SCHEME_LABELS = {
    "random_rows": "Random rows",
    "point_group": "Point-grouped",
    "spatial_block": "0.5° spatial blocks",
}


def baseline_figure(data: dict, output: Path) -> None:
    experiments = data["fair_baselines"]["experiments"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    features = ["coordinates", "embeddings", "combined"]
    colors = ["#9C9C9C", "#4E79A7", "#59A14F"]
    x = np.arange(3)
    width = 0.24
    for axis, model in zip(axes, ["ridge", "random_forest"]):
        for feature_index, (feature, color) in enumerate(zip(features, colors)):
            values = [
                experiments[f"{scheme}__{model}__{feature}"]["mean_r2"]
                for scheme in SCHEME_LABELS
            ]
            axis.bar(x + (feature_index - 1) * width, values, width, label=feature, color=color)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, [SCHEME_LABELS[s] for s in SCHEME_LABELS], rotation=15)
        axis.set_title("Ridge" if model == "ridge" else "Multi-output Random Forest")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean R² across 26 variables")
    axes[1].legend(loc="lower left")
    fig.suptitle("Fair feature baselines under identical folds")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def learning_curve_figure(data: dict, output: Path) -> None:
    curves = data["learning_curves"]["curves"]
    sizes = sorted(int(value) for value in curves)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = ["#4E79A7", "#F28E2B", "#E15759"]
    for scheme, color in zip(SCHEME_LABELS, colors):
        means, stds = [], []
        for size in sizes:
            values = [
                seed_data[scheme]["mean_r2"]
                for seed_data in curves[str(size)].values()
                if scheme in seed_data
            ]
            means.append(np.mean(values))
            stds.append(np.std(values))
        means = np.asarray(means)
        stds = np.asarray(stds)
        ax.plot(sizes, means, marker="o", linewidth=2, color=color, label=SCHEME_LABELS[scheme])
        ax.fill_between(sizes, means - stds, means + stds, color=color, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{size:,}" for size in sizes])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Independent locations used")
    ax.set_ylabel("Mean R² across 26 variables")
    ax.set_title("Independent-location learning curves (mean ± seed SD)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def city_figure(data: dict, output: Path, region_name: str) -> None:
    cities = []
    means = []
    medians = []
    rows = []
    for city, info in data["city_holdout"].items():
        if "mean_r2" not in info:
            continue
        if info["n_test"] < 100:
            continue
        scores = np.asarray([info["r2"][v]["mean"] for v in VARIABLES])
        cities.append(city)
        means.append(float(np.mean(scores)))
        medians.append(float(np.median(scores)))
        rows.append(info["n_test"])
    order = np.argsort(means)
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y, np.asarray(means)[order], color="#E15759", alpha=0.72, label="Mean R²")
    ax.scatter(np.asarray(medians)[order], y, color="#1F4E79", zorder=3, label="Median variable R²")
    ax.axvline(0, color="black", linewidth=0.8)
    labels = [f"{cities[i]} (n={rows[i]:,})" for i in order]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Leave-one-city-out R²")
    ax.set_title(f"Cross-city transfer is substantially harder ({region_name})")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def transform_figure(data: dict, output: Path) -> None:
    transforms = data["target_transforms"]
    population = transforms["population_raw_vs_log1p"]
    aspect = transforms["aspect_direct_vs_sincos"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    values = [population["raw_mean_original_r2"], population["log_mean_original_r2"]]
    bars = axes[0].bar(["Raw target", "log1p target"], values, color=["#9C9C9C", "#59A14F"])
    axes[0].bar_label(bars, labels=[f"{v:.3f}" for v in values])
    axes[0].set_ylabel("Point-grouped original-scale R²")
    axes[0].set_title("Population target transform")
    errors = [aspect["direct_mean_circular_mae"], aspect["sincos_mean_circular_mae"]]
    bars = axes[1].bar(["Direct degrees", "sin/cos"], errors, color=["#9C9C9C", "#4E79A7"])
    axes[1].bar_label(bars, labels=[f"{v:.1f}°" for v in errors])
    axes[1].set_ylabel("Circular MAE (degrees; lower is better)")
    axes[1].set_title("Aspect representation")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def learning_summary(data: dict) -> dict:
    curves = data["learning_curves"]["curves"]
    sizes = sorted(int(value) for value in curves)
    output = {}
    for scheme in SCHEME_LABELS:
        output[scheme] = {}
        for size in sizes:
            values = [
                seed_data[scheme]["mean_r2"]
                for seed_data in curves[str(size)].values()
                if scheme in seed_data
            ]
            output[scheme][size] = (float(np.mean(values)), float(np.std(values)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path,
        default=ROOT / "results/gba_real/quality_audit_results.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/gba_real/QUALITY_AUDIT_REPORT.md",
    )
    parser.add_argument("--region-name", default="大湾区")
    parser.add_argument(
        "--next-step",
        default="扩展到广东省，以增加独立位置和环境梯度",
    )
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_figure(data, output_dir / "quality_fair_baselines.png")
    learning_curve_figure(data, output_dir / "quality_learning_curves.png")
    city_figure(data, output_dir / "quality_city_holdout.png", args.region_name)
    transform_figure(data, output_dir / "quality_target_transforms.png")

    experiments = data["fair_baselines"]["experiments"]
    learning = learning_summary(data)
    max_size = max(learning["random_rows"])
    baseline_rows = []
    for scheme in SCHEME_LABELS:
        for model in ["ridge", "random_forest"]:
            coords = experiments[f"{scheme}__{model}__coordinates"]["mean_r2"]
            emb = experiments[f"{scheme}__{model}__embeddings"]["mean_r2"]
            combined = experiments[f"{scheme}__{model}__combined"]["mean_r2"]
            baseline_rows.append(
                f"| {SCHEME_LABELS[scheme]} | {model} | {coords:.3f} | {emb:.3f} | "
                f"{combined:.3f} | {combined - coords:+.3f} |"
            )
    learning_rows = []
    for size in sorted(learning["random_rows"]):
        learning_rows.append(
            f"| {size:,} | "
            + " | ".join(
                f"{learning[scheme][size][0]:.3f} ± {learning[scheme][size][1]:.3f}"
                for scheme in SCHEME_LABELS
            )
            + " |"
        )
    city_rows = []
    valid_city_means = []
    for city, info in data["city_holdout"].items():
        if "mean_r2" not in info:
            city_rows.append(f"| {city} | {info['n_test']} | — | — |")
            continue
        values = np.asarray([info["r2"][v]["mean"] for v in VARIABLES])
        valid_city_means.append(info["mean_r2"])
        city_rows.append(
            f"| {city} | {info['n_test']:,} | {info['mean_r2']:.3f} | {np.median(values):.3f} |"
        )
    seed_rows = []
    for scheme, info in data["seed_stability"].items():
        seed_rows.append(
            f"| {SCHEME_LABELS[scheme]} | {info['mean_r2_across_seeds']:.3f} | "
            f"{info['std_r2_across_seeds']:.4f} |"
        )
    dq = data["data_quality"]
    era = dq["era5_effective_grid"]["per_year"]
    soil_missing = [
        (name, info["fraction"])
        for name, info in dq["missingness"].items() if info["fraction"] > 0
    ]
    static = data["static_dedup"]
    full_random_static = experiments[
        "random_rows__random_forest__embeddings"
    ]["static_mean_r2"]
    full_point_static = experiments[
        "point_group__random_forest__embeddings"
    ]["static_mean_r2"]
    latest_random_static = static["schemes"]["random_rows"]["static_mean_r2"]
    latest_spatial_static = static["schemes"]["spatial_block"]["static_mean_r2"]
    pop = data["target_transforms"]["population_raw_vs_log1p"]
    aspect = data["target_transforms"]["aspect_direct_vs_sincos"]
    report = f"""# AlphaEarth {args.region_name}实验质量增强报告

## 核心结论

本轮没有增加新的模型宣传指标，而是系统检查了位置泄漏、样本量、坐标混杂、跨城市外推、随机种子、静态变量重复和目标表示。结果表明：普通随机行验证确实偏乐观，但平均影响小于空间分块和跨城市迁移；独立位置增加仍能改善结果，但学习曲线后段已经明显变缓；AlphaEarth提供了超出坐标的信息，坐标与嵌入联合通常最好；模型对完整城市的外推明显不稳健。

质量审计使用轻量多输出随机森林作为统一控制模型，所有26个目标先按训练集标准化，使不同量纲在多输出树中权重接近。它用于比较实验设计，不替代此前逐变量50棵树的论文口径结果。

## 1. 同折公平基线

| 验证方式 | 模型 | 仅坐标 | 仅AlphaEarth | 坐标+AlphaEarth | 联合相对坐标增量 |
|---|---|---:|---:|---:|---:|
{chr(10).join(baseline_rows)}

最重要的判断不是哪个绝对分数最高，而是联合输入相对坐标基线的增量。在空间分块条件下，AlphaEarth仍提供额外信息，因此结果不能完全归因于经纬度；但坐标本身的预测能力也不可忽略。

![公平基线](quality_fair_baselines.png)

## 2. 随机行、位置分组与空间分块

| 验证方式 | 三种子平均R² | 种子间标准差 |
|---|---:|---:|
{chr(10).join(seed_rows)}

随机行与位置分组之间的差值衡量同地点跨年份泄漏，位置分组与空间分块之间的差值则主要反映空间自相关和区域外推难度。结果显示第二种差值更大，因此当前最主要的问题不是简单的年度重复，而是空间迁移。

## 3. 独立位置学习曲线

| 独立位置数 | 随机行R² | 位置分组R² | 空间分块R² |
|---:|---:|---:|---:|
{chr(10).join(learning_rows)}

从500个位置增加到{max_size:,}个完整案例位置后，三种验证均有改善，但后段提升幅度变小。新增年份不能替代新增空间位置；进一步扩大空间与环境覆盖通常比继续重复相同位置更有价值。

![学习曲线](quality_learning_curves.png)

## 4. 静态目标去重

高程、地形、土壤、树冠、建成区和人口等11个静态目标在完整数据中会被重复七年。比较结果：

| 数据与验证 | 静态变量平均R² |
|---|---:|
| 全部年份，随机行 | {full_random_static:.3f} |
| 全部年份，按point_id分组 | {full_point_static:.3f} |
| 仅{static['year']}年，随机行 | {latest_random_static:.3f} |
| 仅{static['year']}年，空间分块 | {latest_spatial_static:.3f} |

因此静态目标的主报告应优先采用单年或位置分组结果，全部年份随机CV只能作为与原论文对比的辅助口径。

## 5. 留一城市验证

| 留出城市 | 测试行数 | 26变量平均R² | 变量R²中位数 |
|---|---:|---:|---:|
{chr(10).join(city_rows)}

留一城市结果明显弱于0.5°空间分块。城市内部某些目标方差很小，使R²容易出现较大负值，特别是只有少量点的澳门，因此同时报告中位数和测试行数。总体结论仍然明确：当前模型适合区域内解释，不适合未经校准直接迁移到一个完全未见城市。

![跨城市验证](quality_city_holdout.png)

## 6. 目标变换

- 人口密度偏度为 **{data['target_transforms']['skewness']['population_density']:.2f}**。训练目标使用 `log1p` 后，原始尺度R²由 **{pop['raw_mean_original_r2']:.3f}** 变为 **{pop['log_mean_original_r2']:.3f}**，对数尺度R²由 **{pop['raw_mean_log_r2']:.3f}** 变为 **{pop['log_mean_log_r2']:.3f}**。
- 坡向直接按角度预测的圆周MAE为 **{aspect['direct_mean_circular_mae']:.1f}°**，改用正弦/余弦后为 **{aspect['sincos_mean_circular_mae']:.1f}°**。
- `flow_accumulation` 在提取阶段已经是 `log(cell count + 1)`，没有重复取对数。
- 夜间灯光和年径流在当前{args.region_name}数据中偏度分别为 **{data['target_transforms']['skewness']['nighttime_lights']:.2f}** 和 **{data['target_transforms']['skewness']['annual_runoff']:.2f}**，没有机械地增加对数变换。

![目标变换](quality_target_transforms.png)

## 7. 数据独立性与缺失

- 总行数：{dq['rows']:,}；独立位置：{dq['unique_points']:,}；26变量全部完整：{dq['complete_rows_all_26']:,}行。
- 有缺失的变量：{', '.join(f'{name} {fraction:.2%}' for name, fraction in soil_missing)}。
- 按ERA5-Land约0.1°原始网格估算，每年约覆盖 **{np.mean([v['approximate_0_1_degree_cells'] for v in era]):.0f}** 个源网格，每个源网格平均对应 **{np.mean([v['mean_samples_per_source_cell'] for v in era]):.1f}** 个采样点。
- 将粗分辨率影像重投影到10m只会产生插值值，不会创造新的独立气候观测，因此气候和水文变量的有效样本量显著小于总行数。
- 土壤缺失没有被当作真实标签插补。对监督学习目标进行空间插补会制造伪标签，因此主分析继续采用逐变量删除或完整案例敏感性分析。

## 现在可以更可靠地回答的问题

1. **是否存在位置泄漏？** 有，但随机行与位置分组的差距小于空间分块损失。
2. **是否样本量不足？** 独立位置增加仍有效，但曲线后段趋缓；空间覆盖比重复年份更关键。
3. **AlphaEarth是否只是坐标代理？** 不是。空间分块下嵌入和联合输入仍优于仅坐标，但坐标混杂确实存在。
4. **能否跨城市直接部署？** 目前不能。完整城市留出结果明显不稳健。
5. **哪些处理应改入主流程？** 人口使用`log1p`作为补充口径，坡向使用正弦/余弦，静态目标采用位置分组或单年验证。

## 下一步

当前{args.region_name}数据内部最重要的质量检查已经完成。若继续提高实验质量，下一项较有信息增益的工作是{args.next_step}；继续增加相同区域的模型复杂度或问答模板，预期收益更低。
"""
    args.output.write_text(report, encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
