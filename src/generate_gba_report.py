"""Generate tables, figures, and a Chinese report for the final GBA run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.run_gba_real_analysis import CATEGORIES, ROOT


RESULTS_DIR = ROOT / "results" / "gba_real"
ANALYSIS_PATH = RESULTS_DIR / "analysis_results_paper_method.json"
SENSITIVITY_PATH = RESULTS_DIR / "spatial_sensitivity_paper_method.json"

VARIABLE_LABELS = {
    "elevation": "Elevation",
    "slope": "Slope",
    "aspect": "Aspect",
    "flow_accumulation": "Flow accumulation",
    "clay_fraction": "Clay fraction",
    "organic_carbon": "Organic carbon",
    "soil_ph": "Soil pH",
    "water_capacity": "Water capacity",
    "ndvi_mean": "NDVI mean",
    "ndvi_max": "NDVI max",
    "evi_mean": "EVI mean",
    "lai_mean": "LAI mean",
    "tree_cover": "Tree cover",
    "albedo": "Albedo",
    "lst_daytime": "LST daytime",
    "lst_nighttime": "LST nighttime",
    "air_temp_mean": "Air temperature",
    "dew_point_temp": "Dew point",
    "annual_precip": "Annual precipitation",
    "max_monthly_precip": "Max monthly precip.",
    "soil_moisture": "Soil moisture",
    "annual_runoff": "Annual runoff",
    "annual_et": "Annual ET",
    "impervious_surface": "Built-up fraction",
    "nighttime_lights": "Nighttime lights",
    "population_density": "Population density",
}


def category_for(variable: str) -> str:
    for category, variables in CATEGORIES.items():
        if variable in variables:
            return category
    return "other"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    sensitivity = json.loads(SENSITIVITY_PATH.read_text(encoding="utf-8"))

    rf = pd.DataFrame(result["random_forest"]["per_variable"]).T
    rf.index.name = "variable"
    rf = rf.reset_index()
    rf.insert(1, "label", rf["variable"].map(VARIABLE_LABELS))
    rf.insert(2, "category", rf["variable"].map(category_for))
    score_columns = [
        "variable", "label", "category", "n_samples",
        "random_cv_r2_mean", "random_cv_r2_std",
        "spatial_cv_r2_mean", "spatial_cv_r2_std", "delta_r2",
        "top3_dimensions",
    ]
    rf[score_columns].to_csv(RESULTS_DIR / "rf_scores_paper_method.csv", index=False)

    ordered = rf.sort_values("random_cv_r2_mean")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.barh(y - 0.18, ordered["random_cv_r2_mean"], height=0.34,
            color="#2878B5", label="Random 5-fold CV")
    ax.barh(y + 0.18, ordered["spatial_cv_r2_mean"], height=0.34,
            color="#E07A5F", label="0.5° spatial block CV")
    ax.set_yticks(y, ordered["label"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(min(-0.1, ordered["spatial_cv_r2_mean"].min() - 0.05), 1.0)
    ax.set_xlabel("R²")
    ax.set_title("GBA AlphaEarth decoding: random vs spatial validation")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rf_random_vs_spatial.png", dpi=180)
    plt.close(fig)

    rho = np.asarray(result["spearman"]["rho_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(14, 10))
    image = ax.imshow(rho, aspect="auto", cmap="RdBu_r", vmin=-0.9, vmax=0.9)
    ax.set_xticks(np.arange(len(VARIABLE_LABELS)))
    ax.set_xticklabels(VARIABLE_LABELS.values(), rotation=65, ha="right", fontsize=8)
    ax.set_yticks(np.arange(0, 64, 4))
    ax.set_yticklabels([f"A{value:02d}" for value in range(0, 64, 4)])
    ax.set_xlabel("Environmental variable")
    ax.set_ylabel("AlphaEarth dimension")
    ax.set_title("Spearman correlation matrix (64 dimensions × 26 variables)")
    fig.colorbar(image, ax=ax, label="Spearman ρ", shrink=0.8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "spearman_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variable, values in sensitivity["results"].items():
        sizes = sorted(float(value) for value in values)
        means = [values[str(size)]["mean_r2"] for size in sizes]
        ax.plot(sizes, means, marker="o", linewidth=2,
                label=VARIABLE_LABELS[variable])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0.25, 0.5, 1.0])
    ax.set_xlabel("Spatial block size (degrees)")
    ax.set_ylabel("Mean spatial CV R²")
    ax.set_title("Spatial generalization sensitivity to block size")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "spatial_block_sensitivity.png", dpi=180)
    plt.close(fig)

    summary = result["random_forest"]["summary"]
    temporal = result["temporal_stability"]
    abs_rho = np.abs(rho)
    dim_max = np.nanmax(abs_rho, axis=1)
    best = result["spearman"]["best_dimension_per_variable"]
    strongest = sorted(
        ((variable, value["dimension"], value["rho"]) for variable, value in best.items()),
        key=lambda item: abs(item[2]),
        reverse=True,
    )[:8]
    strongest_rows = "\n".join(
        f"| {VARIABLE_LABELS[variable]} | {dimension} | {rho_value:.3f} |"
        for variable, dimension, rho_value in strongest
    )

    top_rf = rf.sort_values("random_cv_r2_mean", ascending=False).head(10)
    top_rf_rows = "\n".join(
        f"| {row.label} | {row.random_cv_r2_mean:.3f} | "
        f"{row.spatial_cv_r2_mean:.3f} | {row.delta_r2:.3f} |"
        for row in top_rf.itertuples()
    )

    report = f"""# AlphaEarth 大湾区真实数据复现报告

## 结论

大湾区真实数据支持“AlphaEarth 嵌入包含可解释的地表信息”，但不支持把原论文在美国本土得到的强预测力和极小空间泛化损失直接外推到大湾区。严格口径主结果中，随机森林平均随机交叉验证 R² 为 **{summary['mean_random_cv_r2']:.3f}**，仅 **{summary['variables_r2_gt_0_9']}/26** 个变量超过 0.9、**{summary['variables_r2_gt_0_7']}/26** 个超过 0.7；前十变量平均空间降幅为 **{summary['top10_mean_spatial_delta_r2']:.3f}**。时间稳定性为 **{temporal['mean_inter_year_r']:.3f}**，Spearman 与随机森林重要性的收敛度为 **{result['method_convergence_r']:.3f}**。

## 数据与单条观测面积

- 区域：粤港澳大湾区 9 个广东城市加香港、澳门，GADM 边界面积约 55,470 km²。
- 时间：2017–2023，共 7 年。
- 网格：0.025°，每年 7,802 个有效位置，共 54,614 行；大湾区纬度下相邻网格约覆盖 7.05–7.20 km²。
- AlphaEarth：官方 Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`，64 维；按论文方法以 1 km scale、500 m 半径圆形缓冲区提取。单条嵌入观测的缓冲面积为 **0.7854 km²**。
- 原生 AlphaEarth 像元：10 m × 10 m，即 100 m²。单个官方 COG 文件为 81.92 km × 81.92 km，即约 6,710.89 km²；文件覆盖面积不是单条观测面积。
- 环境变量：26 项。各连续数据源先以双线性方式统一到 30 m 投影，再按论文标注的 10 m extraction scale 取点值；53,235/54,614 行（97.48%）26 项完整。四项土壤变量各缺失 2.42%，主要是近岸覆盖空缺；未插补。

## 与原论文的关键对比

| 指标 | 原论文 CONUS | 大湾区严格复现 | 差异 |
|---|---:|---:|---:|
| 样本量 | 约 12.1M | 54,614 | -99.55% |
| Spearman 每维最大 |ρ| > 0.5 | 34/64 | {int(np.sum(dim_max > 0.5))}/64 | {int(np.sum(dim_max > 0.5)) - 34:+d} |
| Spearman 每维最大 |ρ| > 0.7 | 6/64 | {int(np.sum(dim_max > 0.7))}/64 | {int(np.sum(dim_max > 0.7)) - 6:+d} |
| RF R² > 0.9 | 12/26 | {summary['variables_r2_gt_0_9']}/26 | {summary['variables_r2_gt_0_9'] - 12:+d} |
| RF R² > 0.7 | 20/26 | {summary['variables_r2_gt_0_7']}/26 | {summary['variables_r2_gt_0_7'] - 20:+d} |
| RF 前十变量平均空间 ΔR² | 0.009 | {summary['top10_mean_spatial_delta_r2']:.3f} | {summary['top10_mean_spatial_delta_r2'] - 0.009:+.3f} |
| 平均时间稳定性 r | 0.963 | {temporal['mean_inter_year_r']:.3f} | {temporal['mean_inter_year_r'] - 0.963:+.3f} |
| 稳定性 r > 0.95 | 51/64 | {temporal['dimensions_r_gt_0_95']}/64 | {temporal['dimensions_r_gt_0_95'] - 51:+d} |
| 稳定性 r > 0.90 | 59/64 | {temporal['dimensions_r_gt_0_90']}/64 | {temporal['dimensions_r_gt_0_90'] - 59:+d} |
| |ρ| 与 RF 重要性相关 | 0.450 | {result['method_convergence_r']:.3f} | {result['method_convergence_r'] - 0.45:+.3f} |

原论文还报告 Transformer 全部 26 项平均空间 ΔR²=0.017；本次没有复现 Transformer 和后续 RAG/LLM 评测，因此不能把 RF 结果冒充 Transformer 对比。

## 最强单维 Spearman 关系

| 环境变量 | 嵌入维度 | ρ |
|---|---:|---:|
{strongest_rows}

## 随机森林前十变量

| 变量 | 随机 CV R² | 0.5° 空间 CV R² | ΔR² |
|---|---:|---:|---:|
{top_rf_rows}

## 空间尺度敏感性

0.5° 块在大湾区形成 37 个空间组，五折样本比例均约 20%。原论文的 2° 块在大湾区只形成 5 个严重不均衡的组，最小折仅 42 行，因此不适用。0.25°、0.5°、1° 敏感性结果显示：海拔、建成区和年降水在更大块下仍保持正 R²；NDVI、LAI、夜间灯光和人口密度随块增大显著下降，1° 时部分变量为负，说明这些关系主要是局地可迁移，而非区域外推稳健。

## 不能归因于“区域”的方法差异

1. 原论文研究区是美国本土，样本量约为本次的 222 倍。
2. PRISM 和 NLCD 不覆盖中国。本次以 ERA5-Land 替代 PRISM 的气温、露点和降水，以 ESA WorldCover 建成区比例替代 NLCD impervious；这些变量不是同源复现。
3. 原论文未完整公开 RF 超参数。本次为 50 棵树、`min_samples_leaf=2`、`max_features=1.0`，每变量最多 30,000 行、5 折。
4. 空间块从论文 2° 改为适合区域范围的 0.5°，并提供 0.25°/1° 敏感性。
5. 为避免粗分辨率数据在小区域统计时只命中像元中心，环境影像内部先统一到 30 m 投影，再以 10 m sampling scale 取点；这与论文仅写“resampled to 10 m”并非逐行代码级完全相同。
6. 未复现多任务 Transformer、FAISS/RAG 与 LLM-as-Judge；报告只对已真实重算的 Spearman、RF、空间验证和时间稳定性负责。

## 文件

- `analysis_results_paper_method.json`：最终完整数值。
- `rf_scores_paper_method.csv`：26 个变量逐项 RF 指标。
- `spatial_sensitivity_paper_method.json`：三种块尺度及每折得分。
- `rf_random_vs_spatial.png`：随机与空间 CV 对比。
- `spearman_heatmap.png`：64×26 Spearman 矩阵。
- `spatial_block_sensitivity.png`：块尺度敏感性。

## 数据来源

- AlphaEarth 官方目录与 Earth Engine：<https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL>
- AlphaEarth GCS/COG 文件结构：<https://developers.google.com/earth-engine/guides/aef_on_gcs_readme>
- ERA5-Land：<https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR>
- GADM 中国行政边界：<https://gadm.org/download_country.html>
- 原论文：Rahman (2026), arXiv:2602.10354。
"""
    (RESULTS_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Saved report and figures in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
