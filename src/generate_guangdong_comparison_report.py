"""Compare the original CONUS paper, GBA reproduction and Guangdong expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter

from src.fetch_gba_environment_gee import VARIABLES


ROOT = Path(__file__).resolve().parent.parent
PAPER = {
    "rows": 12_100_000,
    "spearman_gt_05": 34,
    "spearman_gt_07": 6,
    "rf_gt_09": 12,
    "rf_gt_07": 20,
    "rf_top10_delta": 0.009,
    "transformer_delta": 0.017,
    "temporal": 0.963,
    "convergence": 0.45,
}


def learning_values(audit: dict, scheme: str) -> tuple[list[int], list[float], list[float]]:
    curves = audit["learning_curves"]["curves"]
    sizes = sorted(int(value) for value in curves)
    means, stds = [], []
    for size in sizes:
        values = [
            seed[scheme]["mean_r2"]
            for seed in curves[str(size)].values()
            if scheme in seed
        ]
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))
    return sizes, means, stds


def rf_region_figure(gba: dict, gd: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, metric, title in [
        (axes[0], "random_cv_r2_mean", "Random-row CV"),
        (axes[1], "spatial_cv_r2_mean", "0.5° spatial CV"),
    ]:
        x = np.asarray([gba["random_forest"]["per_variable"][v][metric] for v in VARIABLES])
        y = np.asarray([gd["random_forest"]["per_variable"][v][metric] for v in VARIABLES])
        axis.scatter(x, y, color="#4E79A7", alpha=0.85)
        low = min(-0.2, float(np.min([x.min(), y.min()])))
        high = max(1.0, float(np.max([x.max(), y.max()])))
        axis.plot([low, high], [low, high], linestyle="--", color="#777777")
        axis.axhline(0, color="#BBBBBB", linewidth=0.7)
        axis.axvline(0, color="#BBBBBB", linewidth=0.7)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_xlabel("GBA R²")
        axis.set_ylabel("Guangdong R²")
        axis.set_title(title)
        axis.grid(alpha=0.15)
        for variable, xv, yv in zip(VARIABLES, x, y):
            if abs(yv - xv) > 0.25:
                axis.annotate(variable, (xv, yv), fontsize=7, xytext=(3, 3), textcoords="offset points")
    fig.suptitle("Per-variable RF performance: GBA vs Guangdong")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def learning_figure(gba: dict, gd: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, scheme, title in [
        (axes[0], "point_group", "Point-grouped holdout"),
        (axes[1], "spatial_block", "0.5° spatial-block holdout"),
    ]:
        for audit, label, color in [(gba, "GBA", "#F28E2B"), (gd, "Guangdong", "#4E79A7")]:
            sizes, means, stds = learning_values(audit, scheme)
            sizes_arr = np.asarray(sizes)
            means_arr = np.asarray(means)
            stds_arr = np.asarray(stds)
            axis.plot(sizes, means, marker="o", color=color, label=label)
            axis.fill_between(sizes_arr, means_arr - stds_arr, means_arr + stds_arr, color=color, alpha=0.15)
        axis.set_xscale("log")
        all_sizes = sorted(set(learning_values(gba, scheme)[0] + learning_values(gd, scheme)[0]))
        axis.set_xticks(all_sizes, [f"{value:,}" for value in all_sizes], rotation=25)
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_title(title)
        axis.set_xlabel("Independent locations")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean R² across 26 variables")
    axes[0].legend()
    fig.suptitle("Does broader spatial coverage improve generalization?")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def spatial_feature_figure(gba: dict, gd: dict, output: Path) -> None:
    features = ["coordinates", "embeddings", "combined"]
    labels = ["Coordinates", "AlphaEarth", "Combined"]
    x = np.arange(3)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.8))
    values = []
    for audit in [gba, gd]:
        experiments = audit["fair_baselines"]["experiments"]
        values.append([
            experiments[f"spatial_block__random_forest__{feature}"]["mean_r2"]
            for feature in features
        ])
    ax.bar(x - width / 2, values[0], width, label="GBA", color="#F28E2B")
    ax.bar(x + width / 2, values[1], width, label="Guangdong", color="#4E79A7")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean 0.5° spatial CV R²")
    ax.set_title("Spatial feature baselines at two regional extents")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def coverage_figure(boundary_path: Path, points_path: Path, output: Path) -> None:
    boundary = gpd.read_file(boundary_path).to_crs(4326)
    points = pd.read_parquet(points_path, columns=["lon", "lat"])
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    boundary.plot(ax=ax, facecolor="#EAF2F8", edgecolor="#456A86", linewidth=0.7)
    ax.scatter(
        points["lon"], points["lat"], s=0.35, color="#D64B3C", alpha=0.42,
        rasterized=True, label=f"0.025° samples (n={len(points):,}/year)",
    )
    labels = boundary.copy()
    labels["label_point"] = labels.representative_point()
    for _, row in labels.iterrows():
        ax.annotate(
            row["city"], (row["label_point"].x, row["label_point"].y),
            ha="center", va="center", fontsize=6.5, color="#263746",
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Guangdong Province sampling coverage (21 prefectures)")
    ax.legend(loc="lower left", markerscale=7, frameon=True)
    ax.set_aspect("equal")
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def metric_row(label: str, paper, gba, gd, formatter=".3f") -> str:
    def value(item):
        if item is None:
            return "—"
        if isinstance(item, int):
            return f"{item:,}"
        return format(item, formatter)
    return f"| {label} | {value(paper)} | {value(gba)} | {value(gd)} |"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gba-analysis", type=Path,
        default=ROOT / "results/gba_real/analysis_results_paper_method.json",
    )
    parser.add_argument(
        "--gd-analysis", type=Path,
        default=ROOT / "results/guangdong_real/analysis_results_paper_method.json",
    )
    parser.add_argument(
        "--gba-audit", type=Path,
        default=ROOT / "results/gba_real/quality_audit_results.json",
    )
    parser.add_argument(
        "--gd-audit", type=Path,
        default=ROOT / "results/guangdong_real/quality_audit_results.json",
    )
    parser.add_argument(
        "--gba-transformer", type=Path,
        default=ROOT / "results/gba_real/transformer_results.json",
    )
    parser.add_argument(
        "--gd-transformer", type=Path,
        default=ROOT / "results/guangdong_real/transformer_results.json",
    )
    parser.add_argument(
        "--region-summary", type=Path,
        default=ROOT / "data/processed/guangdong_sampling_points_0025deg.summary.json",
    )
    parser.add_argument(
        "--boundary", type=Path,
        default=ROOT / "data/raw/guangdong_boundary_gadm41.geojson",
    )
    parser.add_argument(
        "--points", type=Path,
        default=ROOT / "data/processed/guangdong_sampling_points_0025deg.parquet",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/guangdong_real/COMPARISON_REPORT.md",
    )
    args = parser.parse_args()

    gba = json.loads(args.gba_analysis.read_text(encoding="utf-8"))
    gd = json.loads(args.gd_analysis.read_text(encoding="utf-8"))
    gba_audit = json.loads(args.gba_audit.read_text(encoding="utf-8"))
    gd_audit = json.loads(args.gd_audit.read_text(encoding="utf-8"))
    gba_tf = json.loads(args.gba_transformer.read_text(encoding="utf-8"))
    gd_tf = json.loads(args.gd_transformer.read_text(encoding="utf-8"))
    region = json.loads(args.region_summary.read_text(encoding="utf-8"))
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_figure(args.boundary, args.points, output_dir / "guangdong_sampling_coverage.png")
    rf_region_figure(gba, gd, output_dir / "gba_vs_guangdong_rf.png")
    learning_figure(gba_audit, gd_audit, output_dir / "gba_vs_guangdong_learning.png")
    spatial_feature_figure(gba_audit, gd_audit, output_dir / "gba_vs_guangdong_spatial_features.png")

    gba_rf = gba["random_forest"]["summary"]
    gd_rf = gd["random_forest"]["summary"]
    rows = [
        metric_row("总行数", PAPER["rows"], gba["data"]["rows"], gd["data"]["rows"], ","),
        metric_row("独立位置", None, gba["data"]["locations"], gd["data"]["locations"], ","),
        metric_row("Spearman维度 |ρ|>0.5", PAPER["spearman_gt_05"], gba["spearman"]["dimensions_abs_rho_gt_0_5"], gd["spearman"]["dimensions_abs_rho_gt_0_5"], "d"),
        metric_row("Spearman维度 |ρ|>0.7", PAPER["spearman_gt_07"], gba["spearman"]["dimensions_abs_rho_gt_0_7"], gd["spearman"]["dimensions_abs_rho_gt_0_7"], "d"),
        metric_row("RF平均随机CV R²", None, gba_rf["mean_random_cv_r2"], gd_rf["mean_random_cv_r2"]),
        metric_row("RF变量R²>0.9", PAPER["rf_gt_09"], gba_rf["variables_r2_gt_0_9"], gd_rf["variables_r2_gt_0_9"], "d"),
        metric_row("RF变量R²>0.7", PAPER["rf_gt_07"], gba_rf["variables_r2_gt_0_7"], gd_rf["variables_r2_gt_0_7"], "d"),
        metric_row("RF前十平均空间ΔR²", PAPER["rf_top10_delta"], gba_rf["top10_mean_spatial_delta_r2"], gd_rf["top10_mean_spatial_delta_r2"]),
        metric_row("平均时间稳定性", PAPER["temporal"], gba["temporal_stability"]["mean_inter_year_r"], gd["temporal_stability"]["mean_inter_year_r"]),
        metric_row("Spearman-RF收敛", PAPER["convergence"], gba["method_convergence_r"], gd["method_convergence_r"]),
        metric_row("Transformer平均随机CV R²", None, gba_tf["summary"]["mean_random_cv_r2"], gd_tf["summary"]["mean_random_cv_r2"]),
        metric_row("Transformer平均空间ΔR²", PAPER["transformer_delta"], gba_tf["summary"]["mean_spatial_delta_r2"], gd_tf["summary"]["mean_spatial_delta_r2"]),
    ]
    gd_quality_spatial = gd_audit["seed_stability"]["spatial_block"]["mean_r2_across_seeds"]
    gba_quality_spatial = gba_audit["seed_stability"]["spatial_block"]["mean_r2_across_seeds"]
    report = f"""# AlphaEarth：大湾区、广东省与原论文对比

## 区域与设计

广东省扩展使用与大湾区严格复现相同的0.025°网格、2017–2023时段、1 km AlphaEarth提取尺度、500 m缓冲区和26个环境变量。广东省边界面积约 **{region['area_km2']:,.0f} km²**，覆盖21个地级市；香港和澳门不属于广东省，因此未包含。每年 **{region['points_per_year']:,}** 个独立位置，共 **{region['rows_2017_2023']:,}** 行。

![广东省采样覆盖](guangdong_sampling_coverage.png)

## 核心指标

| 指标 | 原论文CONUS | 大湾区 | 广东省 |
|---|---:|---:|---:|
{chr(10).join(rows)}

广东省与大湾区的正式RF都固定为每变量最多30,000行，因此其差异更接近“相同训练量、不同空间与环境覆盖”的对照；质量审计则使用各区域全部完整案例，反映增加独立位置后的实际收益。

## 逐变量RF变化

![逐变量RF比较](gba_vs_guangdong_rf.png)

对角线以上表示广东省优于大湾区。随机CV改善说明更大的环境梯度更容易被嵌入解释；空间CV改善才是更重要的证据，因为它表示模型对未见区域的迁移增强。

## 独立位置学习曲线

![学习曲线比较](gba_vs_guangdong_learning.png)

大湾区最多约7,600个完整位置，广东省可以继续观察到更大样本规模。若广东省曲线继续上升，说明此前确有独立位置不足；若在更大范围仍快速平台化，则模型或数据分辨率成为主要限制。

## 坐标与AlphaEarth贡献

![空间基线比较](gba_vs_guangdong_spatial_features.png)

三随机种子的轻量质量审计中，空间分块平均R²从大湾区 **{gba_quality_spatial:.3f}** 变为广东省 **{gd_quality_spatial:.3f}**。联合坐标和AlphaEarth是否仍优于单独输入，是判断嵌入是否提供额外环境信息的主要依据。

## 解释边界

1. 广东省仍远小于CONUS，无法复制美国大陆的气候、地形和生态梯度。
2. 两个中国区域都使用ERA5-Land和WorldCover替代美国限定的PRISM与NLCD。
3. 正式RF采用30,000行上限；广东省的全部175,875行不会全部进入每个逐变量RF模型。
4. Transformer仍是适合区域数据的2层4头缩小版，不等同于论文约500万行的4层8头模型。
5. 0.5°是两区域主要空间验证口径；广东省另外提供0.5°、1°、2°敏感性，用于接近论文2°分块。
"""
    args.output.write_text(report, encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
