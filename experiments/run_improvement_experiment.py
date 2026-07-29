"""Run paired baseline/improvement experiments and generate deliverables.

Usage:
    python experiments/run_improvement_experiment.py
    python experiments/run_improvement_experiment.py --quick

All comparisons are paired by dataset seed and split seed. The full run uses
20 independent datasets, a five-model ablation suite, spatial block holdout,
and a sample-size/noise robustness grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generate_synthetic_data import generate_dataset
from src.improved_extra_trees import MODEL_CONFIGS, evaluate_configuration

OUT_DIR = ROOT / "results" / "improvement"
FIGURE_DIR = OUT_DIR / "figures"

MODEL_LABELS = {
    "rf_baseline": "RF baseline\n50 trees",
    "et_random_thresholds": "ET thresholds\n50 trees",
    "et_more_trees": "ET thresholds\n100 trees",
    "et_feature_subsample": "ET + feature\nsubsampling",
    "et_full": "Full ET\n100 trees, 0.7 features",
}


def spatial_block_groups(coords: np.ndarray, block_size_deg: float = 4.0) -> np.ndarray:
    """Assign coordinates to deterministic rectangular spatial blocks."""

    coords = np.asarray(coords, dtype=float)
    lon_bin = np.floor((coords[:, 0] - coords[:, 0].min()) / block_size_deg).astype(int)
    lat_bin = np.floor((coords[:, 1] - coords[:, 1].min()) / block_size_deg).astype(int)
    return lon_bin * (lat_bin.max() + 1) + lat_bin


def paired_statistics(baseline: np.ndarray, improved: np.ndarray, seed: int = 20260729) -> dict:
    """Paired tests, bootstrap CI, and paired standardized effect size."""

    baseline = np.asarray(baseline, dtype=float)
    improved = np.asarray(improved, dtype=float)
    if baseline.shape != improved.shape or baseline.ndim != 1 or len(baseline) < 2:
        raise ValueError("paired_statistics requires equally sized 1-D arrays")

    delta = improved - baseline
    rng = np.random.default_rng(seed)
    boot = np.mean(
        delta[rng.integers(0, len(delta), size=(10000, len(delta)))],
        axis=1,
    )
    if np.allclose(delta, 0):
        t_stat, t_p, w_stat, w_p = 0.0, 1.0, 0.0, 1.0
    else:
        t_result = ttest_rel(improved, baseline)
        t_stat, t_p = float(t_result.statistic), float(t_result.pvalue)
        try:
            w_result = wilcoxon(improved, baseline, zero_method="wilcox")
            w_stat, w_p = float(w_result.statistic), float(w_result.pvalue)
        except ValueError:
            w_stat, w_p = 0.0, 1.0

    sd_delta = float(np.std(delta, ddof=1))
    return {
        "n_pairs": int(len(delta)),
        "baseline_mean": float(np.mean(baseline)),
        "improved_mean": float(np.mean(improved)),
        "mean_delta": float(np.mean(delta)),
        "delta_ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
        "paired_t_statistic": t_stat,
        "paired_t_pvalue": t_p,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_pvalue": w_p,
        "cohens_dz": float(np.mean(delta) / sd_delta) if sd_delta > 0 else 0.0,
        "improved_win_rate": float(np.mean(delta > 0)),
    }


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg false-discovery-rate adjustment."""

    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out.tolist()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one_condition(
    seed: int,
    n_samples: int,
    noise_level: float,
    split_type: str,
    configs,
    n_jobs: int,
) -> tuple[list[dict], list[dict]]:
    """Generate one dataset and evaluate all requested models on one split."""

    x, y, coords, gt, names, _ = generate_dataset(
        n_samples=n_samples,
        noise_level=noise_level,
        random_seed=seed,
    )
    indices = np.arange(n_samples)
    if split_type == "random":
        train, test = train_test_split(
            indices, test_size=0.25, random_state=10000 + seed
        )
    elif split_type == "spatial":
        groups = spatial_block_groups(coords)
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=20000 + seed
        )
        train, test = next(splitter.split(x, y, groups))
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    aggregate_rows = []
    variable_rows = []
    for config in configs:
        aggregate, per_variable = evaluate_configuration(
            config=config,
            embeddings=x,
            targets=y,
            train_indices=train,
            test_indices=test,
            variable_names=names,
            ground_truth=gt,
            random_seed=seed,
            n_jobs=n_jobs,
        )
        metadata = {
            "seed": seed,
            "n_samples": n_samples,
            "noise_level": noise_level,
            "split_type": split_type,
        }
        aggregate_rows.append({**metadata, **aggregate})
        variable_rows.extend({**metadata, **row} for row in per_variable)
    return aggregate_rows, variable_rows


def select(rows: list[dict], **conditions) -> list[dict]:
    return [
        row for row in rows
        if all(row.get(key) == value for key, value in conditions.items())
    ]


def make_summary(aggregate_rows: list[dict], variable_rows: list[dict], main_n: int) -> dict:
    main = select(
        aggregate_rows,
        n_samples=main_n,
        noise_level=0.3,
        split_type="random",
    )
    by_model = defaultdict(dict)
    for row in main:
        by_model[row["model"]][row["seed"]] = row

    common_seeds = sorted(set(by_model["rf_baseline"]) & set(by_model["et_full"]))
    baseline = np.array([by_model["rf_baseline"][s]["macro_test_r2"] for s in common_seeds])
    improved = np.array([by_model["et_full"][s]["macro_test_r2"] for s in common_seeds])
    primary = paired_statistics(baseline, improved)

    ablations = {}
    for model, seed_rows in by_model.items():
        seeds = sorted(set(by_model["rf_baseline"]) & set(seed_rows))
        base_values = np.array([by_model["rf_baseline"][s]["macro_test_r2"] for s in seeds])
        model_values = np.array([seed_rows[s]["macro_test_r2"] for s in seeds])
        ablations[model] = paired_statistics(base_values, model_values)

    spatial_rows = select(
        aggregate_rows,
        n_samples=main_n,
        noise_level=0.3,
        split_type="spatial",
    )
    spatial_by_model = defaultdict(dict)
    for row in spatial_rows:
        spatial_by_model[row["model"]][row["seed"]] = row
    spatial_seeds = sorted(
        set(spatial_by_model["rf_baseline"]) & set(spatial_by_model["et_full"])
    )
    spatial = paired_statistics(
        np.array([spatial_by_model["rf_baseline"][s]["macro_test_r2"] for s in spatial_seeds]),
        np.array([spatial_by_model["et_full"][s]["macro_test_r2"] for s in spatial_seeds]),
    )

    main_variables = select(
        variable_rows,
        n_samples=main_n,
        noise_level=0.3,
        split_type="random",
    )
    variable_names = sorted({row["variable"] for row in main_variables})
    per_variable = []
    raw_pvalues = []
    for variable in variable_names:
        base_map = {
            row["seed"]: row["test_r2"]
            for row in main_variables
            if row["model"] == "rf_baseline" and row["variable"] == variable
        }
        improved_map = {
            row["seed"]: row["test_r2"]
            for row in main_variables
            if row["model"] == "et_full" and row["variable"] == variable
        }
        seeds = sorted(set(base_map) & set(improved_map))
        stats = paired_statistics(
            np.array([base_map[s] for s in seeds]),
            np.array([improved_map[s] for s in seeds]),
        )
        per_variable.append({"variable": variable, **stats})
        raw_pvalues.append(stats["paired_t_pvalue"])
    for row, adjusted in zip(per_variable, benjamini_hochberg(raw_pvalues)):
        row["paired_t_qvalue_bh"] = adjusted

    robustness = []
    robustness_rows = [row for row in aggregate_rows if row["split_type"] == "robustness"]
    condition_keys = sorted({
        (row["n_samples"], row["noise_level"]) for row in robustness_rows
    })
    for n_samples, noise_level in condition_keys:
        condition = [
            row for row in robustness_rows
            if row["n_samples"] == n_samples and row["noise_level"] == noise_level
        ]
        cond_by_model = defaultdict(dict)
        for row in condition:
            cond_by_model[row["model"]][row["seed"]] = row
        seeds = sorted(set(cond_by_model["rf_baseline"]) & set(cond_by_model["et_full"]))
        stats = paired_statistics(
            np.array([cond_by_model["rf_baseline"][s]["macro_test_r2"] for s in seeds]),
            np.array([cond_by_model["et_full"][s]["macro_test_r2"] for s in seeds]),
        )
        robustness.append({"n_samples": n_samples, "noise_level": noise_level, **stats})

    support_stats = {}
    for metric in ["support_map", "support_recall_at_true_k", "mapping_recall_at_3"]:
        support_stats[metric] = paired_statistics(
            np.array([by_model["rf_baseline"][s][metric] for s in common_seeds]),
            np.array([by_model["et_full"][s][metric] for s in common_seeds]),
        )

    runtime = {}
    for model, seed_rows in by_model.items():
        values = np.array([row["fit_eval_seconds"] for row in seed_rows.values()])
        runtime[model] = {
            "median_seconds": float(np.median(values)),
            "mean_seconds": float(np.mean(values)),
        }
    runtime["et_full_vs_rf_ratio"] = (
        runtime["et_full"]["median_seconds"] / runtime["rf_baseline"]["median_seconds"]
    )

    return {
        "primary_random_holdout": primary,
        "spatial_block_holdout": spatial,
        "support_recovery": support_stats,
        "ablation": ablations,
        "robustness": robustness,
        "per_variable": per_variable,
        "runtime": runtime,
    }


def save_figures(summary: dict, aggregate_rows: list[dict], main_n: int) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
    })
    navy, orange, teal, gray = "#204060", "#D26A3A", "#2A7F7F", "#8795A1"

    main = select(
        aggregate_rows,
        n_samples=main_n,
        noise_level=0.3,
        split_type="random",
    )
    by_model = defaultdict(dict)
    for row in main:
        by_model[row["model"]][row["seed"]] = row
    seeds = sorted(set(by_model["rf_baseline"]) & set(by_model["et_full"]))
    base = np.array([by_model["rf_baseline"][s]["macro_test_r2"] for s in seeds])
    full = np.array([by_model["et_full"][s]["macro_test_r2"] for s in seeds])

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(base, full, color=teal, edgecolor="white", s=58, alpha=.9)
    low = min(base.min(), full.min()) - .01
    high = max(base.max(), full.max()) + .01
    ax.plot([low, high], [low, high], "--", color=gray, lw=1.2, label="No change")
    ax.set(xlabel="RF baseline macro test R²", ylabel="Improved ExtraTrees macro test R²",
           title="Paired performance across independent dataset seeds")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig1_paired_random_r2.png", bbox_inches="tight")
    plt.close(fig)

    per_var = sorted(summary["per_variable"], key=lambda row: row["mean_delta"])
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    y = np.arange(len(per_var))
    means = np.array([row["mean_delta"] for row in per_var])
    lower = means - np.array([row["delta_ci95"][0] for row in per_var])
    upper = np.array([row["delta_ci95"][1] for row in per_var]) - means
    colors = [teal if row["paired_t_qvalue_bh"] < .05 else gray for row in per_var]
    ax.barh(y, means, xerr=[lower, upper], color=colors, alpha=.9, capsize=2)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_yticks(y, [row["variable"] for row in per_var])
    ax.set(xlabel="Mean paired ΔR² (ExtraTrees - RF)",
           title="Per-variable gains with 95% bootstrap confidence intervals")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig2_per_variable_delta.png", bbox_inches="tight")
    plt.close(fig)

    model_order = [config.name for config in MODEL_CONFIGS]
    means = np.array([summary["ablation"][model]["mean_delta"] for model in model_order])
    error_low = means - np.array([
        summary["ablation"][m]["delta_ci95"][0] for m in model_order
    ])
    error_high = np.array([
        summary["ablation"][m]["delta_ci95"][1] for m in model_order
    ]) - means
    fig, ax = plt.subplots(figsize=(8.1, 4.7))
    bars = ax.bar(
        np.arange(len(model_order)), means,
        yerr=[np.maximum(error_low, 0), np.maximum(error_high, 0)],
        color=[navy, gray, "#6F8FAF", "#74A6A6", orange],
        capsize=3,
    )
    ax.set_xticks(np.arange(len(model_order)), [MODEL_LABELS[m] for m in model_order])
    ax.set(ylabel="Paired macro ΔR² vs RF",
           title="Ablation: contribution of each ExtraTrees modification")
    ax.grid(axis="y", alpha=.2)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, value + .0007, f"{value:+.3f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig3_ablation.png", bbox_inches="tight")
    plt.close(fig)

    robust = summary["robustness"]
    ns = sorted({row["n_samples"] for row in robust})
    noises = sorted({row["noise_level"] for row in robust})
    matrix = np.full((len(noises), len(ns)), np.nan)
    for row in robust:
        matrix[noises.index(row["noise_level"]), ns.index(row["n_samples"])] = row["mean_delta"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    for i in range(len(noises)):
        for j in range(len(ns)):
            ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center",
                    color="white" if matrix[i, j] > np.nanmedian(matrix) else "#1F2933",
                    fontweight="bold")
    ax.set_xticks(range(len(ns)), ns)
    ax.set_yticks(range(len(noises)), noises)
    ax.set(xlabel="Samples", ylabel="Added embedding noise",
           title="Robustness grid: paired macro ΔR²")
    fig.colorbar(im, ax=ax, label="ExtraTrees - RF")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig4_robustness.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for model in model_order:
        x = summary["runtime"][model]["median_seconds"]
        yval = summary["ablation"][model]["improved_mean"]
        ax.scatter(x, yval, s=90, label=MODEL_LABELS[model].replace("\n", " "))
    ax.set(xlabel="Median fit + evaluation time per seed (s)",
           ylabel="Mean macro test R²", title="Accuracy-compute trade-off")
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig5_accuracy_cost.png", bbox_inches="tight")
    plt.close(fig)

    metrics = ["support_map", "support_recall_at_true_k", "mapping_recall_at_3"]
    labels = ["Support mAP", "Recall@true-k", "Mapping recall@3"]
    base_values = [summary["support_recovery"][m]["baseline_mean"] for m in metrics]
    full_values = [summary["support_recovery"][m]["improved_mean"] for m in metrics]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    width = .36
    ax.bar(x - width/2, base_values, width, color=navy, label="RF baseline")
    ax.bar(x + width/2, full_values, width, color=orange, label="Improved ExtraTrees")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set(ylabel="Recovery score", title="Dimension-support recovery comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig6_support_recovery.png", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Two-seed smoke run")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_n = 1200
    main_seeds = list(range(2 if args.quick else 20))
    robust_seeds = list(range(2 if args.quick else 8))

    aggregate_rows = []
    variable_rows = []

    for seed in main_seeds:
        print(f"[main random] seed={seed}")
        agg, var = run_one_condition(
            seed, main_n, 0.3, "random", MODEL_CONFIGS, args.n_jobs
        )
        aggregate_rows.extend(agg)
        variable_rows.extend(var)

        print(f"[spatial] seed={seed}")
        spatial_configs = (MODEL_CONFIGS[0], MODEL_CONFIGS[-1])
        agg, var = run_one_condition(
            seed, main_n, 0.3, "spatial", spatial_configs, args.n_jobs
        )
        aggregate_rows.extend(agg)
        variable_rows.extend(var)

    for n_samples in (500, 1200):
        for noise_level in (0.3, 0.9):
            for seed in robust_seeds:
                print(f"[robustness] n={n_samples} noise={noise_level} seed={seed}")
                agg, var = run_one_condition(
                    seed, n_samples, noise_level, "random",
                    (MODEL_CONFIGS[0], MODEL_CONFIGS[-1]), args.n_jobs,
                )
                for row in agg:
                    row["split_type"] = "robustness"
                for row in var:
                    row["split_type"] = "robustness"
                aggregate_rows.extend(agg)
                variable_rows.extend(var)

    write_csv(OUT_DIR / "aggregate_metrics.csv", aggregate_rows)
    write_csv(OUT_DIR / "per_variable_metrics.csv", variable_rows)

    summary = make_summary(aggregate_rows, variable_rows, main_n)
    experiment = {
        "hypothesis": (
            "Extremely randomized split thresholds plus feature subsampling will "
            "increase held-out environmental-variable reconstruction R2 relative "
            "to the baseline Random Forest while preserving dimension recovery."
        ),
        "primary_endpoint": "Paired macro held-out R2 across 26 variables and 20 seeds",
        "main_n_samples": main_n,
        "main_noise_level": 0.3,
        "main_seeds": main_seeds,
        "robustness_seeds": robust_seeds,
        "model_configs": [config.to_dict() for config in MODEL_CONFIGS],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "summary": summary,
    }
    with (OUT_DIR / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment, handle, ensure_ascii=False, indent=2)
    save_figures(summary, aggregate_rows, main_n)
    print(json.dumps(summary["primary_random_holdout"], indent=2))
    print(f"Results written to {OUT_DIR}")


if __name__ == "__main__":
    main()
