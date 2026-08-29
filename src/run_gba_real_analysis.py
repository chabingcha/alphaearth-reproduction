"""Run the paper-aligned analyses on the real GBA dataset.

This module deliberately does not read any of the repository's hard-coded
``GBA_KNOWN_RESULTS``.  Every reported number is recomputed from the aligned
real AlphaEarth and environmental Parquet tables.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV

from src.fetch_gba_environment_gee import VARIABLES


ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_COLUMNS = [f"A{dim:02d}" for dim in range(64)]
CATEGORIES = {
    "terrain": ["elevation", "slope", "aspect", "flow_accumulation"],
    "soil": ["clay_fraction", "organic_carbon", "soil_ph", "water_capacity"],
    "vegetation": ["ndvi_mean", "ndvi_max", "evi_mean", "lai_mean", "tree_cover", "albedo"],
    "temperature": ["lst_daytime", "lst_nighttime", "air_temp_mean", "dew_point_temp"],
    "climate": ["annual_precip", "max_monthly_precip"],
    "hydrology": ["soil_moisture", "annual_runoff", "annual_et"],
    "urban": ["impervious_surface", "nighttime_lights", "population_density"],
}


def load_aligned(embeddings_path: Path, environment_path: Path) -> pd.DataFrame:
    embeddings = pd.read_parquet(embeddings_path)
    environment = pd.read_parquet(environment_path)
    env_values = environment[["point_id", "year", *VARIABLES]]
    aligned = embeddings.merge(
        env_values,
        on=["point_id", "year"],
        how="inner",
        validate="one_to_one",
    )
    if len(aligned) != len(embeddings):
        raise ValueError(
            f"Alignment lost rows: embeddings={len(embeddings)}, aligned={len(aligned)}"
        )
    return aligned


def spearman_matrix(frame: pd.DataFrame, max_samples: int, seed: int) -> dict:
    if len(frame) > max_samples:
        sample = frame.sample(max_samples, random_state=seed)
    else:
        sample = frame
    x = sample[EMBEDDING_COLUMNS].to_numpy(dtype=np.float64)
    rho = np.full((64, len(VARIABLES)), np.nan)
    for variable_index, variable in enumerate(VARIABLES):
        y = sample[variable].to_numpy(dtype=np.float64)
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if valid.sum() < 3:
            continue
        ranked_x = np.apply_along_axis(rankdata, 0, x[valid])
        ranked_y = rankdata(y[valid])
        centered_x = ranked_x - ranked_x.mean(axis=0)
        centered_y = ranked_y - ranked_y.mean()
        denominator = np.sqrt(
            np.sum(centered_x ** 2, axis=0) * np.sum(centered_y ** 2)
        )
        rho[:, variable_index] = centered_x.T @ centered_y / denominator

    best = {}
    for variable_index, variable in enumerate(VARIABLES):
        column = rho[:, variable_index]
        if np.isnan(column).all():
            continue
        dim = int(np.nanargmax(np.abs(column)))
        best[variable] = {"dimension": f"A{dim:02d}", "rho": float(column[dim])}
    per_dimension_max = np.nanmax(np.abs(rho), axis=1)
    return {
        "n_samples": int(len(sample)),
        "rho_matrix": rho.tolist(),
        "best_dimension_per_variable": best,
        "max_abs_rho": float(np.nanmax(np.abs(rho))),
        "dimensions_abs_rho_gt_0_5": int(np.sum(per_dimension_max > 0.5)),
        "dimensions_abs_rho_gt_0_7": int(np.sum(per_dimension_max > 0.7)),
    }


def block_groups(frame: pd.DataFrame, block_size_deg: float) -> np.ndarray:
    lon_block = np.floor((frame["lon"] - frame["lon"].min()) / block_size_deg).astype(int)
    lat_block = np.floor((frame["lat"] - frame["lat"].min()) / block_size_deg).astype(int)
    return (lat_block * (lon_block.max() + 1) + lon_block).to_numpy()


def rf_analysis(
    frame: pd.DataFrame,
    max_samples: int,
    trees: int,
    folds: int,
    block_size_deg: float,
    seed: int,
) -> dict:
    results = {}
    importances = np.full((64, len(VARIABLES)), np.nan)
    rng = np.random.default_rng(seed)

    for variable_index, variable in enumerate(VARIABLES):
        columns = ["lon", "lat", *EMBEDDING_COLUMNS, variable]
        data = frame[columns].dropna()
        if len(data) > max_samples:
            data = data.sample(max_samples, random_state=seed)
        x = data[EMBEDDING_COLUMNS].to_numpy(dtype=np.float32)
        y = data[variable].to_numpy(dtype=np.float32)
        groups = block_groups(data, block_size_deg)

        random_splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        spatial_folds = min(folds, len(np.unique(groups)))
        spatial_splitter = GroupKFold(n_splits=spatial_folds)

        def new_model() -> RandomForestRegressor:
            return RandomForestRegressor(
                n_estimators=trees,
                min_samples_leaf=2,
                max_features=1.0,
                random_state=seed,
                n_jobs=-1,
            )

        random_scores = []
        for train, test in random_splitter.split(x):
            model = new_model().fit(x[train], y[train])
            random_scores.append(r2_score(y[test], model.predict(x[test])))

        spatial_scores = []
        for train, test in spatial_splitter.split(x, y, groups):
            model = new_model().fit(x[train], y[train])
            spatial_scores.append(r2_score(y[test], model.predict(x[test])))

        final_model = new_model().fit(x, y)
        importance_count = min(5000, len(x))
        importance_idx = rng.choice(len(x), importance_count, replace=False)
        importance = permutation_importance(
            final_model,
            x[importance_idx],
            y[importance_idx],
            n_repeats=3,
            random_state=seed,
            n_jobs=-1,
        ).importances_mean
        importances[:, variable_index] = importance
        top3 = np.argsort(importance)[-3:][::-1]

        random_mean = float(np.mean(random_scores))
        spatial_mean = float(np.mean(spatial_scores))
        results[variable] = {
            "n_samples": int(len(data)),
            "random_cv_r2_mean": random_mean,
            "random_cv_r2_std": float(np.std(random_scores)),
            "spatial_cv_r2_mean": spatial_mean,
            "spatial_cv_r2_std": float(np.std(spatial_scores)),
            "delta_r2": random_mean - spatial_mean,
            "top3_dimensions": [f"A{dim:02d}" for dim in top3],
        }
        print(
            f"  RF {variable_index + 1:02d}/{len(VARIABLES)} {variable}: "
            f"random R2={random_mean:.3f}, spatial R2={spatial_mean:.3f}",
            flush=True,
        )

    random_values = [value["random_cv_r2_mean"] for value in results.values()]
    delta_values = [value["delta_r2"] for value in results.values()]
    top10 = sorted(
        results.values(),
        key=lambda value: value["random_cv_r2_mean"],
        reverse=True,
    )[:10]
    return {
        "settings": {
            "max_samples_per_variable": max_samples,
            "trees": trees,
            "folds": folds,
            "spatial_block_size_deg": block_size_deg,
        },
        "per_variable": results,
        "permutation_importance_matrix": importances.tolist(),
        "summary": {
            "mean_random_cv_r2": float(np.mean(random_values)),
            "best_random_cv_r2": float(np.max(random_values)),
            "variables_r2_gt_0_9": int(np.sum(np.asarray(random_values) > 0.9)),
            "variables_r2_gt_0_7": int(np.sum(np.asarray(random_values) > 0.7)),
            "mean_spatial_delta_r2": float(np.mean(delta_values)),
            "mean_abs_spatial_delta_r2": float(np.mean(np.abs(delta_values))),
            "top10_mean_spatial_delta_r2": float(
                np.mean([value["delta_r2"] for value in top10])
            ),
        },
    }


def temporal_stability(frame: pd.DataFrame, max_samples_per_year: int, seed: int) -> dict:
    profiles = {}
    for year, group in frame.groupby("year"):
        result = spearman_matrix(group, max_samples_per_year, seed)
        profiles[int(year)] = np.asarray(result["rho_matrix"], dtype=float)

    per_dimension = []
    for dim in range(64):
        correlations = []
        for year_a, year_b in combinations(sorted(profiles), 2):
            a = profiles[year_a][dim]
            b = profiles[year_b][dim]
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() >= 3 and np.std(a[valid]) > 0 and np.std(b[valid]) > 0:
                correlations.append(pearsonr(a[valid], b[valid]).statistic)
        per_dimension.append(float(np.mean(correlations)) if correlations else None)
    finite = np.asarray([value for value in per_dimension if value is not None])
    return {
        "per_dimension_mean_pairwise_r": per_dimension,
        "mean_inter_year_r": float(finite.mean()),
        "dimensions_r_gt_0_95": int(np.sum(finite > 0.95)),
        "dimensions_r_gt_0_90": int(np.sum(finite > 0.90)),
    }


def spatial_baseline(frame: pd.DataFrame, folds: int, seed: int) -> dict:
    coordinates = frame[["lon", "lat"]].to_numpy(dtype=float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    results = {}
    for variable in VARIABLES:
        y = frame[variable].to_numpy(dtype=float)
        valid = np.isfinite(y)
        scores = []
        for train, test in splitter.split(coordinates[valid]):
            model = make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False),
                StandardScaler(),
                RidgeCV(alphas=np.logspace(-3, 3, 13)),
            )
            x_valid = coordinates[valid]
            y_valid = y[valid]
            model.fit(x_valid[train], y_valid[train])
            scores.append(model.score(x_valid[test], y_valid[test]))
        results[variable] = float(np.mean(scores))
    return results


def method_convergence(spearman: dict, rf: dict) -> float:
    rho = np.abs(np.asarray(spearman["rho_matrix"], dtype=float)).ravel()
    importance = np.asarray(rf["permutation_importance_matrix"], dtype=float).ravel()
    valid = np.isfinite(rho) & np.isfinite(importance)
    return float(pearsonr(rho[valid], importance[valid]).statistic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "data" / "processed" / "gba_aef_2017_2023.parquet",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=ROOT / "data" / "processed" / "gba_environment_2017_2023.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "gba_real" / "analysis_results.json",
    )
    parser.add_argument("--max-rf-samples", type=int, default=30000)
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--block-size-deg", type=float, default=0.5)
    parser.add_argument("--embedding-scale-m", type=int, default=640)
    parser.add_argument("--support-area-km2", type=float, default=0.4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = load_aligned(args.embeddings, args.environment)
    print(f"Aligned real observations: {len(frame):,}", flush=True)
    spearman = spearman_matrix(frame, max_samples=len(frame), seed=args.seed)
    print("Spearman analysis complete", flush=True)
    rf = rf_analysis(
        frame,
        max_samples=args.max_rf_samples,
        trees=args.trees,
        folds=args.folds,
        block_size_deg=args.block_size_deg,
        seed=args.seed,
    )
    print("Random Forest and spatial CV complete", flush=True)
    temporal = temporal_stability(frame, max_samples_per_year=10000, seed=args.seed)
    baseline = spatial_baseline(frame, folds=args.folds, seed=args.seed)

    output = {
        "data": {
            "rows": int(len(frame)),
            "locations": int(frame["point_id"].nunique()),
            "years": sorted(int(year) for year in frame["year"].unique()),
            "cities": sorted(str(city) for city in frame["city"].unique()),
            "embedding_resolution_m": args.embedding_scale_m,
            "observation_support_area_km2": args.support_area_km2,
            "sampling_spacing_deg": 0.025,
        },
        "spearman": spearman,
        "random_forest": rf,
        "temporal_stability": temporal,
        "coordinate_only_baseline_r2": baseline,
        "method_convergence_r": method_convergence(spearman, rf),
        "paper_reference": {
            "rf_variables_r2_gt_0_9": 12,
            "transformer_mean_spatial_delta_r2": 0.017,
            "rf_top10_mean_spatial_delta_r2": 0.009,
            "mean_temporal_stability_r": 0.963,
            "spearman_rf_convergence_r": 0.45,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "rf_summary": rf["summary"],
        "temporal_stability": temporal["mean_inter_year_r"],
        "method_convergence": output["method_convergence_r"],
    }, indent=2))


if __name__ == "__main__":
    main()
