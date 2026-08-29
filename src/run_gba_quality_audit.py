"""High-value robustness experiments for the real GBA reproduction.

This audit focuses on scientific quality rather than matching the paper's
scale: location leakage, independent-location learning curves, fair feature
baselines, seed stability, city holdouts, static-target deduplication, circular
aspect handling, target skew and coarse-grid effective sample size.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from src.fetch_gba_environment_gee import VARIABLES
from src.run_gba_real_analysis import EMBEDDING_COLUMNS, block_groups, load_aligned


ROOT = Path(__file__).resolve().parent.parent
STATIC_VARIABLES = [
    "elevation", "slope", "aspect", "flow_accumulation",
    "clay_fraction", "organic_carbon", "soil_ph", "water_capacity",
    "tree_cover", "impervious_surface", "population_density",
]
DYNAMIC_VARIABLES = [name for name in VARIABLES if name not in STATIC_VARIABLES]
SEEDS = [42, 123, 2026]


def coordinate_features(frame: pd.DataFrame) -> np.ndarray:
    lon = frame["lon"].to_numpy(dtype=np.float32)
    lat = frame["lat"].to_numpy(dtype=np.float32)
    lon = lon - lon.mean()
    lat = lat - lat.mean()
    return np.column_stack([lon, lat, lon * lon, lat * lat, lon * lat]).astype(np.float32)


def feature_matrix(frame: pd.DataFrame, feature_set: str) -> np.ndarray:
    coords = coordinate_features(frame)
    embeddings = frame[EMBEDDING_COLUMNS].to_numpy(dtype=np.float32)
    if feature_set == "coordinates":
        return coords
    if feature_set == "embeddings":
        return embeddings
    if feature_set == "combined":
        return np.column_stack([coords, embeddings]).astype(np.float32)
    raise ValueError(feature_set)


def standardize_train_test(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x[train].mean(axis=0)
    x_std = x[train].std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_mean = y[train].mean(axis=0)
    y_std = y[train].std(axis=0)
    y_std[y_std < 1e-8] = 1.0
    return (
        (x[train] - x_mean) / x_std,
        (x[test] - x_mean) / x_std,
        (y[train] - y_mean) / y_std,
        y[test],
        y_mean,
        y_std,
    )


def new_model(model_name: str, seed: int, trees: int):
    if model_name == "ridge":
        return Ridge(alpha=10.0)
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=trees,
            min_samples_leaf=10,
            max_depth=18,
            max_features=0.5,
            max_samples=0.7,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(model_name)


def metric_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    r2 = np.array([r2_score(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])])
    mae = np.array([mean_absolute_error(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])])
    rmse = np.array([
        mean_squared_error(y_true[:, j], y_pred[:, j]) ** 0.5
        for j in range(y_true.shape[1])
    ])
    scale = np.std(y_true, axis=0)
    nrmse = np.divide(rmse, scale, out=np.full_like(rmse, np.nan), where=scale > 0)
    return {"r2": r2, "mae": mae, "nrmse": nrmse}


def fit_evaluate(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    model_name: str,
    seed: int,
    trees: int,
) -> dict:
    x_train, x_test, y_train_scaled, y_test, y_mean, y_std = standardize_train_test(
        x, y, train, test
    )
    model = new_model(model_name, seed, trees)
    model.fit(x_train, y_train_scaled)
    prediction = model.predict(x_test) * y_std + y_mean
    metrics = metric_arrays(y_test, prediction)
    result = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "r2": metrics["r2"].tolist(),
        "mae": metrics["mae"].tolist(),
        "nrmse": metrics["nrmse"].tolist(),
    }
    del model
    gc.collect()
    return result


def split_folds(frame: pd.DataFrame, scheme: str, folds: int, seed: int):
    indices = np.arange(len(frame))
    if scheme == "random_rows":
        return list(KFold(folds, shuffle=True, random_state=seed).split(indices))
    if scheme == "point_group":
        groups = frame["point_id"].to_numpy()
    elif scheme == "spatial_block":
        groups = block_groups(frame, 0.5)
    else:
        raise ValueError(scheme)
    return list(GroupKFold(folds).split(indices, groups=groups))


def summarize_fold_results(results: list[dict]) -> dict:
    output = {}
    for metric in ["r2", "mae", "nrmse"]:
        matrix = np.asarray([item[metric] for item in results], dtype=float)
        output[metric] = {
            variable: {
                "mean": float(np.nanmean(matrix[:, j])),
                "std": float(np.nanstd(matrix[:, j])),
                "fold_values": matrix[:, j].tolist(),
            }
            for j, variable in enumerate(VARIABLES)
        }
        output[f"mean_{metric}"] = float(np.nanmean(matrix))
        output[f"static_mean_{metric}"] = float(
            np.nanmean(matrix[:, [VARIABLES.index(v) for v in STATIC_VARIABLES]])
        )
        output[f"dynamic_mean_{metric}"] = float(
            np.nanmean(matrix[:, [VARIABLES.index(v) for v in DYNAMIC_VARIABLES]])
        )
    return output


def run_fair_baselines(frame: pd.DataFrame, trees: int, folds: int) -> dict:
    complete = frame.dropna(subset=VARIABLES).reset_index(drop=True)
    y = complete[VARIABLES].to_numpy(dtype=np.float32)
    output = {"rows": len(complete), "folds": folds, "experiments": {}}
    configurations = [
        ("ridge", "coordinates"),
        ("ridge", "embeddings"),
        ("ridge", "combined"),
        ("random_forest", "coordinates"),
        ("random_forest", "embeddings"),
        ("random_forest", "combined"),
    ]
    for scheme in ["random_rows", "point_group", "spatial_block"]:
        for model_name, feature_set in configurations:
            key = f"{scheme}__{model_name}__{feature_set}"
            x = feature_matrix(complete, feature_set)
            results = []
            for fold_index, (train, test) in enumerate(
                split_folds(complete, scheme, folds, 42), start=1
            ):
                results.append(
                    fit_evaluate(x, y, train, test, model_name, 42 + fold_index, trees)
                )
            output["experiments"][key] = summarize_fold_results(results)
            print(
                f"  baseline {key}: mean R2={output['experiments'][key]['mean_r2']:.3f}",
                flush=True,
            )
    return output


def run_seed_stability(frame: pd.DataFrame, trees: int, folds: int) -> dict:
    complete = frame.dropna(subset=VARIABLES).reset_index(drop=True)
    x = feature_matrix(complete, "embeddings")
    y = complete[VARIABLES].to_numpy(dtype=np.float32)
    output = {}
    for scheme in ["random_rows", "point_group", "spatial_block"]:
        seed_results = {}
        for seed in SEEDS:
            results = []
            for fold_index, (train, test) in enumerate(
                split_folds(complete, scheme, folds, seed), start=1
            ):
                results.append(
                    fit_evaluate(x, y, train, test, "random_forest", seed + fold_index, trees)
                )
            seed_results[str(seed)] = summarize_fold_results(results)
            print(
                f"  seed {scheme} {seed}: mean R2={seed_results[str(seed)]['mean_r2']:.3f}",
                flush=True,
            )
        values = np.asarray([seed_results[str(seed)]["mean_r2"] for seed in SEEDS])
        output[scheme] = {
            "per_seed": seed_results,
            "mean_r2_across_seeds": float(values.mean()),
            "std_r2_across_seeds": float(values.std()),
        }
    return output


def holdout_split(frame: pd.DataFrame, scheme: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(frame))
    if scheme == "random_rows":
        shuffled = rng.permutation(all_indices)
        cut = int(0.8 * len(shuffled))
        return shuffled[:cut], shuffled[cut:]
    if scheme == "point_group":
        groups = frame["point_id"].unique()
        test_groups = set(rng.choice(groups, max(1, int(0.2 * len(groups))), replace=False))
        test_mask = frame["point_id"].isin(test_groups).to_numpy()
    elif scheme == "spatial_block":
        group_values = block_groups(frame, 0.5)
        groups = np.unique(group_values)
        test_groups = set(rng.choice(groups, max(1, int(0.2 * len(groups))), replace=False))
        test_mask = np.isin(group_values, list(test_groups))
    else:
        raise ValueError(scheme)
    return all_indices[~test_mask], all_indices[test_mask]


def run_learning_curves(frame: pd.DataFrame, trees: int) -> dict:
    complete = frame.dropna(subset=VARIABLES).reset_index(drop=True)
    all_points = complete["point_id"].unique()
    requested_sizes = [500, 1000, 2500, 5000, 10000, len(all_points)]
    sizes = sorted(set(min(size, len(all_points)) for size in requested_sizes))
    output = {"available_complete_case_points": int(len(all_points)), "curves": {}}
    for size in sizes:
        size_results = {}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            selected = rng.choice(all_points, size=size, replace=False)
            subset = complete[complete["point_id"].isin(selected)].reset_index(drop=True)
            x = feature_matrix(subset, "embeddings")
            y = subset[VARIABLES].to_numpy(dtype=np.float32)
            seed_schemes = {}
            for scheme in ["random_rows", "point_group", "spatial_block"]:
                train, test = holdout_split(subset, scheme, seed)
                if len(test) < 100 or len(train) < 100:
                    continue
                result = fit_evaluate(
                    x, y, train, test, "random_forest", seed, trees
                )
                summary = summarize_fold_results([result])
                seed_schemes[scheme] = summary
            size_results[str(seed)] = seed_schemes
        output["curves"][str(size)] = size_results
        print(f"  learning curve locations={size:,} complete", flush=True)
    return output


def run_static_dedup(frame: pd.DataFrame, trees: int, folds: int) -> dict:
    latest = frame[frame["year"] == frame["year"].max()].dropna(subset=VARIABLES).reset_index(drop=True)
    x = feature_matrix(latest, "embeddings")
    y = latest[VARIABLES].to_numpy(dtype=np.float32)
    output = {"year": int(latest["year"].iloc[0]), "rows": len(latest), "schemes": {}}
    for scheme in ["random_rows", "spatial_block"]:
        results = []
        for fold_index, (train, test) in enumerate(
            split_folds(latest, scheme, folds, 42), start=1
        ):
            results.append(
                fit_evaluate(x, y, train, test, "random_forest", 42 + fold_index, trees)
            )
        output["schemes"][scheme] = summarize_fold_results(results)
        print(
            f"  static dedup {scheme}: static mean R2="
            f"{output['schemes'][scheme]['static_mean_r2']:.3f}", flush=True,
        )
    return output


def run_city_holdout(frame: pd.DataFrame, trees: int) -> dict:
    complete = frame.dropna(subset=VARIABLES).reset_index(drop=True)
    x = feature_matrix(complete, "embeddings")
    y = complete[VARIABLES].to_numpy(dtype=np.float32)
    output = {}
    for city in sorted(complete["city"].unique()):
        test = np.flatnonzero((complete["city"] == city).to_numpy())
        train = np.flatnonzero((complete["city"] != city).to_numpy())
        if len(test) < 20:
            output[city] = {"n_test": int(len(test)), "status": "too few rows"}
            continue
        result = fit_evaluate(x, y, train, test, "random_forest", 42, trees)
        summary = summarize_fold_results([result])
        output[city] = {"n_test": int(len(test)), **summary}
        print(f"  city holdout {city}: mean R2={summary['mean_r2']:.3f}", flush=True)
    return output


def circular_error_degrees(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.abs((predicted - true + 180.0) % 360.0 - 180.0)


def run_target_transforms(frame: pd.DataFrame, trees: int, folds: int) -> dict:
    output = {
        "flow_accumulation_note": "Already log(cell count + 1) in the extraction pipeline; no second log applied.",
        "skewness": {},
    }
    for variable in ["population_density", "nighttime_lights", "annual_runoff", "flow_accumulation"]:
        values = frame[variable].dropna()
        output["skewness"][variable] = float(values.skew())

    population = frame.dropna(subset=["population_density"]).reset_index(drop=True)
    x = feature_matrix(population, "embeddings")
    y_raw = population[["population_density"]].to_numpy(dtype=np.float32)
    groups = population["point_id"].to_numpy()
    raw_scores, log_scores = [], []
    for fold_index, (train, test) in enumerate(
        GroupKFold(folds).split(x, groups=groups), start=1
    ):
        for transform, destination in [(False, raw_scores), (True, log_scores)]:
            y_train_target = np.log1p(y_raw) if transform else y_raw
            x_train, x_test, y_train_scaled, _, y_mean, y_std = standardize_train_test(
                x, y_train_target, train, test
            )
            model = new_model("random_forest", 42 + fold_index, trees)
            model.fit(x_train, y_train_scaled.ravel())
            predicted_target = model.predict(x_test) * y_std[0] + y_mean[0]
            predicted_raw = np.expm1(predicted_target) if transform else predicted_target
            destination.append(
                {
                    "original_scale_r2": float(r2_score(y_raw[test, 0], predicted_raw)),
                    "original_scale_mae": float(mean_absolute_error(y_raw[test, 0], predicted_raw)),
                    "log_scale_r2": float(
                        r2_score(np.log1p(y_raw[test, 0]), np.log1p(np.maximum(predicted_raw, 0)))
                    ),
                }
            )
            del model
            gc.collect()
    output["population_raw_vs_log1p"] = {
        "raw_target": raw_scores,
        "log1p_target": log_scores,
        "raw_mean_original_r2": float(np.mean([v["original_scale_r2"] for v in raw_scores])),
        "log_mean_original_r2": float(np.mean([v["original_scale_r2"] for v in log_scores])),
        "raw_mean_log_r2": float(np.mean([v["log_scale_r2"] for v in raw_scores])),
        "log_mean_log_r2": float(np.mean([v["log_scale_r2"] for v in log_scores])),
    }

    aspect = frame.dropna(subset=["aspect"]).reset_index(drop=True)
    x = feature_matrix(aspect, "embeddings")
    degrees = aspect["aspect"].to_numpy(dtype=np.float32)
    radians = np.radians(degrees)
    y_direct = degrees[:, None]
    y_circular = np.column_stack([np.sin(radians), np.cos(radians)]).astype(np.float32)
    groups = aspect["point_id"].to_numpy()
    direct_errors, circular_errors = [], []
    for fold_index, (train, test) in enumerate(
        GroupKFold(folds).split(x, groups=groups), start=1
    ):
        for target, destination, circular in [
            (y_direct, direct_errors, False), (y_circular, circular_errors, True)
        ]:
            x_train, x_test, y_train_scaled, _, y_mean, y_std = standardize_train_test(
                x, target, train, test
            )
            model = new_model("random_forest", 42 + fold_index, trees)
            model.fit(x_train, y_train_scaled.ravel() if target.shape[1] == 1 else y_train_scaled)
            predicted = model.predict(x_test)
            if target.shape[1] == 1:
                predicted = predicted[:, None]
            predicted = predicted * y_std + y_mean
            if circular:
                predicted_degrees = (
                    np.degrees(np.arctan2(predicted[:, 0], predicted[:, 1])) + 360
                ) % 360
            else:
                predicted_degrees = predicted[:, 0] % 360
            errors = circular_error_degrees(degrees[test], predicted_degrees)
            destination.append(
                {
                    "circular_mae_degrees": float(np.mean(errors)),
                    "circular_median_ae_degrees": float(np.median(errors)),
                }
            )
            del model
            gc.collect()
    output["aspect_direct_vs_sincos"] = {
        "direct_degrees": direct_errors,
        "sin_cos": circular_errors,
        "direct_mean_circular_mae": float(np.mean([v["circular_mae_degrees"] for v in direct_errors])),
        "sincos_mean_circular_mae": float(np.mean([v["circular_mae_degrees"] for v in circular_errors])),
    }
    return output


def data_quality_diagnostics(frame: pd.DataFrame) -> dict:
    missing = {
        variable: {
            "count": int(frame[variable].isna().sum()),
            "fraction": float(frame[variable].isna().mean()),
        }
        for variable in VARIABLES
    }
    era5_variables = [
        "air_temp_mean", "dew_point_temp", "annual_precip",
        "max_monthly_precip", "soil_moisture", "annual_runoff", "annual_et",
    ]
    coarse = frame.copy()
    coarse["era5_lon_cell"] = np.floor((coarse["lon"] + 180) / 0.1).astype(int)
    coarse["era5_lat_cell"] = np.floor((coarse["lat"] + 90) / 0.1).astype(int)
    per_year = []
    for year, group in coarse.groupby("year"):
        cells = group.groupby(["era5_lon_cell", "era5_lat_cell"]).size()
        per_year.append(
            {
                "year": int(year),
                "rows": int(len(group)),
                "approximate_0_1_degree_cells": int(len(cells)),
                "mean_samples_per_source_cell": float(cells.mean()),
                "max_samples_per_source_cell": int(cells.max()),
            }
        )
    static_year_variation = {}
    for variable in STATIC_VARIABLES:
        by_point = frame.groupby("point_id")[variable].nunique(dropna=True)
        static_year_variation[variable] = {
            "points_with_more_than_one_value": int((by_point > 1).sum()),
            "points_checked": int(len(by_point)),
        }
    return {
        "rows": int(len(frame)),
        "unique_points": int(frame["point_id"].nunique()),
        "complete_rows_all_26": int(frame[VARIABLES].notna().all(axis=1).sum()),
        "missingness": missing,
        "era5_effective_grid": {
            "nominal_resolution": "about 0.1 degree / 11 km",
            "variables": era5_variables,
            "per_year": per_year,
            "warning": "Sampling after bilinear reprojection does not create independent source observations within a source grid cell.",
        },
        "static_target_year_variation": static_year_variation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings", type=Path,
        default=ROOT / "data/processed/gba_aef_gee_1km_2017_2023.parquet",
    )
    parser.add_argument(
        "--environment", type=Path,
        default=ROOT / "data/processed/gba_environment_point_10m_2017_2023.parquet",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/gba_real/quality_audit_results.json",
    )
    parser.add_argument("--trees", type=int, default=30)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    frame = load_aligned(args.embeddings, args.environment)
    started = time.time()
    output = {
        "settings": {
            "trees": args.trees,
            "folds": args.folds,
            "seeds": SEEDS,
            "rf": {
                "min_samples_leaf": 10,
                "max_depth": 18,
                "max_features": 0.5,
                "max_samples": 0.7,
                "multi_output_targets_standardized": True,
            },
            "note": "Quality-audit models are lightweight multi-output controls and do not replace the paper-comparable single-target RF results.",
        },
        "data_quality": data_quality_diagnostics(frame),
    }
    print("Running fair feature/model baselines", flush=True)
    output["fair_baselines"] = run_fair_baselines(frame, args.trees, args.folds)
    print("Running seed stability", flush=True)
    output["seed_stability"] = run_seed_stability(frame, args.trees, args.folds)
    print("Running independent-location learning curves", flush=True)
    output["learning_curves"] = run_learning_curves(frame, args.trees)
    print("Running static-target deduplication", flush=True)
    output["static_dedup"] = run_static_dedup(frame, args.trees, args.folds)
    print("Running leave-one-city-out validation", flush=True)
    output["city_holdout"] = run_city_holdout(frame, args.trees)
    print("Running target transform sensitivity", flush=True)
    output["target_transforms"] = run_target_transforms(frame, args.trees, args.folds)
    output["runtime_seconds"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "runtime_minutes": output["runtime_seconds"] / 60,
                "complete_rows": output["data_quality"]["complete_rows_all_26"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
