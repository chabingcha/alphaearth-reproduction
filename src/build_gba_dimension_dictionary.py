"""Combine Spearman, Random Forest and Transformer evidence into a dictionary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from src.fetch_gba_environment_gee import VARIABLES
from src.run_gba_real_analysis import CATEGORIES, EMBEDDING_COLUMNS


ROOT = Path(__file__).resolve().parent.parent
CATEGORY_BY_VARIABLE = {
    variable: category for category, variables in CATEGORIES.items() for variable in variables
}
VARIABLE_METADATA = {
    "elevation": ("高程", "m"),
    "slope": ("坡度", "degree"),
    "aspect": ("坡向", "degree"),
    "flow_accumulation": ("汇流累积（对数）", "log(cell count + 1)"),
    "clay_fraction": ("黏土比例", "%"),
    "organic_carbon": ("土壤有机碳", "g/kg"),
    "soil_ph": ("土壤pH", "pH"),
    "water_capacity": ("田间持水量", "%"),
    "ndvi_mean": ("年均NDVI", "index"),
    "ndvi_max": ("年最大NDVI", "index"),
    "evi_mean": ("年均EVI", "index"),
    "lai_mean": ("年均叶面积指数", "m2/m2"),
    "tree_cover": ("树冠覆盖率", "%"),
    "albedo": ("短波反照率", "fraction"),
    "lst_daytime": ("日间地表温度", "degC"),
    "lst_nighttime": ("夜间地表温度", "degC"),
    "air_temp_mean": ("年均气温", "degC"),
    "dew_point_temp": ("年均露点温度", "degC"),
    "annual_precip": ("年降水量", "mm"),
    "max_monthly_precip": ("最大月降水量", "mm"),
    "soil_moisture": ("表层土壤湿度", "m3/m3"),
    "annual_runoff": ("年径流量", "mm"),
    "annual_et": ("年蒸散量", "mm"),
    "impervious_surface": ("建成区比例", "%"),
    "nighttime_lights": ("夜间灯光", "nW/cm2/sr"),
    "population_density": ("人口密度", "people/km2"),
}


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return float("nan")
    return float(pearsonr(a[valid], b[valid]).statistic)


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.maximum(np.nan_to_num(matrix, nan=0.0), 0.0)
    sums = matrix.sum(axis=0, keepdims=True)
    return np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums > 0)


def top_method_variable(matrix: np.ndarray, dim: int) -> int:
    return int(np.nanargmax(matrix[dim]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        type=Path,
        default=ROOT / "results/gba_real/analysis_results_paper_method.json",
    )
    parser.add_argument(
        "--transformer",
        type=Path,
        default=ROOT / "results/gba_real/transformer_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/gba_real/dimension_dictionary.json",
    )
    parser.add_argument(
        "--dimensions-csv",
        type=Path,
        default=ROOT / "results/gba_real/dimension_dictionary.csv",
    )
    parser.add_argument(
        "--variables-csv",
        type=Path,
        default=ROOT / "results/gba_real/variable_dictionary.csv",
    )
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    transformer = json.loads(args.transformer.read_text(encoding="utf-8"))
    rho = np.asarray(analysis["spearman"]["rho_matrix"], dtype=float)
    rho_abs = np.abs(rho)
    rf_raw = np.asarray(
        analysis["random_forest"]["permutation_importance_matrix"], dtype=float
    )
    transformer_raw = np.asarray(
        transformer["permutation_importance_matrix"], dtype=float
    )
    if rho.shape != (64, 26) or rf_raw.shape != rho.shape or transformer_raw.shape != rho.shape:
        raise ValueError(
            f"Expected three 64x26 matrices, got {rho.shape}, {rf_raw.shape}, "
            f"{transformer_raw.shape}"
        )
    rf = normalize_columns(rf_raw)
    transformer_importance = normalize_columns(transformer_raw)
    spearman_support = normalize_columns(rho_abs)
    combined = (spearman_support + rf + transformer_importance) / 3.0
    temporal = analysis["temporal_stability"]["per_dimension_mean_pairwise_r"]

    dimension_entries = {}
    double_agreement = 0
    triple_agreement = 0
    robust_count = 0
    for dim, name in enumerate(EMBEDDING_COLUMNS):
        method_indices = {
            "spearman": top_method_variable(rho_abs, dim),
            "random_forest": top_method_variable(rf, dim),
            "transformer": top_method_variable(transformer_importance, dim),
        }
        votes = {}
        for method, variable_index in method_indices.items():
            votes.setdefault(variable_index, []).append(method)
        best_vote_index, agreeing_methods = max(
            votes.items(), key=lambda item: (len(item[1]), combined[dim, item[0]])
        )
        if len(agreeing_methods) < 2:
            best_vote_index = int(np.argmax(combined[dim]))
            agreeing_methods = [
                method
                for method, variable_index in method_indices.items()
                if variable_index == best_vote_index
            ]
        agreement_count = len(agreeing_methods)
        double_agreement += agreement_count >= 2
        triple_agreement += agreement_count == 3
        variable = VARIABLES[best_vote_index]
        stability = temporal[dim]
        robust = agreement_count >= 2 and stability is not None and stability >= 0.9
        robust_count += robust
        display, unit = VARIABLE_METADATA[variable]
        dimension_entries[name] = {
            "primary_variable": variable,
            "primary_variable_zh": display,
            "category": CATEGORY_BY_VARIABLE[variable],
            "unit": unit,
            "direction": "positive" if rho[dim, best_vote_index] >= 0 else "negative",
            "spearman_rho": float(rho[dim, best_vote_index]),
            "rf_importance": float(rf[dim, best_vote_index]),
            "transformer_importance": float(
                transformer_importance[dim, best_vote_index]
            ),
            "combined_support": float(combined[dim, best_vote_index]),
            "method_choices": {
                method: VARIABLES[index] for method, index in method_indices.items()
            },
            "agreeing_methods": agreeing_methods,
            "agreement_count": agreement_count,
            "concordant": agreement_count >= 2,
            "temporal_stability_r": stability,
            "robust": robust,
            "top3_combined_variables": [
                {
                    "variable": VARIABLES[index],
                    "score": float(combined[dim, index]),
                    "spearman_rho": float(rho[dim, index]),
                }
                for index in np.argsort(combined[dim])[-3:][::-1]
            ],
        }

    variable_entries = {}
    rf_results = analysis["random_forest"]["per_variable"]
    transformer_results = transformer["per_variable"]
    for j, variable in enumerate(VARIABLES):
        display, unit = VARIABLE_METADATA[variable]
        top_combined = np.argsort(combined[:, j])[-5:][::-1]
        top_spearman = np.argsort(rho_abs[:, j])[-5:][::-1]
        top_rf = np.argsort(rf[:, j])[-5:][::-1]
        top_transformer = np.argsort(transformer_importance[:, j])[-5:][::-1]
        variable_entries[variable] = {
            "display_name_zh": display,
            "category": CATEGORY_BY_VARIABLE[variable],
            "unit": unit,
            "top_dimensions_combined": [EMBEDDING_COLUMNS[i] for i in top_combined],
            "top_dimensions_spearman": [EMBEDDING_COLUMNS[i] for i in top_spearman],
            "top_dimensions_random_forest": [EMBEDDING_COLUMNS[i] for i in top_rf],
            "top_dimensions_transformer": [
                EMBEDDING_COLUMNS[i] for i in top_transformer
            ],
            "best_spearman_rho": float(rho[top_spearman[0], j]),
            "rf_random_cv_r2": rf_results[variable]["random_cv_r2_mean"],
            "rf_spatial_cv_r2": rf_results[variable]["spatial_cv_r2_mean"],
            "transformer_random_cv_r2": transformer_results[variable][
                "random_cv_r2_mean"
            ],
            "transformer_spatial_cv_r2": transformer_results[variable][
                "spatial_cv_r2_mean"
            ],
        }

    correlations = {
        "spearman_vs_rf": safe_correlation(rho_abs.ravel(), rf.ravel()),
        "spearman_vs_transformer": safe_correlation(
            rho_abs.ravel(), transformer_importance.ravel()
        ),
        "rf_vs_transformer": safe_correlation(
            rf.ravel(), transformer_importance.ravel()
        ),
    }
    output = {
        "method": {
            "description": "three-method dimension dictionary",
            "primary_rule": (
                "Use a >=2 method vote when available; otherwise use the highest "
                "mean normalized support across absolute Spearman, RF permutation "
                "importance and Transformer permutation importance."
            ),
            "robust_rule": "at least two methods agree and temporal stability r >= 0.90",
        },
        "summary": {
            "dimensions": 64,
            "variables": 26,
            "dimensions_two_or_more_methods_agree": int(double_agreement),
            "dimensions_all_three_methods_agree": int(triple_agreement),
            "robust_dimensions": int(robust_count),
            "method_matrix_correlations": correlations,
        },
        "dimensions": dimension_entries,
        "variables": variable_entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    with args.dimensions_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "dimension", "primary_variable", "primary_variable_zh", "category",
            "direction", "spearman_rho", "rf_importance",
            "transformer_importance", "combined_support", "agreement_count",
            "agreeing_methods", "temporal_stability_r", "robust",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dimension, info in dimension_entries.items():
            writer.writerow(
                {
                    key: (
                        dimension
                        if key == "dimension"
                        else ";".join(info[key])
                        if key == "agreeing_methods"
                        else info[key]
                    )
                    for key in fields
                }
            )

    with args.variables_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "variable", "display_name_zh", "category", "unit",
            "top_dimensions_combined", "best_spearman_rho", "rf_random_cv_r2",
            "rf_spatial_cv_r2", "transformer_random_cv_r2",
            "transformer_spatial_cv_r2",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variable, info in variable_entries.items():
            writer.writerow(
                {
                    key: (
                        variable
                        if key == "variable"
                        else ";".join(info[key])
                        if key == "top_dimensions_combined"
                        else info[key]
                    )
                    for key in fields
                }
            )
    print(json.dumps(output["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
