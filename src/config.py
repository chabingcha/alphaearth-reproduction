"""
Configuration constants for the AlphaEarth reproduction pipeline.

All paper-derived parameters, environmental variable definitions,
and path configurations in one place.
"""

from pathlib import Path
import numpy as np

# ── Paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FINAL_DIR = RESULTS_DIR / "final"
RAW_DIR = RESULTS_DIR / "raw_control"
NONLINEAR_DIR = RESULTS_DIR / "nonlinear"
FIGURES_DIR = RESULTS_DIR / "figures"
FAISS_DIR = RESULTS_DIR / "faiss"
REPORT_DIR = ROOT / "report"

# Ensure all output directories exist
for d in [FINAL_DIR, RAW_DIR, NONLINEAR_DIR, FIGURES_DIR, FAISS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Geographic Parameters ────────────────────────────────────────
# GBA (Greater Bay Area) — our reproduction region
GBA = {
    "name": "Greater Bay Area (粤港澳大湾区)",
    "lon_min": 113.0, "lon_max": 115.0,
    "lat_min": 22.0, "lat_max": 24.0,
    "spacing_deg": 0.025,          # ~2.75 km at equator
    "resolution_m": 640,           # Overview level 5
    "description": "2°×2° region encompassing Guangzhou, Shenzhen, Hong Kong, Macau",
}

# CONUS — paper's study area
CONUS = {
    "name": "Continental United States",
    "lon_min": -125.0, "lon_max": -66.5,
    "lat_min": 24.5, "lat_max": 49.5,
    "spacing_deg": 0.025,
    "description": "~8M km², full continental scale",
}

# ── Model Hyperparameters ────────────────────────────────────────
SPEARMAN_N_SAMPLES = 1_000_000        # Paper: n=1M
RF_N_SAMPLES = 700_000                # Paper: n=700K
RF_CV_FOLDS = 5                        # Paper: 5-fold CV
TRANSFORMER_N_SAMPLES = 5_000_000     # Paper: n=5M (we skip this)
N_ESTIMATORS = 100                     # RF trees
RANDOM_SEED = 42

# Spatial CV
BLOCK_SIZE_DEG = 2.0                   # 2°×2° blocks
SPATIAL_CV_FOLDS = 5

# Temporal
YEARS = list(range(2017, 2024))        # 2017–2023
TEMPORAL_TRAIN_YEARS = 5               # Train on 5 years, test on remaining 2

# FAISS
FAISS_NLIST = 256                      # Paper: 3500; ours: 256 for 234K vectors
FAISS_K_NEIGHBORS = 10                 # k-NN for RAG context

# ── Environmental Variable Definitions ────────────────────────────
# Paper's 26 variables across 7 categories
PAPER_VARIABLES = {
    "Terrain": ["elevation", "slope", "aspect", "flow_accumulation"],
    "Soil": ["clay_fraction", "organic_carbon", "ph", "water_capacity"],
    "Vegetation": ["ndvi_mean", "ndvi_max", "evi_mean", "lai_mean", "tree_cover", "albedo"],
    "Temperature": ["lst_daytime", "lst_nighttime", "mean_air_temp", "dew_point_temp"],
    "Climate": ["annual_precip", "max_monthly_precip"],
    "Hydrology": ["soil_moisture", "annual_runoff", "annual_et"],
    "Urban": ["impervious_surface", "nighttime_lights", "population_density"],
}

# GBA reproduction: 21 variables across 5 categories
GBA_VARIABLES = {
    "Climate": ["precip", "solar_rad", "vpd", "wind_speed"],
    "Vegetation": ["ndvi", "evi", "lai", "fvc"],
    "Hydrology": ["soil_moisture_l1", "soil_moisture_l2", "evapotranspiration", "runoff"],
    "Temperature": ["lst_day", "lst_night", "t_air_max", "t_air_min", "t_air_mean"],
    "Terrain": ["elevation", "slope", "aspect", "tpi"],
}

GBA_VAR_LIST = [v for cat in GBA_VARIABLES.values() for v in cat]
GBA_CATEGORIES = list(GBA_VARIABLES.keys())

# Variable display names for plots
VAR_DISPLAY_NAMES = {
    "precip": "Precipitation",
    "solar_rad": "Solar Radiation",
    "vpd": "VPD",
    "wind_speed": "Wind Speed",
    "ndvi": "NDVI",
    "evi": "EVI",
    "lai": "LAI",
    "fvc": "FVC",
    "soil_moisture_l1": "Soil Moisture L1",
    "soil_moisture_l2": "Soil Moisture L2",
    "evapotranspiration": "ET",
    "runoff": "Runoff",
    "lst_day": "LST Day",
    "lst_night": "LST Night",
    "t_air_max": "T_air Max",
    "t_air_min": "T_air Min",
    "t_air_mean": "T_air Mean",
    "elevation": "Elevation",
    "slope": "Slope",
    "aspect": "Aspect",
    "tpi": "TPI",
}

# ── Paper Results (for comparison) ───────────────────────────────
PAPER_RESULTS = {
    "mean_spatial_cv_delta": 0.017,
    "mean_temporal_stability_r": 0.963,
    "method_convergence_pearson_r": 0.45,
    "variables_r2_above_0.90": "12/26",
    "best_r2": 0.97,
    "lsi_overall_score": 3.74,
    "lsi_grounding_score": 3.93,
    "lsi_coherence_score": 4.25,
    "lsi_scores": {
        "Grounding": (3.93, 0.65),
        "Coherence": (4.25, 0.52),
        "Scientific Accuracy": (3.62, 0.78),
        "Completeness": (3.48, 0.82),
        "Practical Utility": (3.42, 0.88),
    },
    # Approximate category-level R² from the paper (CONUS scale)
    "category_r2_conus": {
        "Climate": 0.82,
        "Vegetation": 0.65,
        "Hydrology": 0.73,
        "Temperature": 0.97,
        "Terrain": 0.88,
    },
}

# Known GBA results (from the reproduction report)
GBA_KNOWN_RESULTS = {
    "ridge_r2": {
        "precip": 0.3824, "solar_rad": 0.2186, "vpd": 0.4312, "wind_speed": 0.2680,
        "ndvi": 0.1253, "evi": 0.1387, "lai": 0.0786, "fvc": 0.1319,
        "soil_moisture_l1": 0.2284, "soil_moisture_l2": 0.2150,
        "evapotranspiration": 0.1905, "runoff": 0.4594,
        "lst_day": 0.1429, "lst_night": 0.1247,
        "t_air_max": 0.2634, "t_air_min": 0.2400, "t_air_mean": 0.1718,
        "elevation": 0.0853, "slope": 0.0703, "aspect": 0.0058, "tpi": 0.0083,
    },
    "ridge_r2_debiased": {
        "precip": 0.6200, "solar_rad": 0.6523, "vpd": 0.5785, "wind_speed": 0.6934,
        "ndvi": 0.3214, "evi": 0.3377, "lai": 0.1850, "fvc": 0.3167,
        "soil_moisture_l1": 0.2952, "soil_moisture_l2": 0.2871,
        "evapotranspiration": 0.3313, "runoff": 0.6126,
        "lst_day": 0.2858, "lst_night": 0.4569,
        "t_air_max": 0.4635, "t_air_min": 0.6504, "t_air_mean": 0.6196,
        "elevation": 0.2236, "slope": 0.1352, "aspect": 0.0096, "tpi": 0.0097,
    },
    "rf_test_r2": {
        "precip": -0.9085, "solar_rad": -0.7775, "vpd": -1.3500, "wind_speed": 0.3311,
        "ndvi": 0.3656, "evi": 0.4035, "lai": 0.2801, "fvc": 0.3491,
        "soil_moisture_l1": 0.0562, "soil_moisture_l2": -0.0779,
        "evapotranspiration": 0.2439, "runoff": -0.9787,
        "lst_day": 0.3668, "lst_night": 0.4736,
        "t_air_max": 0.1290, "t_air_min": 0.3835, "t_air_mean": 0.3157,
        "elevation": 0.3242, "slope": 0.2378, "aspect": 0.0334, "tpi": 0.0654,
    },
    "xgb_test_r2": {
        "precip": -0.8966, "solar_rad": -0.7697, "vpd": -1.4659, "wind_speed": 0.3508,
        "ndvi": 0.3557, "evi": 0.3652, "lai": 0.2543, "fvc": 0.3319,
        "soil_moisture_l1": -0.0073, "soil_moisture_l2": -0.1036,
        "evapotranspiration": 0.2049, "runoff": -0.9572,
        "lst_day": 0.2986, "lst_night": 0.4787,
        "t_air_max": 0.1120, "t_air_min": 0.3070, "t_air_mean": 0.3116,
        "elevation": 0.3102, "slope": 0.2046, "aspect": 0.0224, "tpi": 0.0542,
    },
    "category_raw": {
        "Climate": 0.6361, "Vegetation": 0.2902, "Hydrology": 0.3816,
        "Temperature": 0.4952, "Terrain": 0.0945,
    },
    "category_debiased": {
        "Climate": 0.3251, "Vegetation": 0.1186, "Hydrology": 0.2733,
        "Temperature": 0.1886, "Terrain": 0.0424,
    },
    "category_rf": {
        "Climate": -0.6762, "Vegetation": 0.3496, "Hydrology": -0.1891,
        "Temperature": 0.3337, "Terrain": 0.1652,
    },
    "category_xgb": {
        "Climate": -0.6954, "Vegetation": 0.3268, "Hydrology": -0.2158,
        "Temperature": 0.3016, "Terrain": 0.1478,
    },
    "spatial_encoding": {
        "lat_r2": 0.7773,
        "lon_r2": 0.7540,
        "mean_spatial_r2_per_dim": 0.3396,
        "dims_spatial_r2_gt_0_5": "13/64",
        "variance_removed_by_debiasing": 0.721,
        "post_debias_lat_r2": 0.0000,
    },
    "comparison_with_paper": {
        "mean_ridge_r2_paper": 0.97,
        "mean_ridge_r2_ours_raw": 0.385,
        "mean_ridge_r2_ours_debiased": 0.190,
        "best_r2_ours_raw": 0.6934,
        "best_r2_ours_debiased": 0.459,
        "vars_r2_gt_0_9_paper": "12/26",
        "vars_r2_gt_0_9_ours": "0/21",
        "mean_spatial_cv_delta_paper": 0.017,
        "mean_spatial_cv_delta_ours": 1.6063,
        "spatial_cv_factor": 95,
    },
}

# Visualization style — consistent with the existing figure_generation.py
COLORS = {
    'dark':    '#2C3E50',
    'blue':    '#2980B9',
    'green':   '#27AE60',
    'red':     '#C0392B',
    'orange':  '#E67E22',
    'purple':  '#8E44AD',
    'teal':    '#1ABC9C',
    'yellow':  '#F1C40F',
    'gray':    '#7F8C8D',
    'light':   '#ECF0F1',
}
