"""
Synthetic Data Generation for AlphaEarth Embedding Interpretability Reproduction

Generates a dataset that mimics the structure of the AlphaEarth paper:
- 64-dimensional embedding vectors with known ground-truth environmental mappings
- Environmental variables with controlled correlation to specific embedding dimensions
- Spatial coordinates for spatial validation

Ground Truth Design:
- 12 of 64 embedding dimensions are "active" (encode specific environmental variables)
- The remaining 52 are "noise" dimensions (uncorrelated with any variable)
- This mirrors the paper's finding that 12/26 variables had R² > 0.90
- We embed known ground-truth mappings so we can evaluate recovery accuracy

Paper reference: Rahman (2026), arXiv:2602.10354v1
"""

import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json

np.random.seed(42)

OUT_DIR = Path(__file__).parent


def generate_dataset(
    n_samples: int = 10000,
    n_embedding_dims: int = 64,
    n_active_dims: int = 12,
    n_env_vars: int = 26,
    noise_level: float = 0.3,
    spatial_range: tuple = ((-125.0, -66.5), (24.5, 49.5)),  # CONUS-like
):
    """
    Generate synthetic dataset mimicking AlphaEarth embedding structure.

    Parameters
    ----------
    n_samples : int
        Number of sample points (paper: 12.1M; we use 10K for rapid reproduction)
    n_embedding_dims : int
        Number of embedding dimensions (paper: 64)
    n_active_dims : int
        Number of dimensions that encode environmental signal (paper: ~12)
    n_env_vars : int
        Number of environmental variables (paper: 26)
    noise_level : float
        Fraction of variance that is noise (0-1)
    spatial_range : tuple
        ((lon_min, lon_max), (lat_min, lat_max)) — CONUS range

    Returns
    -------
    embeddings : ndarray (n_samples, n_embedding_dims)
    env_vars : ndarray (n_samples, n_env_vars)
    coords : ndarray (n_samples, 2) — (lon, lat)
    ground_truth : dict — mapping of active_dim_idx → env_var_idx → true_correlation
    env_var_names : list
    """
    (lon_min, lon_max), (lat_min, lat_max) = spatial_range

    # ── Spatial coordinates (uniform grid with jitter) ──
    lons = np.random.uniform(lon_min, lon_max, n_samples)
    lats = np.random.uniform(lat_min, lat_max, n_samples)
    coords = np.column_stack([lons, lats])

    # ── Environmental variable names (matching paper's 26) ──
    env_var_names = [
        # Terrain (4)
        "elevation", "slope", "aspect", "flow_accumulation",
        # Soil (4)
        "clay_fraction", "organic_carbon", "soil_ph", "water_capacity",
        # Vegetation (6)
        "ndvi_mean", "ndvi_max", "evi_mean", "lai_mean", "tree_cover", "albedo",
        # Temperature (4)
        "lst_daytime", "lst_nighttime", "air_temp_mean", "dew_point_temp",
        # Climate (2)
        "annual_precip", "max_monthly_precip",
        # Hydrology (3)
        "soil_moisture", "annual_runoff", "annual_et",
        # Urban (3)
        "impervious_surface", "nighttime_lights", "population_density",
    ]

    # ── Generate base environmental variables with spatial structure ──
    # Use lat/lon to create realistic spatial patterns
    env_vars = np.zeros((n_samples, n_env_vars))

    # Latitudinal gradients (temperature, vegetation)
    lat_norm = (lats - lat_min) / (lat_max - lat_min)  # 0 to 1

    # Longitudinal gradients (precipitation — east is wetter)
    lon_norm = (lons - lon_min) / (lon_max - lon_min)

    # Elevation-like: combination of lat and lon with nonlinearity
    base_elev = np.sin(lat_norm * np.pi) * np.cos(lon_norm * np.pi * 0.7) * 3000 + \
                np.random.randn(n_samples) * 200

    # ── Fill environmental variables with realistic spatial patterns ──
    # Terrain
    env_vars[:, 0] = base_elev  # elevation
    env_vars[:, 1] = np.abs(np.gradient(base_elev)) * 10 + np.random.randn(n_samples) * 2  # slope
    env_vars[:, 2] = np.sin(lon_norm * 2 * np.pi) * 180 + np.random.randn(n_samples) * 30  # aspect
    env_vars[:, 3] = np.log(np.abs(base_elev) + 1) * 2 + np.random.randn(n_samples) * 1  # flow accumulation

    # Soil
    env_vars[:, 4] = 30 + 20 * lat_norm + np.random.randn(n_samples) * 5  # clay
    env_vars[:, 5] = 5 - 3 * lat_norm + np.random.randn(n_samples) * 1.5  # organic carbon
    env_vars[:, 6] = 6.5 + lat_norm - 0.5 * lon_norm + np.random.randn(n_samples) * 0.5  # pH
    env_vars[:, 7] = 100 + 50 * lat_norm + np.random.randn(n_samples) * 15  # water capacity

    # Vegetation
    ndvi_base = 0.3 + 0.4 * (1 - lat_norm) + 0.2 * lon_norm + np.random.randn(n_samples) * 0.1
    env_vars[:, 8] = ndvi_base  # ndvi_mean
    env_vars[:, 9] = ndvi_base + 0.1 + np.random.randn(n_samples) * 0.05  # ndvi_max
    env_vars[:, 10] = ndvi_base * 0.8 + np.random.randn(n_samples) * 0.08  # evi_mean
    env_vars[:, 11] = ndvi_base * 2 + 1 + np.random.randn(n_samples) * 0.5  # lai_mean
    env_vars[:, 12] = 50 * ndvi_base + np.random.randn(n_samples) * 10  # tree_cover
    env_vars[:, 13] = 0.15 + 0.1 * (1 - ndvi_base) + np.random.randn(n_samples) * 0.03  # albedo

    # Temperature
    temp_base = 25 - 35 * lat_norm + np.random.randn(n_samples) * 3
    env_vars[:, 14] = temp_base + 5 + np.random.randn(n_samples) * 2  # lst_daytime
    env_vars[:, 15] = temp_base - 3 + np.random.randn(n_samples) * 2  # lst_nighttime
    env_vars[:, 16] = temp_base + np.random.randn(n_samples) * 1.5  # air_temp_mean
    env_vars[:, 17] = temp_base - 5 + np.random.randn(n_samples) * 3  # dew_point

    # Climate
    precip_base = 500 + 1500 * lon_norm + 500 * (1 - lat_norm) + np.random.randn(n_samples) * 200
    env_vars[:, 18] = precip_base  # annual_precip
    env_vars[:, 19] = precip_base / 6 + np.random.randn(n_samples) * 30  # max_monthly_precip

    # Hydrology
    env_vars[:, 20] = 0.3 + 0.2 * lat_norm + 0.1 * lon_norm + np.random.randn(n_samples) * 0.05  # soil_moisture
    env_vars[:, 21] = precip_base * 0.3 + np.random.randn(n_samples) * 50  # annual_runoff
    env_vars[:, 22] = temp_base * 30 + 500 + np.random.randn(n_samples) * 100  # annual_et

    # Urban
    env_vars[:, 23] = np.exp(-((lon_norm - 0.7)**2 + (lat_norm - 0.45)**2) / 0.02) * 80 + \
                      np.random.randn(n_samples) * 5  # impervious
    env_vars[:, 24] = np.exp(-((lon_norm - 0.7)**2 + (lat_norm - 0.45)**2) / 0.05) * 50 + \
                      np.random.randn(n_samples) * 3  # nighttime_lights
    env_vars[:, 25] = np.exp(-((lon_norm - 0.7)**2 + (lat_norm - 0.45)**2) / 0.03) * 5000 + \
                      np.random.randn(n_samples) * 200  # population_density

    # ── Standardize environmental variables ──
    scaler = StandardScaler()
    env_vars_scaled = scaler.fit_transform(env_vars)

    # ── Generate embedding dimensions ──
    embeddings = np.zeros((n_samples, n_embedding_dims))

    # Randomly choose which dimensions are "active" (encode environmental signal)
    active_dim_indices = np.random.choice(n_embedding_dims, n_active_dims, replace=False)
    active_dim_indices.sort()

    # Randomly assign each active dim to one environmental variable
    # (some vars get multiple dims, some get none — as in the paper)
    assigned_vars = np.random.choice(n_env_vars, n_active_dims, replace=True)

    ground_truth = {}
    for i, (dim_idx, var_idx) in enumerate(zip(active_dim_indices, assigned_vars)):
        # Correlation strength: 0.5–0.95 (paper reports strongest |ρ| > 0.8)
        true_corr = np.random.uniform(0.5, 0.95)

        # Create embedding dimension with known correlation to env var
        signal = env_vars_scaled[:, var_idx]
        noise = np.random.randn(n_samples)
        # Blend to achieve desired correlation
        alpha = true_corr
        embeddings[:, dim_idx] = alpha * signal + np.sqrt(1 - alpha**2) * noise

        ground_truth[int(dim_idx)] = {
            "env_var_idx": int(var_idx),
            "env_var_name": env_var_names[var_idx],
            "true_correlation": float(true_corr),
        }

    # Fill remaining (non-active) dimensions with pure noise
    inactive_mask = np.ones(n_embedding_dims, dtype=bool)
    inactive_mask[active_dim_indices] = False
    inactive_indices = np.where(inactive_mask)[0]
    embeddings[:, inactive_indices] = np.random.randn(n_samples, len(inactive_indices))

    # Add small noise to all dimensions (measurement noise)
    embeddings += np.random.randn(n_samples, n_embedding_dims) * noise_level * 0.2

    # ── Standardize embeddings ──
    embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)

    print(f"Generated synthetic dataset:")
    print(f"  Samples: {n_samples}")
    print(f"  Embedding dimensions: {n_embedding_dims}")
    print(f"  Active dimensions: {n_active_dims} (indices: {active_dim_indices.tolist()})")
    print(f"  Environmental variables: {n_env_vars}")
    print(f"  Ground truth mappings: {len(ground_truth)} dimensions → variables")
    print(f"  Noise level: {noise_level}")

    return embeddings, env_vars_scaled, coords, ground_truth, env_var_names, active_dim_indices


def save_dataset(embeddings, env_vars, coords, ground_truth, env_var_names, active_dims):
    """Save generated dataset to disk."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUT_DIR / "embeddings.npy", embeddings)
    np.save(OUT_DIR / "env_vars.npy", env_vars)
    np.save(OUT_DIR / "coords.npy", coords)
    np.save(OUT_DIR / "active_dims.npy", active_dims)

    with open(OUT_DIR / "ground_truth.json", "w") as f:
        json.dump({
            "mappings": {str(k): v for k, v in ground_truth.items()},
            "env_var_names": env_var_names,
            "n_samples": int(embeddings.shape[0]),
            "n_embedding_dims": int(embeddings.shape[1]),
            "n_env_vars": int(env_vars.shape[1]),
        }, f, indent=2)

    print(f"\nDataset saved to: {OUT_DIR.resolve()}")
    print(f"  embeddings.npy: {embeddings.shape}")
    print(f"  env_vars.npy: {env_vars.shape}")
    print(f"  coords.npy: {coords.shape}")
    print(f"  ground_truth.json: {len(ground_truth)} mappings")


if __name__ == "__main__":
    embeddings, env_vars, coords, ground_truth, env_var_names, active_dims = \
        generate_dataset(n_samples=10000)
    save_dataset(embeddings, env_vars, coords, ground_truth, env_var_names, active_dims)
