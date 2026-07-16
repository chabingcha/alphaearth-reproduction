"""
Data preprocessing for the AlphaEarth reproduction pipeline.

Key operations:
  - Spatial debiasing: Remove lat/lon signal from each embedding dimension
  - Train/test splitting: Random, spatial block, and temporal strategies
  - Normalization: L2 normalization for FAISS cosine similarity
  - Variable alignment: Match embeddings to environmental variables
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import RidgeCV
import warnings

from .config import RANDOM_SEED, BLOCK_SIZE_DEG, SPATIAL_CV_FOLDS, YEARS

rng = np.random.default_rng(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════
# Spatial Debiasing
# ══════════════════════════════════════════════════════════════════

def spatial_debiasing(embeddings, coords, degree=2, verbose=True):
    """
    Remove spatial position signal from each embedding dimension.

    For each dimension d:
        emb_d ≈ f(lat, lon) + residual
    where f is a polynomial of given degree. The residual is the
    spatially-debiased embedding.

    This is the KEY methodological contribution of our reproduction:
    without debiasing, spatial autocorrelation inflates R² by 2-10×.

    Args:
        embeddings: (N, D) array of embedding vectors
        coords: (N, 2) array of [lat, lon] coordinates
        degree: Polynomial degree for spatial regression
        verbose: Print debiasing statistics

    Returns:
        debiased_embeddings: (N, D) residuals after spatial regression
        stats: dict with per-dimension and aggregate spatial R²
    """
    N, D = embeddings.shape
    lat, lon = coords[:, 0], coords[:, 1]

    # Build polynomial spatial features
    spatial_features = [np.ones(N)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            spatial_features.append((lat ** (d - i)) * (lon ** i))
    X_spatial = np.column_stack(spatial_features)

    # Debiase each dimension
    debiased = np.zeros_like(embeddings, dtype=np.float32)
    spatial_r2_per_dim = np.zeros(D)
    all_residuals = np.zeros(N)
    all_original = np.zeros(N)

    for d in range(D):
        y = embeddings[:, d]
        # Ridge regression to predict embedding dim from spatial position
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
        model.fit(X_spatial, y)
        y_pred = model.predict(X_spatial)
        residual = y - y_pred

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        spatial_r2_per_dim[d] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        debiased[:, d] = residual.astype(np.float32)
        all_residuals += residual ** 2
        all_original += y ** 2

    # Aggregate statistics
    total_variance = np.sum(all_original)
    residual_variance = np.sum(all_residuals)
    variance_removed = 1 - residual_variance / total_variance if total_variance > 0 else 0

    dims_high_spatial = np.sum(spatial_r2_per_dim > 0.5)
    mean_spatial_r2 = np.mean(spatial_r2_per_dim)

    stats = {
        "per_dim_spatial_r2": spatial_r2_per_dim.tolist(),
        "mean_spatial_r2": float(mean_spatial_r2),
        "dims_spatial_r2_gt_0_5": f"{dims_high_spatial}/{D}",
        "variance_removed_by_debiasing": float(variance_removed),
        "degree": degree,
        "n_spatial_features": X_spatial.shape[1],
    }

    if verbose:
        print(f"\n  Spatial Debiasing Results (degree={degree}):")
        print(f"    Mean spatial R² per dimension: {mean_spatial_r2:.4f}")
        print(f"    Dims with spatial R² > 0.5: {dims_high_spatial}/{D} ({dims_high_spatial/D*100:.0f}%)")
        print(f"    Total variance removed: {variance_removed:.1%}")
        print(f"    Residual variance (environmental signal): {(1-variance_removed):.1%}")

    return debiased, stats


def verify_debiasing(embeddings_debiased, coords, verbose=True):
    """
    Verify that spatial signal has been removed.

    Checks that lat/lon can no longer predict embedding dimensions.

    Returns:
        dict with post-debiasing lat/lon R²
    """
    lat, lon = coords[:, 0], coords[:, 1]
    D = embeddings_debiased.shape[1]

    # Try to predict lat from debiased embeddings
    model_lat = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model_lat.fit(embeddings_debiased, lat)
    lat_r2 = model_lat.score(embeddings_debiased, lat)

    model_lon = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model_lon.fit(embeddings_debiased, lon)
    lon_r2 = model_lon.score(embeddings_debiased, lon)

    if verbose:
        print(f"\n  Debiasing Verification:")
        print(f"    Post-debias latitude predictability:  R² = {lat_r2:.6f} (should be ≈ 0)")
        print(f"    Post-debias longitude predictability: R² = {lon_r2:.6f} (should be ≈ 0)")
        if lat_r2 < 0.01 and lon_r2 < 0.01:
            print(f"    ✓ Spatial signal successfully removed")
        else:
            print(f"    ⚠ Residual spatial signal detected — consider higher polynomial degree")

    return {"post_debias_lat_r2": float(lat_r2), "post_debias_lon_r2": float(lon_r2)}


# ══════════════════════════════════════════════════════════════════
# Train/Test Splits
# ══════════════════════════════════════════════════════════════════

def random_split(embeddings, env_vars, coords=None, test_size=0.2):
    """Standard random train/test split."""
    N = len(embeddings)
    indices = rng.permutation(N)
    split_idx = int(N * (1 - test_size))
    return indices[:split_idx], indices[split_idx:]


def spatial_block_split(coords, n_folds=SPATIAL_CV_FOLDS, block_size=BLOCK_SIZE_DEG):
    """
    Split by spatial blocks for spatial cross-validation.

    Partitions the study area into block_size° × block_size° blocks,
    then assigns blocks to folds. This ensures samples in the same
    block go to the same fold — forcing the model to predict held-out
    REGIONS, not held-out POINTS.

    Args:
        coords: (N, 2) [lat, lon]
        n_folds: Number of CV folds
        block_size: Block size in degrees

    Returns:
        List of (train_indices, test_indices) tuples for each fold
    """
    lat, lon = coords[:, 0], coords[:, 1]

    # Assign each sample to a spatial block
    lat_block = np.floor((lat - lat.min()) / block_size).astype(int)
    lon_block = np.floor((lon - lon.min()) / block_size).astype(int)

    # Combine into unique block IDs
    block_ids = lat_block * (lon_block.max() + 1) + lon_block
    unique_blocks = np.unique(block_ids)

    if len(unique_blocks) < n_folds:
        warnings.warn(
            f"Only {len(unique_blocks)} spatial blocks available for {n_folds} folds. "
            f"Using {len(unique_blocks)} folds instead."
        )
        n_folds = max(2, len(unique_blocks))

    # GroupKFold on block IDs
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(np.arange(len(coords)), groups=block_ids))

    return splits, block_ids


def temporal_split(years, train_years=None, test_years=None):
    """
    Temporal train/test split — train on some years, test on others.

    This is the stricter validation: can embeddings predict environmental
    variables in years they haven't seen?

    Args:
        years: (N,) year labels
        train_years: List of years for training (default: first 5 of 7)
        test_years: List of years for testing (default: last 2 of 7)

    Returns:
        train_indices, test_indices
    """
    if train_years is None:
        train_years = YEARS[:5]  # 2017-2021
    if test_years is None:
        test_years = YEARS[5:]   # 2022-2023

    train_idx = np.where(np.isin(years, train_years))[0]
    test_idx = np.where(np.isin(years, test_years))[0]

    return train_idx, test_idx


# ══════════════════════════════════════════════════════════════════
# Normalization & Utility
# ══════════════════════════════════════════════════════════════════

def normalize_embeddings(embeddings, method="l2"):
    """
    Normalize embeddings for FAISS similarity search.

    Args:
        embeddings: (N, D)
        method: "l2" for L2 normalization, "standard" for z-score

    Returns:
        normalized embeddings
    """
    if method == "l2":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        return embeddings / norms
    elif method == "standard":
        scaler = StandardScaler()
        return scaler.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def align_embeddings_env(embeddings, coords, env_vars, var_names):
    """
    Verify and align embeddings with environmental variables.

    Ensures consistent shapes and returns aligned data dict.
    """
    N = len(embeddings)
    assert len(coords) == N, f"Coord mismatch: {len(coords)} vs {N}"
    assert len(env_vars) == N, f"Env var mismatch: {len(env_vars)} vs {N}"

    return {
        "embeddings": embeddings,
        "coords": coords,
        "env_vars": env_vars,
        "var_names": var_names,
        "n_samples": N,
        "n_dims": embeddings.shape[1],
        "n_vars": len(var_names),
    }


def compute_spatial_encoding_metrics(embeddings, coords, verbose=True):
    """
    Measure how strongly embeddings encode spatial position.

    Trains a Ridge model: (lat, lon) ← f(embeddings)
    to quantify spatial information leakage.

    Returns:
        dict with lat_r2, lon_r2, and per-dimension metrics
    """
    lat, lon = coords[:, 0], coords[:, 1]
    D = embeddings.shape[1]

    # Predict lat/lon from embeddings
    model_lat = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model_lat.fit(embeddings, lat)
    lat_r2 = model_lat.score(embeddings, lat)

    model_lon = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model_lon.fit(embeddings, lon)
    lon_r2 = model_lon.score(embeddings, lon)

    if verbose:
        print(f"\n  Spatial Encoding Analysis (raw embeddings):")
        print(f"    Latitude predictability:  R² = {lat_r2:.4f}")
        print(f"    Longitude predictability: R² = {lon_r2:.4f}")
        print(f"    Mean spatial R²:           {(lat_r2 + lon_r2)/2:.4f}")

    return {
        "lat_r2": float(lat_r2),
        "lon_r2": float(lon_r2),
        "mean_spatial_r2": float((lat_r2 + lon_r2) / 2),
    }
