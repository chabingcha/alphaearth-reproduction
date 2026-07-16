"""
Core interpretability methods for AlphaEarth embedding analysis.

Implements the three complementary approaches from the paper:
  1. Spearman Rank Correlation (linear monotonic)
  2. Random Forest Regression (nonlinear, feature importance)
  3. Ridge Regression (linear baseline)

Also includes XGBoost as an additional nonlinear baseline.

Key finding from GBA reproduction:
  - Raw embeddings: moderate R² (0.39 mean) driven by spatial autocorrelation
  - Debiased embeddings: low R² (0.19 mean) — genuine environmental signal
  - Nonlinear methods: improve stable variables (vegetation 200%, terrain 300%)
    but catastrophically overfit temporally unstable climate variables
"""

import numpy as np
import time
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import warnings

from .config import (
    RANDOM_SEED, SPEARMAN_N_SAMPLES, RF_N_SAMPLES,
    N_ESTIMATORS, RF_CV_FOLDS, GBA_CATEGORIES, GBA_VAR_LIST,
)

rng = np.random.default_rng(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════
# Spearman Rank Correlation
# ══════════════════════════════════════════════════════════════════

def spearman_correlation(embeddings, env_vars, var_names=None,
                         max_samples=SPEARMAN_N_SAMPLES, verbose=True):
    """
    Compute Spearman rank correlation between each embedding dimension
    and each environmental variable.

    Paper approach: n=1M random subsample, 64×26 matrix, p<0.001 for all.

    Args:
        embeddings: (N, D) array
        env_vars: (N, M) array
        var_names: List of variable names
        max_samples: Subsample size (memory constraint for large N)

    Returns:
        rho_matrix: (D, M) Spearman ρ matrix
        p_matrix: (D, M) p-value matrix
        best_dim_per_var: dict mapping var_name → (best_dim, |ρ|)
    """
    N = len(embeddings)
    D = embeddings.shape[1]
    M = env_vars.shape[1]

    # Subsample if needed
    if N > max_samples:
        idx = rng.choice(N, max_samples, replace=False)
        X = embeddings[idx]
        Y = env_vars[idx]
    else:
        X = embeddings
        Y = env_vars
        idx = np.arange(N)

    if verbose:
        print(f"\n  Spearman ρ: {len(idx):,} samples, {D}×{M} matrix")

    # Compute correlations
    rho = np.zeros((D, M))
    pvals = np.zeros((D, M))
    t0 = time.time()

    for d in range(D):
        for m in range(M):
            r, p = spearmanr(X[:, d], Y[:, m])
            rho[d, m] = r if not np.isnan(r) else 0
            pvals[d, m] = p if not np.isnan(p) else 1.0

    elapsed = time.time() - t0
    if verbose:
        n_nonzero = np.sum(np.abs(rho) > 1e-6)
        max_abs = np.max(np.abs(rho))
        print(f"    Computed in {elapsed:.1f}s")
        print(f"    Nonzero correlations: {n_nonzero}/{D*M}")
        print(f"    Max |ρ|: {max_abs:.4f}")
        print(f"    Min p-value: {np.min(pvals):.2e}")

    # Best dimension per variable (max absolute ρ)
    best_dim_per_var = {}
    if var_names:
        for m, name in enumerate(var_names):
            best_d = np.argmax(np.abs(rho[:, m]))
            best_dim_per_var[name] = {
                "dim": int(best_d),
                "rho": float(rho[best_d, m]),
                "p_value": float(pvals[best_d, m]),
            }

    return {
        "rho_matrix": rho,
        "p_matrix": pvals,
        "best_dim_per_var": best_dim_per_var,
        "n_samples": len(idx),
        "time_seconds": elapsed,
    }


# ══════════════════════════════════════════════════════════════════
# Random Forest Regression
# ══════════════════════════════════════════════════════════════════

def random_forest_regression(embeddings, env_vars, var_names=None,
                             n_estimators=N_ESTIMATORS, cv_folds=RF_CV_FOLDS,
                             max_samples=RF_N_SAMPLES, compute_importance=True,
                             verbose=True):
    """
    Train a separate Random Forest for each environmental variable.

    Paper approach: n=700K, 26 separate RFs, 5-fold CV, permutation importance.

    Args:
        embeddings: (N, D) array
        env_vars: (N, M) array
        var_names: Variable names
        n_estimators: Number of trees per forest
        cv_folds: Cross-validation folds
        max_samples: Subsample size
        compute_importance: Compute permutation importance

    Returns:
        dict with cv_r2, test_r2, importance matrices, etc.
    """
    N, D = embeddings.shape
    M = env_vars.shape[1]

    # Subsample
    if N > max_samples:
        idx = rng.choice(N, max_samples, replace=False)
        X = embeddings[idx]
        Y = env_vars[idx]
    else:
        X = embeddings
        Y = env_vars
        idx = np.arange(N)

    if verbose:
        print(f"\n  Random Forest: {len(idx):,} samples, {M} regressors, "
              f"{n_estimators} trees, {cv_folds}-fold CV")

    cv_r2 = np.zeros(M)
    importance = np.zeros((D, M))
    models = {}

    t0 = time.time()
    for m in range(M):
        y = Y[:, m]
        # Filter NaN/Inf
        valid = np.isfinite(y)
        if valid.sum() < 100:
            warnings.warn(f"Variable {var_names[m] if var_names else m}: insufficient valid samples")
            cv_r2[m] = np.nan
            continue

        X_valid, y_valid = X[valid], y[valid]

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

        try:
            # Cross-validation R²
            scores = cross_val_score(rf, X_valid, y_valid, cv=cv_folds,
                                     scoring='r2', n_jobs=-1)
            cv_r2[m] = np.mean(scores)

            # Fit on all data for importance
            if compute_importance:
                rf.fit(X_valid, y_valid)
                # Permutation importance on subset (expensive)
                imp_result = permutation_importance(
                    rf, X_valid[:min(10000, len(X_valid))], y_valid[:min(10000, len(X_valid))],
                    n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1
                )
                importance[:, m] = imp_result.importances_mean

            models[m] = rf
        except Exception as e:
            warnings.warn(f"RF for var {m}: {e}")
            cv_r2[m] = np.nan

    elapsed = time.time() - t0
    if verbose:
        print(f"    Completed in {elapsed:.1f}s")
        valid_r2 = cv_r2[np.isfinite(cv_r2)]
        if len(valid_r2) > 0:
            print(f"    Mean CV R²: {np.mean(valid_r2):.4f}")
            print(f"    Best R²:    {np.max(valid_r2):.4f}")
            print(f"    Worst R²:   {np.min(valid_r2):.4f}")

    # Per-variable results
    per_var = {}
    if var_names:
        for m, name in enumerate(var_names):
            top_dims = np.argsort(np.abs(importance[:, m]))[::-1][:3]
            per_var[name] = {
                "cv_r2": float(cv_r2[m]) if np.isfinite(cv_r2[m]) else None,
                "top_dims": [{"dim": int(d), "importance": float(importance[d, m])}
                            for d in top_dims],
            }

    return {
        "cv_r2": cv_r2.tolist(),
        "importance": importance.tolist(),
        "per_var": per_var,
        "n_samples": len(idx),
        "time_seconds": elapsed,
    }


# ══════════════════════════════════════════════════════════════════
# Ridge Regression (Linear Baseline)
# ══════════════════════════════════════════════════════════════════

def ridge_regression(embeddings, env_vars, var_names=None,
                     cv_folds=RF_CV_FOLDS, verbose=True):
    """
    Ridge regression as a linear baseline for comparison.

    Each environmental variable is predicted from all 64 embedding dimensions.
    This is the simplest model — if it achieves high R², the relationship
    is mostly linear.

    Args:
        embeddings: (N, D) array
        env_vars: (N, M) array
        var_names: Variable names
        cv_folds: Cross-validation folds

    Returns:
        dict with r2 per variable and aggregate stats
    """
    N, D = embeddings.shape
    M = env_vars.shape[1]

    if verbose:
        print(f"\n  Ridge Regression: {N:,} samples, {M} targets")

    r2_values = {}
    coefficients = {}
    t0 = time.time()

    for m in range(M):
        y = env_vars[:, m]
        valid = np.isfinite(y)
        if valid.sum() < 100:
            r2_values[var_names[m] if var_names else str(m)] = np.nan
            continue

        X_valid, y_valid = embeddings[valid], y[valid]

        ridge = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=cv_folds)
        ridge.fit(X_valid, y_valid)

        name = var_names[m] if var_names else str(m)
        r2_values[name] = float(ridge.score(X_valid, y_valid))
        coefficients[name] = ridge.coef_.tolist()

    elapsed = time.time() - t0

    valid_r2 = [v for v in r2_values.values() if np.isfinite(v)]
    category_means = {}
    if var_names:
        for cat, cat_vars in GBA_CATEGORIES.items():
            cat_r2 = [r2_values.get(v, np.nan) for v in cat_vars if v in r2_values]
            cat_r2 = [v for v in cat_r2 if np.isfinite(v)]
            if cat_r2:
                category_means[cat] = float(np.mean(cat_r2))

    if verbose:
        print(f"    Completed in {elapsed:.1f}s")
        print(f"    Mean R²: {np.mean(valid_r2):.4f}")
        print(f"    By category: {', '.join(f'{k}={v:.3f}' for k, v in category_means.items())}")

    return {
        "r2": r2_values,
        "coefficients": coefficients,
        "category_means": category_means,
        "mean_r2": float(np.mean(valid_r2)),
        "time_seconds": elapsed,
    }


# ══════════════════════════════════════════════════════════════════
# XGBoost Regression (Additional Nonlinear Baseline)
# ══════════════════════════════════════════════════════════════════

def xgboost_regression(embeddings, env_vars, var_names=None,
                       cv_folds=RF_CV_FOLDS, max_samples=RF_N_SAMPLES,
                       verbose=True):
    """
    XGBoost as an additional nonlinear baseline.

    Gradient boosting often outperforms Random Forest on tabular data,
    but may be more prone to overfitting on small regional datasets.

    Args:
        embeddings: (N, D)
        env_vars: (N, M)
        var_names: Variable names
        cv_folds: CV folds
        max_samples: Subsample size

    Returns:
        dict with test_r2 per variable
    """
    try:
        import xgboost as xgb
    except ImportError:
        if verbose:
            print("  ⚠ XGBoost not installed — skipping")
        return {"test_r2": {}, "error": "xgboost not installed"}

    N = len(embeddings)
    if N > max_samples:
        idx = rng.choice(N, max_samples, replace=False)
        X = embeddings[idx]
        Y = env_vars[idx]
    else:
        X = embeddings
        Y = env_vars

    if verbose:
        print(f"\n  XGBoost: {len(X):,} samples, {Y.shape[1]} targets")

    M = Y.shape[1]
    test_r2 = {}
    category_means = {}
    t0 = time.time()

    for m in range(M):
        y = Y[:, m]
        valid = np.isfinite(y)
        if valid.sum() < 100:
            continue

        X_valid, y_valid = X[valid], y[valid]

        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_SEED, verbosity=0,
        )

        try:
            scores = cross_val_score(model, X_valid, y_valid, cv=cv_folds,
                                     scoring='r2', n_jobs=-1)
            name = var_names[m] if var_names else str(m)
            test_r2[name] = float(np.mean(scores))
        except Exception as e:
            warnings.warn(f"XGBoost for var {m}: {e}")

    elapsed = time.time() - t0

    # Category means
    if var_names:
        for cat, cat_vars in GBA_CATEGORIES.items():
            cat_r2 = [test_r2.get(v, np.nan) for v in cat_vars if v in test_r2]
            cat_r2 = [v for v in cat_r2 if np.isfinite(v)]
            if cat_r2:
                category_means[cat] = float(np.mean(cat_r2))

    if verbose:
        valid_r2 = [v for v in test_r2.values() if np.isfinite(v)]
        print(f"    Completed in {elapsed:.1f}s")
        if valid_r2:
            print(f"    Mean test R²: {np.mean(valid_r2):.4f}")

    return {
        "test_r2": test_r2,
        "category_means": category_means,
        "time_seconds": elapsed,
    }


# ══════════════════════════════════════════════════════════════════
# Dimension Dictionary
# ══════════════════════════════════════════════════════════════════

def compute_dimension_dictionary(spearman_results, rf_results, ridge_results,
                                  var_names=None, verbose=True):
    """
    Build a dimension dictionary mapping each of the 64 embedding dimensions
    to its most strongly associated environmental variable(s).

    Concordance: A dimension is "concordant" if ≥2 methods agree on its
    primary associated variable.

    Paper reports r=0.45 Pearson correlation between Spearman |ρ| and
    RF permutation importance — moderate convergence.

    Returns:
        dict: dimension_index → {primary_var, methods, concordance}
    """
    if var_names is None:
        var_names = GBA_VAR_LIST

    D = 64
    M = len(var_names)

    # Extract method-specific importance per dimension
    spearman_importance = np.abs(np.array(spearman_results["rho_matrix"]))  # D×M
    rf_importance = np.array(rf_results["importance"])  # D×M

    # Ridge coefficients as importance proxy
    ridge_importance = np.zeros((D, M))
    for m, name in enumerate(var_names):
        if name in ridge_results.get("coefficients", {}):
            ridge_importance[:len(ridge_results["coefficients"][name]), m] = (
                np.abs(ridge_results["coefficients"][name])
            )

    dictionary = {}
    concordance_count = 0

    for d in range(D):
        if d >= spearman_importance.shape[0]:
            break

        # Best variable per method
        best_s = np.argmax(spearman_importance[d, :])
        best_rf = np.argmax(rf_importance[d, :]) if d < rf_importance.shape[0] else best_s
        best_ridge = np.argmax(ridge_importance[d, :]) if d < ridge_importance.shape[0] else best_s

        # Check concordance
        methods_agree = [best_s, best_rf, best_ridge]
        from collections import Counter
        counts = Counter(methods_agree)
        most_common_var, n_agree = counts.most_common(1)[0]
        is_concordant = n_agree >= 2

        if is_concordant:
            concordance_count += 1

        dictionary[f"dim_{d:02d}"] = {
            "primary_var": var_names[most_common_var] if most_common_var < len(var_names) else f"var_{most_common_var}",
            "spearman_best": var_names[best_s] if best_s < len(var_names) else f"var_{best_s}",
            "rf_best": var_names[best_rf] if best_rf < len(var_names) else f"var_{best_rf}",
            "ridge_best": var_names[best_ridge] if best_ridge < len(var_names) else f"var_{best_ridge}",
            "n_methods_agree": int(n_agree),
            "concordant": is_concordant,
            "spearman_rho": float(spearman_importance[d, best_s]),
            "rf_importance": float(rf_importance[d, best_rf]) if d < rf_importance.shape[0] else 0,
        }

    if verbose:
        print(f"\n  Dimension Dictionary:")
        print(f"    Concordant dimensions (≥2 methods): {concordance_count}/{len(dictionary)}")
        print(f"    Concordance rate: {concordance_count/len(dictionary):.1%}")

    return dictionary
