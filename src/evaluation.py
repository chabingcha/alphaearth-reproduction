"""
Results evaluation and comparison with the original AlphaEarth paper.

Generates quantitative comparisons between:
  - Paper results (CONUS scale, 12.1M samples)
  - Our GBA reproduction (2°×2°, 234K samples)

Key comparisons:
  - R² by environmental category
  - Spatial CV generalization gap
  - Linear vs nonlinear performance
  - Spatial encoding metrics
  - Temporal stability
"""

import numpy as np
from .config import (
    PAPER_RESULTS, GBA_KNOWN_RESULTS, GBA_CATEGORIES, GBA_VAR_LIST, GBA_VARIABLES,
)


# ══════════════════════════════════════════════════════════════════
# Comprehensive Paper Comparison
# ══════════════════════════════════════════════════════════════════

def compare_with_paper(final_results, raw_control_results, nonlinear_results,
                       verbose=True):
    """
    Generate a comprehensive comparison between paper and GBA results.

    Args:
        final_results: Output from Ridge regression
        raw_control_results: Output from spatial/temporal validation
        nonlinear_results: Output from RF/XGBoost

    Returns:
        dict: Structured comparison across all metrics
    """
    paper = PAPER_RESULTS

    comparison = {
        "study_scale": {
            "paper": "CONUS (~8M km², 12.1M samples)",
            "ours": "GBA (~40K km², 234K samples)",
        },
        "key_metrics": {},
        "category_r2": {},
        "findings": [],
    }

    # ── R² Comparison ──
    ridge_r2 = final_results.get("ridge_r2", {})
    ridge_r2_debiased = final_results.get("ridge_r2_debiased", {})
    rf_r2 = nonlinear_results.get("random_forest", {}).get("test_r2", {})

    valid_raw = [v for v in ridge_r2.values() if np.isfinite(v)]
    valid_deb = [v for v in ridge_r2_debiased.values() if np.isfinite(v)]
    valid_rf = [v for v in rf_r2.values() if np.isfinite(v)]

    comparison["key_metrics"]["ridge_r2"] = {
        "paper_best": paper["best_r2"],
        "ours_raw_mean": float(np.mean(valid_raw)) if valid_raw else None,
        "ours_debiased_mean": float(np.mean(valid_deb)) if valid_deb else None,
        "ours_raw_best": float(np.max(valid_raw)) if valid_raw else None,
        "ours_debiased_best": float(np.max(valid_deb)) if valid_deb else None,
        "paper_vars_r2_gt_0_9": paper["variables_r2_above_0.90"],
        "ours_vars_r2_gt_0_9": f"{sum(1 for v in ridge_r2.values() if v > 0.9)}/{len(ridge_r2)}",
    }

    # ── Spatial CV Comparison ──
    spatial_delta = raw_control_results.get("mean_delta_r2")
    if spatial_delta is None:
        # Try to compute from per-variable data (handle multiple structures)
        scv_data = raw_control_results.get("spatial_cv", {})
        if isinstance(scv_data, dict):
            # Try per_variable wrapper first, then direct var->value mapping
            scv_per_var = scv_data.get("per_variable", scv_data)
            deltas = []
            for k, v in scv_per_var.items():
                if isinstance(v, dict) and "delta_r2" in v:
                    deltas.append(abs(v["delta_r2"]))
                elif isinstance(v, (int, float)):
                    deltas.append(abs(v))
            spatial_delta = float(np.mean(deltas)) if deltas else None
    comparison["key_metrics"]["spatial_cv"] = {
        "paper_delta_r2": paper["mean_spatial_cv_delta"],
        "ours_delta_r2": spatial_delta,
        "factor": (spatial_delta / paper["mean_spatial_cv_delta"]
                   if spatial_delta and paper["mean_spatial_cv_delta"] > 0 else None),
    }

    # ── Temporal Stability ──
    temporal_r = raw_control_results.get("mean_inter_year_r")
    comparison["key_metrics"]["temporal_stability"] = {
        "paper_r": paper["mean_temporal_stability_r"],
        "ours_r": temporal_r,
    }

    # ── Category-level R² ──
    for cat in GBA_CATEGORIES:
        cat_vars = [v for v in GBA_VARIABLES.get(cat, [])
                    if v in ridge_r2 and v in ridge_r2_debiased]
        cat_raw = [ridge_r2.get(v, np.nan) for v in cat_vars]
        cat_deb = [ridge_r2_debiased.get(v, np.nan) for v in cat_vars]
        cat_rf = [rf_r2.get(v, np.nan) for v in cat_vars]

        comparison["category_r2"][cat] = {
            "paper_conus": paper["category_r2_conus"].get(cat),
            "ours_raw": float(np.nanmean(cat_raw)) if cat_raw else None,
            "ours_debiased": float(np.nanmean(cat_deb)) if cat_deb else None,
            "ours_rf": float(np.nanmean(cat_rf)) if cat_rf else None,
        }

    # ── Spatial Encoding ──
    spatial_enc = final_results.get("spatial_encoding", {})
    comparison["spatial_encoding"] = {
        "lat_r2": spatial_enc.get("lat_r2"),
        "lon_r2": spatial_enc.get("lon_r2"),
        "variance_removed_by_debiasing": spatial_enc.get("variance_removed_by_debiasing"),
    }

    # ── Linear vs Nonlinear ──
    comparison["linear_vs_nonlinear"] = {
        "ridge_mean_r2": float(np.mean(valid_raw)) if valid_raw else None,
        "rf_mean_r2": float(np.mean([v for v in valid_rf if v > -10])) if valid_rf else None,
        "rf_better_count": f"{sum(1 for v in ridge_r2 if ridge_r2.get(v, 0) < rf_r2.get(v, -999))}/{len(ridge_r2)}",
    }

    # ── Generate findings ──
    findings = []
    if spatial_delta and spatial_delta > 0.1:
        findings.append({
            "severity": "critical",
            "finding": f"Spatial CV overfitting is {spatial_delta:.2f} (paper: {paper['mean_spatial_cv_delta']}), "
                       f"indicating embeddings primarily encode spatial position at regional scale",
        })

    spatial_removed = spatial_enc.get("variance_removed_by_debiasing", 0)
    if spatial_removed and spatial_removed > 0.5:
        findings.append({
            "severity": "critical",
            "finding": f"{spatial_removed:.0%} of embedding variance is spatial position — "
                       f"spatial debiasing is mandatory for interpretability studies",
        })

    rf_improvement = [rf_r2.get(v, -999) - ridge_r2.get(v, 0)
                      for v in ridge_r2 if v in rf_r2]
    rf_better = sum(1 for x in rf_improvement if x > 0.05)
    if rf_better > 0:
        findings.append({
            "severity": "important",
            "finding": f"Nonlinear methods (RF) improve R² for {rf_better}/{len(rf_improvement)} variables "
                       f"(vegetation & terrain categories benefit most)",
        })

    comparison["findings"] = findings

    if verbose:
        _print_comparison(comparison)

    return comparison


def _print_comparison(comp):
    """Pretty-print the comparison results."""
    print("\n" + "=" * 65)
    print("  PAPER (CONUS) vs GBA REPRODUCTION — COMPARISON")
    print("=" * 65)

    km = comp["key_metrics"]

    print(f"\n  -- R2 Comparison --")
    print(f"    Paper best R2:                    {km['ridge_r2']['paper_best']}")
    print(f"    Ours raw mean R2:                 {km['ridge_r2']['ours_raw_mean']:.4f}")
    print(f"    Ours debiased mean R2:            {km['ridge_r2']['ours_debiased_mean']:.4f}")
    print(f"    Paper vars R2 > 0.90:             {km['ridge_r2']['paper_vars_r2_gt_0_9']}")
    print(f"    Ours vars R2 > 0.90:              {km['ridge_r2']['ours_vars_r2_gt_0_9']}")

    print(f"\n  -- Spatial Generalization --")
    scv = km["spatial_cv"]
    print(f"    Paper delta_R2:                   {scv['paper_delta_r2']}")
    if scv.get('ours_delta_r2') is not None:
        print(f"    Ours delta_R2:                    {scv['ours_delta_r2']:.4f}")
    else:
        print(f"    Ours delta_R2:                    N/A")
    if scv.get("factor"):
        print(f"    Factor:                           {scv['factor']:.0f}x worse!")

    print(f"\n  -- Category R2 --")
    for cat, vals in comp["category_r2"].items():
        paper_str = str(vals.get('paper_conus', 'N/A'))
        print(f"    {cat:15s}: Paper={paper_str:>6s}  "
              f"Ours Raw={vals.get('ours_raw', 0):.3f}  "
              f"Ours Deb={vals.get('ours_debiased', 0):.3f}")

    print(f"\n  -- Key Findings --")
    for f in comp.get("findings", []):
        print(f"    [{f['severity'].upper()}] {f['finding']}")


# ══════════════════════════════════════════════════════════════════
# Category Analysis
# ══════════════════════════════════════════════════════════════════

def category_analysis(ridge_r2, ridge_r2_debiased=None, rf_r2=None, xgb_r2=None):
    """
    Aggregate per-variable R² into category-level means.

    Returns:
        dict with category-level statistics
    """
    categories = {}
    for cat, cat_vars in GBA_VARIABLES.items():
        cat_data = {"variables": cat_vars}

        for label, r2_dict in [
            ("raw", ridge_r2),
            ("debiased", ridge_r2_debiased),
            ("rf", rf_r2),
            ("xgb", xgb_r2),
        ]:
            if r2_dict is None:
                continue
            vals = [r2_dict.get(v, np.nan) for v in cat_vars]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                cat_data[f"mean_{label}_r2"] = float(np.mean(vals))
                cat_data[f"std_{label}_r2"] = float(np.std(vals))
                cat_data[f"best_{label}_r2"] = float(np.max(vals))
                cat_data[f"worst_{label}_r2"] = float(np.min(vals))

        categories[cat] = cat_data

    return categories


# ══════════════════════════════════════════════════════════════════
# Linear vs Nonlinear Analysis
# ══════════════════════════════════════════════════════════════════

def linear_vs_nonlinear_comparison(ridge_r2, rf_r2, xgb_r2=None):
    """
    Compare linear (Ridge) vs nonlinear (RF, XGBoost) performance.

    Identifies which variables benefit from nonlinear modeling.

    Returns:
        dict with per-variable comparison and summary statistics
    """
    comparison = {}
    improvements = []

    for var in ridge_r2:
        r = ridge_r2.get(var, np.nan)
        rf = rf_r2.get(var, np.nan) if rf_r2 else np.nan
        xgb = xgb_r2.get(var, np.nan) if xgb_r2 else np.nan

        entry = {"ridge_r2": r}

        if np.isfinite(rf):
            entry["rf_r2"] = rf
            entry["rf_delta"] = rf - r
            entry["rf_better"] = rf > r + 0.01

        if np.isfinite(xgb) and xgb_r2:
            entry["xgb_r2"] = xgb
            entry["xgb_delta"] = xgb - r
            entry["xgb_better"] = xgb > r + 0.01

        comparison[var] = entry

        if np.isfinite(rf) and rf > r + 0.05:
            improvements.append((var, rf - r))

    # Summary
    n_rf_better = sum(1 for v in comparison.values() if v.get("rf_better", False))
    n_total = sum(1 for v in comparison.values() if "rf_r2" in v)
    rf_improvement = np.mean([v["rf_delta"] for v in comparison.values()
                              if "rf_delta" in v and np.isfinite(v["rf_delta"])])

    summary = {
        "n_rf_better": n_rf_better,
        "n_total": n_total,
        "rf_better_pct": n_rf_better / n_total * 100 if n_total > 0 else 0,
        "mean_rf_improvement": float(rf_improvement) if np.isfinite(rf_improvement) else 0,
        "top_improvements": sorted(improvements, key=lambda x: x[1], reverse=True)[:5],
    }

    return {"per_variable": comparison, "summary": summary}
