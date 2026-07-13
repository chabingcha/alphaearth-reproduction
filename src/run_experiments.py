"""
Main experimental pipeline orchestrator for the AlphaEarth reproduction.

Executes the complete 7-phase pipeline:
  Phase 1: Data Loading (synthetic or real)
  Phase 2: Preprocessing & Spatial Debiasing
  Phase 3: Interpretability Analysis (Spearman + Ridge + RF + XGBoost)
  Phase 4: Validation (Spatial CV, Temporal CV, Stability)
  Phase 5: FAISS Index + RAG Demo
  Phase 6: Evaluation & Paper Comparison
  Phase 7: Visualization Generation

Usage:
  python -m src.run_experiments              # Full pipeline with synthetic data
  python -m src.run_experiments --real       # Use real data if available
  python -m src.run_experiments --quick      # Fast mode: skip heavy computations
  python -m src.run_experiments --known      # Use known results (fastest, for demo)
"""

import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path

# Fix GBK encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add parent to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    YEARS, GBA_VAR_LIST, GBA_CATEGORIES, GBA_VARIABLES,
    FINAL_DIR, RAW_DIR, NONLINEAR_DIR, FAISS_DIR, FIGURES_DIR,
    RANDOM_SEED,
)
from src.data_loader import (
    load_real_data, generate_synthetic_data, load_known_results, save_results,
)
from src.preprocessing import (
    spatial_debiasing, verify_debiasing, normalize_embeddings,
    compute_spatial_encoding_metrics, align_embeddings_env,
)
from src.interpretability import (
    spearman_correlation, random_forest_regression, ridge_regression,
    xgboost_regression, compute_dimension_dictionary,
)
from src.validation import (
    spatial_block_cv, temporal_split_validation,
    compute_temporal_stability, analyze_spatial_encoding,
)
from src.faiss_rag import build_faiss_index, run_demo_queries, save_faiss_index
from src.evaluation import compare_with_paper, category_analysis, linear_vs_nonlinear_comparison
from src.visualization import generate_all_figures


# ══════════════════════════════════════════════════════════════════
def run_full_pipeline(args):
    """Execute the complete 7-phase experimental pipeline."""
    t_start = time.time()

    print("=" * 70)
    print("  AlphaEarth Reproduction — Full Experimental Pipeline")
    print("  Target: GBA region (113°E–115°E, 22°N–24°N)")
    print("=" * 70)

    # ── Phase 1: Data Loading ────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PHASE 1: Data Loading")
    print(f"{'─'*60}")

    if args.real:
        try:
            data = load_real_data()
            data_source = "real"
        except FileNotFoundError:
            print("  Real data not found. Falling back to synthetic data.")
            data = generate_synthetic_data(region="GBA", output_path="data/gba_synthetic.parquet")
            data_source = "synthetic (fallback)"
    else:
        data = generate_synthetic_data(
            region="GBA",
            spatial_signal_strength=0.72,
            output_path="data/gba_synthetic.parquet" if args.save_data else None,
        )
        data_source = "synthetic"

    print(f"  Data source: {data_source}")
    print(f"  Samples: {len(data['embeddings']):,}")
    print(f"  Embedding dims: {data['embeddings'].shape[1]}")
    print(f"  Environmental vars: {len(data['var_names'])}")

    embeddings = data["embeddings"]
    env_vars = data["env_vars"]
    coords = data["coords"]
    years = data["years"]
    var_names = data["var_names"]

    # ── Phase 2: Preprocessing ───────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PHASE 2: Preprocessing & Spatial Debiasing")
    print(f"{'─'*60}")

    # Spatial encoding analysis (before debiasing)
    spatial_enc_raw = compute_spatial_encoding_metrics(embeddings, coords)

    # Spatial debiasing
    embeddings_debiased, debias_stats = spatial_debiasing(embeddings, coords, degree=2)
    verify_debiasing(embeddings_debiased, coords)

    # Normalize for FAISS
    embeddings_norm = normalize_embeddings(embeddings, method="l2")
    embeddings_debiased_norm = normalize_embeddings(embeddings_debiased, method="l2")

    print(f"\n  Preprocessing complete:")
    print(f"    Raw embeddings:      {embeddings.shape}")
    print(f"    Debiased embeddings: {embeddings_debiased.shape}")
    print(f"    Normalized (L2):     {embeddings_norm.shape}")

    # ── Phase 3: Interpretability Analysis ───────────────────────
    print(f"\n{'─'*60}")
    print("  PHASE 3: Interpretability Analysis")
    print(f"{'─'*60}")

    # 3a: Spearman Correlation
    spearman_results = spearman_correlation(embeddings, env_vars, var_names,
                                            max_samples=200000 if args.quick else 1000000)

    # 3b: Ridge Regression (raw and debiased)
    ridge_raw = ridge_regression(embeddings, env_vars, var_names)
    ridge_debiased = ridge_regression(embeddings_debiased, env_vars, var_names)

    # 3c: Random Forest Regression
    if not args.quick:
        rf_results = random_forest_regression(
            embeddings, env_vars, var_names,
            max_samples=50000 if args.quick else 200000,
            n_estimators=30 if args.quick else 100,
        )
    else:
        rf_results = {"cv_r2": [], "importance": [], "per_var": {}, "error": "quick mode skipped"}

    # 3d: XGBoost (optional)
    try:
        xgb_results = xgboost_regression(
            embeddings, env_vars, var_names,
            max_samples=30000 if args.quick else 100000,
        )
    except Exception as e:
        print(f"  ⚠ XGBoost failed: {e}")
        xgb_results = {"test_r2": {}, "error": str(e)}

    # 3e: Dimension Dictionary
    if not args.quick and "importance" in rf_results:
        dim_dict = compute_dimension_dictionary(
            spearman_results, rf_results, ridge_raw, var_names
        )
    else:
        dim_dict = {}

    # ── Phase 4: Validation ──────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PHASE 4: Validation")
    print(f"{'─'*60}")

    # 4a: Spatial Block CV
    spatial_cv_results = spatial_block_cv(
        embeddings, env_vars, coords, var_names,
        model_type="ridge",
    )

    # 4b: Temporal Split Validation
    temporal_cv_results = temporal_split_validation(
        embeddings, env_vars, years, var_names,
        train_years=YEARS[:5], test_years=YEARS[5:],
        model_type="ridge",
    )

    # 4c: Temporal Stability
    temporal_stability = compute_temporal_stability(
        embeddings, env_vars, years, var_names,
        n_subsample=50000 if args.quick else 300000,
    )

    # 4d: Spatial encoding of debiased
    spatial_enc_debiased = analyze_spatial_encoding(embeddings_debiased, coords)

    # ── Phase 5: FAISS Index + RAG Demo ──────────────────────────
    print(f"\n{'─'*60}")
    print("  PHASE 5: FAISS Index & RAG Demo")
    print(f"{'─'*60}")

    # Build indices
    index_raw, idx_meta = build_faiss_index(embeddings_norm, index_type="IVFFlat")
    index_deb, idx_deb_meta = build_faiss_index(embeddings_debiased_norm, index_type="IVFFlat")

    # Save indices
    if index_raw is not None:
        save_faiss_index(index_raw, "gba_raw")
    if index_deb is not None:
        save_faiss_index(index_deb, "gba_debiased")

    # Demo queries
    if index_raw is not None:
        demo_results = run_demo_queries(
            index_raw, embeddings, env_vars, coords, var_names
        )
    else:
        demo_results = {"error": "FAISS not available"}

    # ── Phase 6: Evaluation & Comparison ─────────────────────────
    print(f"\n{'─'*60}")
    print("  PHASE 6: Evaluation & Paper Comparison")
    print(f"{'─'*60}")

    # Assemble final results
    final_results = {
        "ridge_r2": ridge_raw["r2"],
        "ridge_r2_debiased": ridge_debiased["r2"],
        "ridge_category_means": ridge_raw["category_means"],
        "ridge_debiased_category_means": ridge_debiased["category_means"],
        "spatial_encoding_raw": spatial_enc_raw,
        "spatial_encoding_debiased": spatial_enc_debiased,
        "debiasing_stats": debias_stats,
        "spearman": {
            "best_dim_per_var": spearman_results.get("best_dim_per_var", {}),
            "max_rho": float(np.max(np.abs(spearman_results.get("rho_matrix", [[0]])))),
        },
        "data_source": data_source,
        "n_samples": len(embeddings),
    }

    raw_control_results = {
        "spatial_cv": spatial_cv_results,
        "temporal_cv": temporal_cv_results,
        "temporal_stability": temporal_stability,
        "comparison": {
            "category_raw_vs_deb": {
                cat.lower(): {
                    "raw": ridge_raw["category_means"].get(cat, 0),
                    "debiased": ridge_debiased["category_means"].get(cat, 0),
                }
                for cat in GBA_CATEGORIES
            },
        },
    }

    nonlinear_results = {
        "random_forest": {
            "test_r2": rf_results.get("per_var", {}),
            "cv_r2": rf_results.get("cv_r2", []),
        },
        "xgboost": {
            "test_r2": xgb_results.get("test_r2", {}),
        },
        "linear_vs_nonlinear": linear_vs_nonlinear_comparison(
            ridge_raw["r2"],
            {name: rf_results["per_var"].get(name, {}).get("cv_r2", np.nan)
             for name in var_names} if rf_results.get("per_var") else {},
            xgb_results.get("test_r2"),
        ),
    }

    # Save results
    save_results(final_results, raw_control_results, nonlinear_results)

    # Paper comparison
    comparison = compare_with_paper(final_results, raw_control_results, nonlinear_results)

    # ── Phase 7: Visualization ──────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PHASE 7: Visualization")
    print(f"{'─'*60}")

    generate_all_figures(final_results, raw_control_results, nonlinear_results, spearman_results)

    # ── Complete ─────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE — Total time: {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"""
  Output files:
    Results:
      - {FINAL_DIR / 'final_results.json'}
      - {RAW_DIR / 'raw_control_results.json'}
      - {NONLINEAR_DIR / 'nonlinear_results.json'}

    FAISS Indices:
      - {FAISS_DIR / 'gba_raw.faiss'}
      - {FAISS_DIR / 'gba_debiased.faiss'}

    Figures (7):
      - {FIGURES_DIR / 'comparison_r2_by_category.png'}
      - {FIGURES_DIR / 'comparison_spatial_cv_gap.png'}
      - {FIGURES_DIR / 'comparison_variance_decomposition.png'}
      - {FIGURES_DIR / 'comparison_linear_vs_nonlinear.png'}
      - {FIGURES_DIR / 'comparison_lsi_evaluation.png'}
      - {FIGURES_DIR / 'comparison_spearman_heatmap.png'}
      - {FIGURES_DIR / 'comparison_summary_dashboard.png'}
""")

    return {
        "final": final_results,
        "raw_control": raw_control_results,
        "nonlinear": nonlinear_results,
        "comparison": comparison,
    }


def run_known_results():
    """Run with known results — fastest, uses hardcoded experimental data."""
    print("=" * 70)
    print("  AlphaEarth Reproduction — Using Known Results")
    print("  (Experimentally verified GBA data from the report)")
    print("=" * 70)

    # Load known results
    known = load_known_results()

    # Save to JSON files
    save_results(known["final"], known["raw_control"], known["nonlinear"])

    # Run comparison
    comparison = compare_with_paper(
        known["final"], known["raw_control"], known["nonlinear"]
    )

    # Generate all figures
    generate_all_figures(
        known["final"], known["raw_control"], known["nonlinear"]
    )

    print(f"\n  All results saved and figures generated.")
    print(f"  Key finding: Paper R²=0.97 is primarily CONUS-scale spatial artifact.")
    print(f"  At GBA scale: 72% spatial signal, 28% environmental signal.")

    return known


# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="AlphaEarth Reproduction Pipeline"
    )
    parser.add_argument("--real", action="store_true",
                        help="Use real data (if available)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: smaller subsamples, fewer trees")
    parser.add_argument("--known", action="store_true",
                        help="Use known results (fastest, report data)")
    parser.add_argument("--save-data", action="store_true",
                        help="Save generated synthetic data to parquet")
    args = parser.parse_args()

    if args.known:
        run_known_results()
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
