"""
AlphaEarth Embedding Interpretability — Reproduction Experiment
==============================================================

Reproduces the Random Forest + Spearman correlation interpretability
analysis from:

    Rahman, M. (2026). "Physically Interpretable AlphaEarth Foundation
    Model Embeddings Enable LLM-Based Land Surface Intelligence."
    arXiv:2602.10354v1.

Reproduction Target:
    - RF regression predicting environmental variables from embeddings
    - Spearman rank correlation analysis
    - Spatial block cross-validation
    - Method convergence analysis

Key modifications from the paper:
    - Synthetic data (10K samples vs 12.1M)
    - Known ground-truth mappings for validation
    - CPU-based RF (no GPU/Transformer)
    - Reduced spatial extent

Usage:
    python main.py              # Full pipeline (generate data + analyze)
    python main.py --skip-data  # Skip data generation, run analysis only
    python main.py --n-samples 20000  # Increase sample size
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "data"))

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="AlphaEarth Interpretability Reproduction Experiment"
    )
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data generation (use existing data)")
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Number of synthetic samples (default: 10000)")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="RF trees (default: 100)")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    t_total_start = time.time()

    print("=" * 65)
    print("  AlphaEarth Embedding Interpretability — Reproduction")
    print("  Paper: Rahman (2026), arXiv:2602.10354v1")
    print("=" * 65)
    print(f"\n  Configuration:")
    print(f"    Samples: {args.n_samples}")
    print(f"    RF trees: {args.n_estimators}")
    print(f"    Skip data gen: {args.skip_data}")
    print(f"    Skip figures: {args.skip_figures}")

    # ────────────────────────────────────────────────────────────
    # Phase 1: Data Generation
    # ────────────────────────────────────────────────────────────
    if not args.skip_data:
        print(f"\n{'─'*65}")
        print("  PHASE 1: Synthetic Data Generation")
        print(f"{'─'*65}")

        from generate_synthetic_data import generate_dataset, save_dataset

        embeddings, env_vars, coords, ground_truth, env_var_names, active_dims = \
            generate_dataset(n_samples=args.n_samples, noise_level=0.3)
        save_dataset(embeddings, env_vars, coords, ground_truth, env_var_names, active_dims)
    else:
        print("\n  [SKIP] Using existing data files")

    # ────────────────────────────────────────────────────────────
    # Phase 2: Spearman Correlation Analysis
    # ────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  PHASE 2: Spearman Rank Correlation")
    print(f"{'─'*65}")

    from spearman_analysis import run_spearman_analysis
    spearman_results = run_spearman_analysis()

    # ────────────────────────────────────────────────────────────
    # Phase 3: Random Forest Regression
    # ────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  PHASE 3: Random Forest Regression")
    print(f"{'─'*65}")

    from random_forest_analysis import run_rf_analysis
    rf_results = run_rf_analysis(n_estimators=args.n_estimators)

    # ────────────────────────────────────────────────────────────
    # Phase 4: Spatial Block Validation
    # ────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  PHASE 4: Spatial Block Cross-Validation")
    print(f"{'─'*65}")

    from spatial_validation import run_spatial_validation
    spatial_results = run_spatial_validation(n_estimators=args.n_estimators)

    # ────────────────────────────────────────────────────────────
    # Phase 5: Visualization
    # ────────────────────────────────────────────────────────────
    if not args.skip_figures:
        print(f"\n{'─'*65}")
        print("  PHASE 5: Figure Generation")
        print(f"{'─'*65}")

        from visualization import generate_all_figures
        generate_all_figures()

    # ────────────────────────────────────────────────────────────
    # Final Report
    # ────────────────────────────────────────────────────────────
    t_total = time.time() - t_total_start

    print(f"\n{'='*65}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*65}")
    print(f"  Total runtime: {t_total:.1f}s ({t_total/60:.1f} min)")

    # Print comparison table
    rf_sum = rf_results.get("summary", {})
    sp_sum = spearman_results.get("summary", {})
    sp_eval = spearman_results.get("ground_truth_evaluation", {})
    rf_eval = rf_results.get("ground_truth_evaluation", {})
    conv = rf_results.get("method_convergence", {})
    sp_cv = spatial_results.get("summary", {})

    print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║              RESULTS COMPARISON: Paper vs Ours               ║
  ╠══════════════════════════════╦══════════════╦════════════════╣
  ║ Metric                       ║ Paper (CONUS)║ Ours (Synth.) ║
  ╠══════════════════════════════╬══════════════╬════════════════╣
  ║ Samples                      ║ 12,100,000   ║ {args.n_samples:>12,} ║
  ║ Embedding Dims               ║      64      ║      64       ║
  ║ Env Variables                ║      26      ║      26       ║
  ║ Best R² (RF)                 ║    ≈0.97     ║ {rf_sum.get('best_r2', 0):>10.4f}   ║
  ║ Mean CV R²                   ║    ≈0.85     ║ {rf_sum.get('mean_cv_r2', 0):>10.4f}   ║
  ║ Variables R² > 0.90          ║    12/26     ║ {rf_sum.get('n_vars_r2_above_075', 0):>6}/26*  ║
  ║ Max |Spearman ρ|             ║    >0.80     ║ {sp_sum.get('max_abs_rho', 0):>10.4f}   ║
  ║ Mean Spatial ΔR²             ║    0.017     ║ {abs(sp_cv.get('mean_delta_r2', 0)):>10.4f}   ║
  ║ Method Convergence r         ║    0.45      ║ {conv.get('overall_pearson_r', 0):>10.4f}   ║
  ║ Ground Truth Recovery (Sp.)  ║     N/A      ║ {sp_eval.get('recovery_rate', 0):>10.1%}   ║
  ║ Ground Truth Recovery (RF)   ║     N/A      ║ {rf_eval.get('top3_recovery_rate', 0):>10.1%}   ║
  ╚══════════════════════════════╩══════════════╩════════════════╝
  * Using R² > 0.75 threshold (lower due to synthetic data scale)
""")

    print(f"  Deliverables:")
    print(f"    data/         — Synthetic dataset (embeddings, env vars, coords)")
    print(f"    results/      — Analysis results (JSON)")
    print(f"    figures/      — Publication-quality figures (PNG)")
    print(f"    HowTo.md      — Reproduction instructions")
    print(f"    skills/       — Reusable skill definitions")
    print(f"    report.pdf    — Final report (generated separately)")
    print(f"\n  GitHub: https://github.com/chabingcha/alphaearth-reproduction")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
