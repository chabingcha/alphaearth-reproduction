"""
Spearman Rank Correlation Analysis

Reproduces the paper's Spearman ρ analysis:
- Computes 64×N_vars Spearman correlation matrix between embedding dimensions
  and environmental variables
- Identifies the best dimension per variable (max |ρ|)
- Validates against ground truth (in our synthetic data, we know the true mappings)

Paper reference: Section 3.2, "Spearman Rank Correlation"
  - n = 1,000,000 random subsample
  - 64×26 correlation matrix
  - All nonzero ρ: p < 0.001
  - Best dim per var: argmax |ρ_{ij}|

Modifications for reproduction:
  - n = 10,000 (reduced for rapid experimentation)
  - Synthetic data with known ground truth
  - Same methodology and metrics
"""

import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import json
import time


def compute_spearman_matrix(embeddings, env_vars, env_var_names, subsample=None):
    """
    Compute the full Spearman ρ matrix between all embedding dimensions
    and all environmental variables.

    Parameters
    ----------
    embeddings : ndarray (n_samples, n_dims)
    env_vars : ndarray (n_samples, n_vars)
    env_var_names : list of str
    subsample : int or None
        If provided, use a random subsample for computation

    Returns
    -------
    rho_matrix : ndarray (n_dims, n_vars)
    p_matrix : ndarray (n_dims, n_vars)
    best_dim_per_var : dict — var_name → (dim_idx, rho, p_value)
    top_pairs : list of (dim_idx, var_name, rho) sorted by |ρ|
    """
    n_dims = embeddings.shape[1]
    n_vars = env_vars.shape[1]

    if subsample and subsample < embeddings.shape[0]:
        idx = np.random.choice(embeddings.shape[0], subsample, replace=False)
        X = embeddings[idx]
        Y = env_vars[idx]
    else:
        X = embeddings
        Y = env_vars

    print(f"Computing Spearman rho: {n_dims} dims x {n_vars} vars on {X.shape[0]} samples...")
    t0 = time.time()

    rho_matrix = np.zeros((n_dims, n_vars))
    p_matrix = np.zeros((n_dims, n_vars))

    for i in range(n_dims):
        for j in range(n_vars):
            rho, p = spearmanr(X[:, i], Y[:, j])
            rho_matrix[i, j] = rho
            p_matrix[i, j] = p

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s ({n_dims * n_vars} pairs computed)")

    # ── Best dimension per variable ──
    best_dim_per_var = {}
    for j in range(n_vars):
        abs_rho = np.abs(rho_matrix[:, j])
        best_dim = int(np.argmax(abs_rho))
        best_dim_per_var[env_var_names[j]] = {
            "dim_index": best_dim,
            "spearman_rho": float(rho_matrix[best_dim, j]),
            "p_value": float(p_matrix[best_dim, j]),
            "abs_rho": float(abs_rho[best_dim]),
        }

    # ── Top pairs by |ρ| ──
    pairs = []
    for i in range(n_dims):
        for j in range(n_vars):
            pairs.append((i, env_var_names[j], rho_matrix[i, j]))
    top_pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    # ── Summary statistics ──
    nonzero_mask = np.abs(rho_matrix) > 1e-10
    sig_mask = p_matrix < 0.001
    summary = {
        "n_dims": n_dims,
        "n_vars": n_vars,
        "n_samples": X.shape[0],
        "mean_abs_rho": float(np.mean(np.abs(rho_matrix))),
        "max_abs_rho": float(np.max(np.abs(rho_matrix))),
        "n_significant_pairs": int(np.sum(nonzero_mask & sig_mask)),
        "n_pairs_p_lt_001": int(np.sum(p_matrix < 0.001)),
        "top_10_pairs": [
            {"dim": f"A{p[0]:02d}", "var": p[1], "rho": round(p[2], 4)}
            for p in top_pairs[:10]
        ],
    }

    return rho_matrix, p_matrix, best_dim_per_var, top_pairs, summary


def evaluate_against_ground_truth(best_dim_per_var, ground_truth, rho_matrix, env_var_names):
    """
    Evaluate how well Spearman correlation recovers the known ground-truth mappings.

    Parameters
    ----------
    best_dim_per_var : dict
    ground_truth : dict — {dim_idx: {env_var_idx, env_var_name, true_correlation}}
    rho_matrix : ndarray
    env_var_names : list

    Returns
    -------
    evaluation : dict
    """
    # ── Find which active dims were correctly identified ──
    true_dims = {int(k): v for k, v in ground_truth.items()}

    # For each ground-truth mapping, check if Spearman found it
    recovered = []
    for dim_idx, info in true_dims.items():
        true_var = info["env_var_name"]
        true_corr = info["true_correlation"]
        var_idx = info["env_var_idx"]

        # What did Spearman find for this variable?
        measured_rho = rho_matrix[dim_idx, var_idx]
        abs_measured = abs(measured_rho)

        # Is this dimension the best for this variable?
        best_for_var = best_dim_per_var[true_var]["dim_index"]
        is_best = (best_for_var == dim_idx)

        recovered.append({
            "dim": f"A{dim_idx:02d}",
            "true_var": true_var,
            "true_correlation": round(true_corr, 4),
            "measured_rho": round(float(measured_rho), 4),
            "abs_measured_rho": round(abs_measured, 4),
            "is_top_dim_for_var": is_best,
            "error": round(abs(true_corr - abs_measured), 4),
        })

    # ── Compute recovery metrics ──
    n_total = len(true_dims)
    n_recovered = sum(1 for r in recovered if r["is_top_dim_for_var"])
    mean_abs_rho = np.mean([r["abs_measured_rho"] for r in recovered])
    mean_error = np.mean([r["error"] for r in recovered])

    evaluation = {
        "n_ground_truth_mappings": n_total,
        "n_recovered_as_top_dim": n_recovered,
        "recovery_rate": round(n_recovered / n_total, 4) if n_total > 0 else 0,
        "mean_abs_measured_rho": round(mean_abs_rho, 4),
        "mean_error_vs_ground_truth": round(mean_error, 4),
        "per_mapping": recovered,
    }

    return evaluation


def run_spearman_analysis(data_dir="data", output_dir="results"):
    """Full Spearman analysis pipeline."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    embeddings = np.load(data_dir / "embeddings.npy")
    env_vars = np.load(data_dir / "env_vars.npy")
    with open(data_dir / "ground_truth.json") as f:
        gt = json.load(f)

    # ── Compute Spearman matrix ──
    rho_matrix, p_matrix, best_dim_per_var, top_pairs, summary = \
        compute_spearman_matrix(embeddings, env_vars, gt["env_var_names"],
                                subsample=min(5000, embeddings.shape[0]))

    # ── Evaluate against ground truth ──
    evaluation = evaluate_against_ground_truth(
        best_dim_per_var, gt["mappings"], rho_matrix, gt["env_var_names"]
    )

    # ── Save results ──
    results = {
        "summary": summary,
        "best_dim_per_var": best_dim_per_var,
        "top_20_pairs": [
            {"dim": f"A{p[0]:02d}", "var": p[1], "rho": round(p[2], 4)}
            for p in top_pairs[:20]
        ],
        "ground_truth_evaluation": evaluation,
    }

    with open(output_dir / "spearman_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Print report ──
    print(f"\n{'='*60}")
    print("  SPEARMAN RANK CORRELATION RESULTS")
    print(f"{'='*60}")
    print(f"  Mean |rho|: {summary['mean_abs_rho']:.4f}")
    print(f"  Max  |rho|: {summary['max_abs_rho']:.4f}")
    print(f"  Pairs with p < 0.001: {summary['n_pairs_p_lt_001']}")
    print(f"\n  Top 10 dimension-variable pairs:")
    for p in summary["top_10_pairs"]:
        print(f"    {p['dim']} -> {p['var']:<20s}  rho = {p['rho']:+.4f}")
    print(f"\n  Ground Truth Recovery:")
    print(f"    Recovered: {evaluation['n_recovered_as_top_dim']}/{evaluation['n_ground_truth_mappings']}")
    print(f"    Recovery rate: {evaluation['recovery_rate']:.2%}")
    print(f"    Mean |measured rho|: {evaluation['mean_abs_measured_rho']:.4f}")
    print(f"    Mean error: {evaluation['mean_error_vs_ground_truth']:.4f}")
    print(f"\n  Results saved to: {output_dir / 'spearman_results.json'}")

    return results


if __name__ == "__main__":
    run_spearman_analysis()
