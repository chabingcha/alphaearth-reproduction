"""
Quick run script — generates full results with reduced parameters.
Use this for fast iteration. For full-scale reproduction, use main.py.

Usage: python quick_run.py
"""

import sys, os, time, json, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
sys.path.insert(0, 'data')

N_SAMPLES = 2000
N_ESTIMATORS = 30

t_total = time.time()

# Phase 1: Data
print("=" * 60)
print("  PHASE 1: Data Generation")
print("=" * 60)
from generate_synthetic_data import generate_dataset, save_dataset
embeddings, env_vars, coords, ground_truth, env_var_names, active_dims = \
    generate_dataset(n_samples=N_SAMPLES, noise_level=0.3)
save_dataset(embeddings, env_vars, coords, ground_truth, env_var_names, active_dims)

# Phase 2: Spearman
print("\n" + "=" * 60)
print("  PHASE 2: Spearman")
print("=" * 60)
from spearman_analysis import run_spearman_analysis
spearman_results = run_spearman_analysis()

# Phase 3: RF (OOB, fast)
print("\n" + "=" * 60)
print("  PHASE 3: Random Forest")
print("=" * 60)

# ---- Inline RF for speed (avoid recomputing spearman matrix) ----
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr, pearsonr

# Spearman matrix (needed for convergence)
n_sub = min(1000, N_SAMPLES)
idx_sub = np.random.choice(N_SAMPLES, n_sub, replace=False)
rho_matrix = np.zeros((64, 26))
for i in range(64):
    for j in range(26):
        rho, _ = spearmanr(embeddings[idx_sub, i], env_vars[idx_sub, j])
        rho_matrix[i, j] = rho

# Train RF models
print(f"Training 26 RF models ({N_ESTIMATORS} trees, OOB)...")
t0 = time.time()
cv_r2_scores = {}
perm_importance_matrix = np.zeros((64, 26))
top_dims_per_var = {}

for j in range(26):
    y = env_vars[:, j]
    var_name = env_var_names[j]

    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=42,
                               n_jobs=1, oob_score=True)
    rf.fit(embeddings, y)

    n_perm = min(500, N_SAMPLES)
    perm_idx = np.random.choice(N_SAMPLES, n_perm, replace=False)
    perm_result = permutation_importance(
        rf, embeddings[perm_idx], y[perm_idx],
        n_repeats=3, random_state=42, n_jobs=1
    )
    perm_importance_matrix[:, j] = perm_result.importances_mean

    # Top-3 dims per var
    top3_idx = np.argsort(perm_importance_matrix[:, j])[-3:][::-1]
    top_dims_per_var[var_name] = [
        {"dim_index": int(i), "dim_label": f"A{int(i):02d}",
         "importance": float(perm_importance_matrix[i, j])}
        for i in top3_idx
    ]

    cv_r2_scores[var_name] = {"oob_r2": float(rf.oob_score_)}
    if (j + 1) % 5 == 0:
        print(f"  [{j+1}/26] {var_name}: OOB R2 = {rf.oob_score_:.4f}")

elapsed = time.time() - t0
print(f"  Completed in {elapsed:.1f}s")

# Evaluate vs ground truth
true_dims = {int(k): v for k, v in ground_truth.items()}
rf_eval_mappings = []
n_in_top3 = 0
n_top1 = 0
for dim_idx, info in true_dims.items():
    true_var = info["env_var_name"]
    var_idx = info["env_var_idx"]
    rf_imp = float(perm_importance_matrix[dim_idx, var_idx])
    top3_dims = [d["dim_index"] for d in top_dims_per_var[true_var]]
    in_top3 = dim_idx in top3_dims
    rank = top3_dims.index(dim_idx) + 1 if in_top3 else None
    if in_top3: n_in_top3 += 1
    if rank == 1: n_top1 += 1
    rf_eval_mappings.append({
        "dim": f"A{dim_idx:02d}", "true_var": true_var,
        "in_top3": in_top3, "rank_in_top3": rank,
    })

rf_eval = {
    "n_ground_truth_mappings": len(true_dims),
    "n_in_top3": n_in_top3,
    "n_top1": n_top1,
    "top3_recovery_rate": round(n_in_top3 / len(true_dims), 4) if true_dims else 0,
    "top1_recovery_rate": round(n_top1 / len(true_dims), 4) if true_dims else 0,
    "per_mapping": rf_eval_mappings,
}

# Method convergence
spearman_abs = np.abs(rho_matrix).flatten()
rf_imp_flat = perm_importance_matrix.flatten()
overall_r, overall_p = pearsonr(spearman_abs, rf_imp_flat)
per_dim_r = []
for i in range(64):
    r, _ = pearsonr(np.abs(rho_matrix[i, :]), perm_importance_matrix[i, :])
    per_dim_r.append(float(r))

convergence = {
    "overall_pearson_r": round(float(overall_r), 4),
    "mean_per_dim_r": round(float(np.mean(per_dim_r)), 4),
    "per_dim_pearson_r": [round(r, 4) for r in per_dim_r],
}

r2_values = [v["oob_r2"] for v in cv_r2_scores.values()]
rf_results = {
    "summary": {
        "n_variables": 26, "n_samples": N_SAMPLES, "n_estimators": N_ESTIMATORS,
        "mean_r2": round(float(np.mean(r2_values)), 4),
        "best_r2": round(float(np.max(r2_values)), 4),
        "best_r2_var": env_var_names[int(np.argmax(r2_values))],
        "n_vars_r2_above_050": sum(1 for r in r2_values if r > 0.50),
        "n_vars_r2_above_075": sum(1 for r in r2_values if r > 0.75),
    },
    "r2_per_variable": cv_r2_scores,
    "top3_dims_per_variable": top_dims_per_var,
    "ground_truth_evaluation": rf_eval,
    "method_convergence": convergence,
}

with open("results/rf_results.json", "w") as f:
    json.dump(rf_results, f, indent=2)

print(f"  Mean OOB R2: {np.mean(r2_values):.4f}")
print(f"  Best R2: {np.max(r2_values):.4f}")
print(f"  RF Top-3 recovery: {n_in_top3}/{len(true_dims)} ({rf_eval['top3_recovery_rate']:.1%})")
print(f"  Method convergence r: {overall_r:.4f}")

# Phase 4: Spatial Validation (quick)
print("\n" + "=" * 60)
print("  PHASE 4: Spatial Validation")
print("=" * 60)
from spatial_validation import assign_spatial_blocks
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score

block_ids = assign_spatial_blocks(coords, block_size=3.0)  # Larger blocks for smaller dataset
n_blocks = len(np.unique(block_ids))
cv_folds = min(3, n_blocks)
print(f"  {n_blocks} blocks, {cv_folds}-fold CV")

per_var_results = {}
for j in range(26):
    y = env_vars[:, j]
    var_name = env_var_names[j]

    # Random CV
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    r2_rand = []
    for tr, te in kf.split(embeddings):
        rf = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1)
        rf.fit(embeddings[tr], y[tr])
        r2_rand.append(r2_score(y[te], rf.predict(embeddings[te])))

    # Spatial CV
    try:
        kf_sp = GroupKFold(n_splits=cv_folds)
        r2_spat = []
        for tr, te in kf_sp.split(embeddings, groups=block_ids):
            rf = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1)
            rf.fit(embeddings[tr], y[tr])
            r2_spat.append(r2_score(y[te], rf.predict(embeddings[te])))
    except:
        r2_spat = r2_rand

    delta = np.mean(r2_rand) - np.mean(r2_spat)
    per_var_results[var_name] = {
        "r2_random_mean": round(float(np.mean(r2_rand)), 4),
        "r2_spatial_mean": round(float(np.mean(r2_spat)), 4),
        "delta_r2": round(float(delta), 4),
        "delta_r2_abs": round(float(abs(delta)), 4),
    }
    if (j + 1) % 5 == 0:
        print(f"  [{j+1}/26] {var_name}: dR2 = {delta:+.4f}")

delta_vals = [v["delta_r2"] for v in per_var_results.values()]
spatial_results = {
    "summary": {
        "mean_delta_r2": round(float(np.mean(delta_vals)), 4),
        "mean_abs_delta_r2": round(float(np.mean([abs(d) for d in delta_vals])), 4),
    },
    "per_variable": per_var_results,
}
with open("results/spatial_validation.json", "w") as f:
    json.dump(spatial_results, f, indent=2)

print(f"  Mean dR2: {np.mean(delta_vals):.4f} (paper: 0.017)")

# Phase 5: Figures
print("\n" + "=" * 60)
print("  PHASE 5: Figures")
print("=" * 60)
from visualization import generate_all_figures
generate_all_figures()

t_total_elapsed = time.time() - t_total
print(f"\n{'='*60}")
print(f"  COMPLETE in {t_total_elapsed:.1f}s")
print(f"{'='*60}")
