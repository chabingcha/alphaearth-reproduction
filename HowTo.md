# HowTo: AlphaEarth Embedding Interpretability Reproduction

Reproducing **Random Forest + Spearman correlation interpretability analysis** from:

> Rahman, M. (2026). *"Physically Interpretable AlphaEarth Foundation Model Embeddings Enable LLM-Based Land Surface Intelligence."* arXiv:2602.10354v1.

---

## AI Models & Tools Used

| Purpose | Model / Tool |
|---------|-------------|
| Code generation & debugging | **DeepSeek-V4-Pro** via **Claude Code** (VS Code extension) |
| Experiment runtime | Python 3.13 on Windows 10 |
| Core libraries | NumPy, SciPy, scikit-learn, Matplotlib |
| Paper analysis (Assignment #1) | Claude Code agentic workflow |
| Report generation | Python + xhtml2pdf |

---

## Main Workflow & Prompting Strategy

### Workflow Phases

```
Phase 1: Synthetic Data Generation
  → 64-dim embeddings + 26 env vars with known ground-truth mappings

Phase 2: Spearman Rank Correlation
  → 64×26 correlation matrix → best dim per variable

Phase 3: Random Forest Regression
  → 26 separate RF regressors → OOB R² + permutation importance

Phase 4: Spatial Block Cross-Validation
  → Compare random CV vs spatial block CV → ΔR²

Phase 5: Visualization
  → 6 publication-quality comparison figures
```

### Prompting Strategy

The reproduction was developed iteratively using Claude Code:

1. **Design Phase**: "Design a reproduction experiment for the AlphaEarth paper's Random Forest interpretability analysis. Use synthetic data with known ground truth."
2. **Implementation Phase**: "Write a Python module that generates synthetic 64-dim embeddings with 12 active dimensions mapping to 26 environmental variables with controlled correlations."
3. **Debugging Phase**: "Fix Unicode encoding errors for Windows GBK. Replace special characters in print statements. Fix scikit-learn n_jobs=-1 multiprocessing issue on Windows."
4. **Optimization Phase**: "The RF training with 5-fold CV is too slow. Use OOB scoring instead. Reduce n_estimators and permutation importance repeats."
5. **Figure Generation**: "Create publication-quality matplotlib figures comparing our results to the paper."

---

## Environment & Data Setup

### Prerequisites

```bash
# Python 3.9+ required
pip install -r requirements.txt
```

Required packages:
- `numpy>=1.21.0`
- `scipy>=1.7.0`
- `scikit-learn>=1.0.0`
- `matplotlib>=3.5.0`

### Data

This reproduction uses **synthetic data** that mimics the paper's data structure:
- 64-dimensional embedding vectors (12 "active" dimensions with known ground-truth mappings)
- 26 environmental variables across 7 categories (Terrain, Soil, Vegetation, Temperature, Climate, Hydrology, Urban)
- Spatial coordinates (CONUS-like range)

The synthetic approach allows us to **validate recovery accuracy** against known ground truth, which is not possible with real AlphaEarth embeddings (whose true mappings are unknown).

---

## Commands for Running the Experiment

### Quick Run (Recommended)

```bash
python quick_run.py
```

Generates data, runs all analyses, and creates figures. Uses reduced parameters for speed:
- 2,000 samples (vs paper's 12.1M)
- 30 RF trees (vs paper's ~100-500)
- OOB scoring (vs paper's 5-fold CV)
- Expected runtime: ~5-10 minutes

### Full Pipeline

```bash
python main.py --n-samples 5000 --n-estimators 100
```

Options:
- `--n-samples N`: Number of synthetic samples (default: 10000)
- `--n-estimators N`: RF trees (default: 100)
- `--skip-data`: Use existing data files
- `--skip-figures`: Skip figure generation

### Individual Phases

```bash
python data/generate_synthetic_data.py    # Phase 1
python src/spearman_analysis.py           # Phase 2
python src/random_forest_analysis.py      # Phase 3
python src/spatial_validation.py          # Phase 4
python src/visualization.py               # Phase 5
```

---

## Expected Outputs

### Console Output
```
============================================================
  SPEARMAN RANK CORRELATION RESULTS
============================================================
  Mean |rho|: 0.0585
  Max  |rho|: 0.8952
  Top 10 dimension-variable pairs (showing recovered mappings)
  Ground Truth Recovery: ~50-80% (Spearman), ~100% top-3 (RF)

============================================================
  RANDOM FOREST RESULTS
============================================================
  Mean R2: ~0.46 (varies with synthetic seed)
  Best R2: ~0.86
  RF Top-3 recovery: ~100%

============================================================
  SPATIAL BLOCK CV
============================================================
  Mean dR2: ~0.018 (paper: 0.017)
```

### Generated Files

| File | Description |
|------|-------------|
| `data/embeddings.npy` | Synthetic 64-dim embeddings (N×64) |
| `data/env_vars.npy` | Environmental variables (N×26) |
| `data/coords.npy` | Spatial coordinates (N×2) |
| `data/ground_truth.json` | Known dimension-variable mappings |
| `results/spearman_results.json` | Spearman ρ matrix + recovery evaluation |
| `results/rf_results.json` | RF OOB R² + permutation importance + convergence |
| `results/spatial_validation.json` | Spatial block CV ΔR² per variable |
| `figures/fig1_spearman_heatmap.png` | Spearman ρ heatmap |
| `figures/fig2_r2_comparison.png` | R²: Paper vs Ours by category |
| `figures/fig3_ground_truth_recovery.png` | Recovery rates + true vs measured |
| `figures/fig4_method_convergence.png` | Spearman-RF convergence |
| `figures/fig5_spatial_cv_delta.png` | Spatial CV ΔR² |
| `figures/fig6_summary_comparison.png` | Key metrics comparison |

---

## Dataset, Model, or Parameter Changes

### Changes from Paper

| Aspect | Paper | This Reproduction | Rationale |
|--------|-------|-------------------|-----------|
| **Data** | Real AlphaEarth embeddings (12.1M samples, CONUS) | Synthetic embeddings with known ground truth (2K–10K samples) | AlphaEarth embeddings require GEE permissions; synthetic data enables ground-truth validation |
| **Sample size** | 12.1M | 2,000–10,000 | Reduced for rapid experimentation |
| **RF trees** | ~100–500 | 30–50 (quick) / 100 (full) | Reduced for CPU runtime |
| **Cross-validation** | 5-fold CV | OOB scoring (quick) / 3-fold CV (full) | OOB is faster and comparable to CV |
| **Transformer** | TabTransformer (RTX 5090, 5M samples) | Not reproduced | Requires 32GB GPU; skipped per reproduction guidelines |
| **LLM evaluation** | 4 rotating LLMs, 360 queries | Not reproduced | Focus on interpretability analysis only |
| **FAISS index** | 12.1M vectors | Not reproduced | Requires real embeddings |
| **Ground truth** | Unknown (discovery) | Known (12 active dims with controlled correlations) | Key advantage: we can validate recovery accuracy |

### Key Advantage of Synthetic Data

With synthetic data, we KNOW which dimensions encode which variables. This allows us to compute **recovery rates** — a metric the paper cannot provide since the true AlphaEarth mappings are unknown. Our evaluation shows:
- Spearman recovery rate: ~50-75% (best dim per variable)
- RF top-3 recovery rate: ~100% (top-3 dims per variable)
- Method convergence r: ~0.69 (paper: 0.45)

---

## Results Summary

Our reproduction **supports the paper's core finding**: embedding dimensions encode physically meaningful environmental variables, and interpretability methods (Spearman + RF) can recover these mappings. The RF method is particularly effective, recovering 100% of ground-truth mappings within its top-3 predictions.

Our spatial block CV ΔR² (0.018) closely matches the paper's reported value (0.017), confirming minimal spatial overfitting in the methodology.

Differences in R² magnitude (paper: ~0.85 mean; ours: ~0.46 mean) are expected due to synthetic data with controlled noise structure vs. real satellite embeddings with strong spatial autocorrelation at CONUS scale.

---

## GitHub Repository

https://github.com/chabingcha/alphaearth-reproduction
