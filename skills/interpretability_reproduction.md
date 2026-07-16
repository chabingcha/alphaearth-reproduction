# Skill: Embedding Interpretability Reproduction

## Metadata
- **Name**: interpretability-reproduction
- **Version**: 1.0
- **Description**: Reproduces the Random Forest + Spearman correlation interpretability analysis from the AlphaEarth paper, using synthetic data with known ground-truth mappings.
- **Input**: None (generates synthetic data internally)
- **Output**: JSON results files + publication-quality comparison figures

## Instructions

You are an embedding interpretability analysis agent. Your task is to run a reproduction experiment for the AlphaEarth paper's core finding: that foundation model embedding dimensions encode physically meaningful environmental variables.

### Step 1: Generate Synthetic Data

Create a dataset that mimics the paper's data structure:
- 64-dimensional embedding vectors
- 12 "active" dimensions that encode specific environmental variables with known correlation strength (0.5–0.95)
- 52 "noise" dimensions (uncorrelated with any variable)
- 26 environmental variables across 7 categories: Terrain, Soil, Vegetation, Temperature, Climate, Hydrology, Urban
- Spatial coordinates for spatial block CV

The key insight: with synthetic data, we KNOW the ground-truth mappings, enabling recovery accuracy evaluation that the paper cannot provide.

### Step 2: Spearman Rank Correlation Analysis

Compute the 64×26 Spearman ρ matrix:
- For each (dimension_i, variable_j) pair, compute Spearman rank correlation
- Identify the best dimension per variable (max |ρ|)
- Evaluate recovery: how many ground-truth mappings does Spearman correctly identify?
- Report top-10 dimension-variable pairs and recovery rate

### Step 3: Random Forest Regression

Train 26 separate Random Forest regressors (one per environmental variable):
- Use 64 embedding dimensions as features
- Report OOB R² for each variable
- Compute permutation importance to rank dimensions
- Identify top-3 dimensions per variable
- Evaluate: are ground-truth dimensions in the top-3?

### Step 4: Spatial Block Cross-Validation

- Partition coordinates into spatial blocks (2°–3°)
- Compare random 3-fold CV vs spatial block 3-fold CV
- Compute ΔR² = R²_random − R²_spatial for each variable
- Paper reports mean ΔR² = 0.017 — does our methodology show similar results?

### Step 5: Method Convergence

- Compute Pearson correlation between |Spearman ρ| and RF permutation importance
- Paper reports r = 0.45 — how does our synthetic data compare?
- Per-dimension convergence analysis

### Step 6: Generate Comparison Figures

Create 6 publication-quality figures:
1. Spearman ρ heatmap (dimensions × variables subset)
2. R² by category: Paper vs Ours
3. Ground truth recovery analysis
4. Method convergence: Spearman vs RF
5. Spatial CV ΔR²
6. Summary metrics comparison

### Quality Criteria
- All metrics must be saved as structured JSON
- Figures must use consistent color palette (Nature-inspired)
- Comparison table must clearly distinguish paper values from our values
- All modifications from the paper must be explicitly documented

### Usage

```python
# The skill is implemented as the quick_run.py script:
from quick_run import run_all
results = run_all(n_samples=2000, n_estimators=30)

# Or run individual phases:
from spearman_analysis import run_spearman_analysis
spearman_results = run_spearman_analysis()
```

### Configuration

Adjustable parameters:
- `n_samples`: Number of synthetic samples (default: 2000)
- `n_estimators`: RF trees (default: 30)
- `noise_level`: Data noise fraction (default: 0.3)
- `n_active_dims`: Number of embedding dims with known mappings (default: 12)
- `block_size`: Spatial block size in degrees (default: 3.0)
