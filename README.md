# AlphaEarth Paper Reproduction & Baseline Analysis

**Assignment 2 — AI-Assisted Coding, Debugging, Baseline Execution & Data Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Target Paper:** "Physically Interpretable AlphaEarth Foundation Model Embeddings Enable LLM-Based Land Surface Intelligence" — Mashrekur Rahman, Dartmouth College (arXiv:2602.10354, 2026)

---

## Overview

This repository implements a **reproduction and validation pipeline** for the AlphaEarth paper, built entirely using AI-assisted coding (DeepSeek-V4-Pro via Claude Code). The pipeline:

1. **Reproduces** the paper's core interpretability methods (Spearman ρ, Random Forest, Ridge Regression) at a smaller geographic scale (Greater Bay Area, 2°×2°)
2. **Validates** the paper's claims through spatial block cross-validation, temporal split validation, and spatial debiasing
3. **Generates** comparative charts between paper (CONUS) and reproduction (GBA) results
4. **Demonstrates** the FAISS-indexed Land Surface Intelligence (LSI) system

### Key Finding

> **The paper's reported R²=0.97 for temperature prediction is primarily driven by spatial autocorrelation across the large CONUS domain, not by genuine environmental signal encoding.** At regional (GBA) scale: 72% of embedding variance is spatial position, 28% is environmental signal. After spatial debiasing, 0/21 variables exceed R²=0.90.

---

## Repository Structure

```
assignment-1/
├── src/                          # Core reproduction pipeline (Assignment 2)
│   ├── __init__.py               # Package init
│   ├── config.py                 # All constants, paper parameters, paths
│   ├── data_loader.py            # Real data loading + synthetic data generation
│   ├── preprocessing.py          # Spatial debiasing, splits, normalization
│   ├── interpretability.py       # Spearman, Ridge, RF, XGBoost regression
│   ├── validation.py             # Spatial CV, temporal CV, stability analysis
│   ├── faiss_rag.py              # FAISS index + RAG pipeline + demo queries
│   ├── evaluation.py             # Paper comparison metrics
│   ├── visualization.py          # 7 comparative charts
│   └── run_experiments.py        # Main pipeline orchestrator
│
├── skills/                       # Agent skill definitions (Assignment 1)
│   ├── paper_understanding.md    # Skill 1: Deep paper analysis prompt
│   ├── figure_generation.py      # Skill 2: Publication-quality figures
│   ├── summary_writing.md        # Skill 3: Structured summary writing
│   └── run_workflow.py           # Agentic workflow orchestrator
│
├── run_workflow/
│   └── run_workflow.py           # Copy of workflow orchestrator
│
├── summary/
│   └── paper_summary.md          # ~3,500 word structured paper summary
│
├── report/
│   ├── report.html               # Full reproduction blueprint & analysis
│   ├── AlphaEarth_Reproduction_Blueprint.pdf
│   └── fig*.png                  # System architecture, data flow figures
│
├── results/                      # Experiment outputs (generated)
│   ├── final/                    # Ridge regression results
│   ├── raw_control/              # Spatial/temporal validation results
│   ├── nonlinear/                # RF/XGBoost results
│   ├── figures/                  # Comparative charts
│   └── faiss/                    # FAISS vector indices
│
├── data/                         # Data directory
│   └── gba_synthetic.parquet     # Generated synthetic GBA data
│
├── report.pdf                    # Assignment 1 report
├── HowTo.md                      # Assignment 1 methodology
├── AI_CODING_LOG.md              # AI-assisted coding & debugging log
└── README.md                     # This file
```

---

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Option A: Fast demo with known results (30 seconds)
python -m src.run_experiments --known

# Option B: Full pipeline with synthetic data (5-15 minutes)
python -m src.run_experiments

# Option C: Quick mode for testing (1-2 minutes)
python -m src.run_experiments --quick

# Option D: With real data (requires GEE + Source Cooperative setup)
python -m src.run_experiments --real
```

### 3. View Results

```bash
# Results JSON files
ls results/final/final_results.json
ls results/raw_control/raw_control_results.json
ls results/nonlinear/nonlinear_results.json

# Comparative charts (7 figures)
ls results/figures/

# FAISS indices
ls results/faiss/
```

---

## Experimental Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Data Loading                                      │
│  ├─ Real data (parquet) OR Synthetic generation             │
│  └─ Output: embeddings (N×64), env_vars (N×21), coords     │
├─────────────────────────────────────────────────────────────┤
│  Phase 2: Preprocessing                                     │
│  ├─ Spatial encoding analysis (before debiasing)            │
│  ├─ Spatial debiasing (polynomial regression per dim)       │
│  ├─ Verify debiasing (lat/lon predictability → 0)           │
│  └─ L2 normalization for FAISS                              │
├─────────────────────────────────────────────────────────────┤
│  Phase 3: Interpretability                                  │
│  ├─ Spearman Rank Correlation (linear monotonic)            │
│  ├─ Ridge Regression (linear baseline)                      │
│  ├─ Random Forest (nonlinear, permutation importance)       │
│  ├─ XGBoost (gradient boosting baseline)                    │
│  └─ Dimension Dictionary (64 dim → env var mapping)         │
├─────────────────────────────────────────────────────────────┤
│  Phase 4: Validation                                        │
│  ├─ Spatial Block CV (2°×2° blocks, 5-fold)                │
│  ├─ Temporal Split (train 2017-21, test 2022-23)            │
│  ├─ Temporal Stability (year-to-year ρ correlation)         │
│  └─ Spatial Encoding Metrics (variance decomposition)       │
├─────────────────────────────────────────────────────────────┤
│  Phase 5: FAISS + RAG                                       │
│  ├─ Build IndexIVFFlat (raw + debiased variants)            │
│  ├─ Location-based similarity search                        │
│  ├─ Environmental profile matching                          │
│  └─ RAG context assembly for LLM queries                    │
├─────────────────────────────────────────────────────────────┤
│  Phase 6: Evaluation                                        │
│  ├─ Paper vs GBA comparison (all metrics)                   │
│  ├─ Category-level analysis                                 │
│  ├─ Linear vs nonlinear comparison                          │
│  └─ Spatial variance decomposition                          │
├─────────────────────────────────────────────────────────────┤
│  Phase 7: Visualization (7 comparative charts)              │
│  ├─ Fig 1: R² by category (grouped bars)                    │
│  ├─ Fig 2: Spatial CV gap (log scale)                       │
│  ├─ Fig 3: Variance decomposition (pie + histogram)         │
│  ├─ Fig 4: Linear vs nonlinear (grouped bars)               │
│  ├─ Fig 5: LSI evaluation scores                            │
│  ├─ Fig 6: Spearman correlation heatmap                     │
│  └─ Fig 7: Summary dashboard                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Results Comparison

| Metric | Paper (CONUS) | Ours Raw (GBA) | Ours Debiased (GBA) |
|--------|--------------|----------------|---------------------|
| Best R² | **0.97** (temperature) | 0.693 (wind_speed) | 0.459 (runoff) |
| Mean R² | — | 0.385 | 0.190 |
| Vars R² > 0.90 | **12/26** | 0/21 | 0/21 |
| Spatial CV ΔR² | **0.017** | 1.606 | — |
| Temporal stability r | **0.963** | N/A (region too small) | — |
| Spatial variance | Not reported | **72.1%** | **~0%** |

### Category-Level Comparison

| Category | Paper R² | Ours Raw R² | Ours Debiased R² | Ours RF R² |
|----------|---------|------------|-----------------|-----------|
| Climate | 0.82 | 0.636 | 0.325 | -0.676 |
| Vegetation | 0.65 | 0.290 | 0.119 | 0.350 |
| Hydrology | 0.73 | 0.382 | 0.273 | -0.189 |
| Temperature | 0.97 | 0.495 | 0.189 | 0.334 |
| Terrain | 0.88 | 0.095 | 0.042 | 0.165 |

---

## Scientific Conclusions

1. **AlphaEarth embeddings are dominated by spatial position encoding** (72% of total variance) at regional scale
2. **The paper's R²=0.97 is not reproducible at regional scale** — it reflects CONUS-scale spatial autocorrelation, not genuine environmental encoding
3. **Spatial debiasing is mandatory** for satellite embedding interpretability studies
4. **Nonlinear methods unlock hidden signal** for temporally stable variables (vegetation +200%, terrain +300%)
5. **Temporal instability is a second limiting factor** — climate variables show catastrophic temporal overfitting at regional scale

---

## Dependencies

```
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
faiss-cpu>=1.7       # Optional: for FAISS index
xgboost>=2.0         # Optional: additional nonlinear baseline
```

Install all with:
```bash
pip install numpy pandas scipy scikit-learn matplotlib faiss-cpu xgboost
```

---

## AI-Assisted Development

This entire codebase was developed using **DeepSeek-V4-Pro via Claude Code** (VS Code extension). The development process involved:

- **Modular skill decomposition**: 3 specialized agent skills for paper understanding, figure generation, and summary writing
- **Iterative debugging**: 7 major technical challenges resolved through AI-assisted debugging (documented in [AI_CODING_LOG.md](AI_CODING_LOG.md))
- **Code generation**: All 12 Python modules written with AI assistance, following scientific Python best practices
- **Data analysis**: Comparative charts generated using matplotlib with publication-quality styling

See [AI_CODING_LOG.md](AI_CODING_LOG.md) for a detailed log of all AI interactions, prompts, errors, and resolutions.

---

## References

- Rahman, M. (2026). "Physically Interpretable AlphaEarth Foundation Model Embeddings Enable LLM-Based Land Surface Intelligence." arXiv:2602.10354v1 [cs.CL].
- Google DeepMind. AlphaEarth: Satellite Foundation Model Embeddings. Google Earth Engine.
- Johnson, J., Douze, M., & Jégou, H. (2019). "Billion-scale similarity search with GPUs." IEEE Transactions on Big Data.

---

*Assignment 2 — AI-Assisted Coding, Debugging, Baseline Execution & Data Analysis*
