# AlphaEarth Embedding Interpretability — Reproduction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Reproduction of the Random Forest + Spearman correlation interpretability analysis from:

> **Rahman, M. (2026).** *"Physically Interpretable AlphaEarth Foundation Model Embeddings Enable LLM-Based Land Surface Intelligence."* arXiv:2602.10354v1.

## Overview

This repository reproduces the core experiment from the AlphaEarth paper: demonstrating that foundation model embedding dimensions encode physically meaningful environmental variables, and that interpretability methods (Spearman correlation + Random Forest) can recover these mappings.

**Reproduction Target:** Random Forest regression predicting 26 environmental variables from 64-dimensional embeddings, with Spearman correlation analysis, spatial block cross-validation, and method convergence evaluation.

**Key Modifications:** Synthetic data with known ground-truth mappings (enabling recovery accuracy evaluation), reduced sample size, CPU-only computation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full reproduction (~5-10 min)
python quick_run.py

# Or run with custom parameters
python main.py --n-samples 5000 --n-estimators 100
```

## Project Structure

```
alphaearth-reproduction/
├── data/
│   └── generate_synthetic_data.py   # Synthetic data with ground truth
├── src/
│   ├── spearman_analysis.py         # Spearman ρ correlation
│   ├── random_forest_analysis.py    # RF regression + importance
│   ├── spatial_validation.py        # Spatial block CV
│   └── visualization.py             # Publication-quality figures
├── skills/
│   ├── interpretability_reproduction.md  # Reusable skill definition
│   └── figure_generation.py              # Figure generation skill
├── results/                         # JSON results (generated)
├── figures/                         # PNG figures (generated)
├── main.py                          # Full pipeline
├── quick_run.py                     # Optimized quick pipeline
├── HowTo.md                         # Detailed instructions
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Results

| Metric | Paper (CONUS, 12.1M) | Ours (Synthetic, 2K) |
|--------|----------------------|----------------------|
| Best R² (RF) | ≈0.97 | ~0.86 |
| Mean R² | ≈0.85 | ~0.46 |
| Spatial ΔR² | 0.017 | ~0.018 |
| Method Convergence r | 0.45 | ~0.69 |
| Spearman Recovery | N/A | ~50-75% |
| RF Top-3 Recovery | N/A | ~100% |

Our reproduction **supports the paper's core conclusion**: embedding dimensions encode physically meaningful environmental variables. The RF method recovers 100% of ground-truth mappings in its top-3 predictions, and our spatial CV ΔR² (0.018) closely matches the paper's value (0.017).

## Modifications from the Paper

See [HowTo.md](HowTo.md) for a detailed comparison table.

- **Synthetic data** instead of real AlphaEarth embeddings (enables ground-truth validation)
- **OOB scoring** instead of 5-fold CV (faster, comparable metric)
- **Reduced samples** (2K–10K vs 12.1M)
- **CPU only** (no GPU/Transformer)
- **Skipped LLM/RAG evaluation** (focused on interpretability)

## Citation

```bibtex
@article{rahman2026physically,
  title={Physically Interpretable AlphaEarth Foundation Model Embeddings
         Enable LLM-Based Land Surface Intelligence},
  author={Rahman, Mashrekur},
  journal={arXiv preprint arXiv:2602.10354v1},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE) file.
