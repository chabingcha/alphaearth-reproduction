# Improvement: ExtraTrees for AlphaEarth Embedding Reconstruction

## Hypothesis

The paper's per-variable Random Forest models use bootstrap samples and
optimized split thresholds. On small reproduction datasets, this can increase
tree correlation and split-selection variance. Replacing this component with
Extremely Randomized Trees should reduce variance by randomizing both sample
partition thresholds and feature exposure.

The proposed full configuration uses:

- 100 ExtraTrees estimators;
- `max_features=0.7`;
- the same seed-specific train/test partition as the baseline;
- a strictly held-out test set for all performance claims.

## Clearly marked changes

- `data/generate_synthetic_data.py`: deterministic per-call `random_seed`
  block marked `IMPROVEMENT MODIFICATION`.
- `src/improved_extra_trees.py`: new implementation, marked
  `AI-ASSISTED IMPROVEMENT`.
- `experiments/run_improvement_experiment.py`: paired experiment, ablations,
  robustness tests, statistics, and figure generation.
- `tests/test_improvement.py`: regression tests for model construction and
  support-recovery metrics.
- `results/improvement/`: generated experiment data and visualizations.
- `report/`: final proposal and performance analysis report.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest tests/test_improvement.py
python experiments/run_improvement_experiment.py
```

For a two-seed smoke test:

```bash
python experiments/run_improvement_experiment.py --quick
```

The full experiment is CPU parallelized inside scikit-learn and writes stable
CSV, JSON, and PNG outputs under `results/improvement/`.
