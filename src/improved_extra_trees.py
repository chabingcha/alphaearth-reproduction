"""
Improved tree-ensemble module for AlphaEarth embedding reconstruction.

=== AI-ASSISTED IMPROVEMENT ===
This module implements the proposed replacement for the baseline Random
Forest component: Extremely Randomized Trees (ExtraTrees) with controlled
feature subsampling. It also centralizes leakage-free held-out evaluation and
ground-truth support-recovery metrics.
=== END AI-ASSISTED IMPROVEMENT ===
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import average_precision_score, r2_score


@dataclass(frozen=True)
class ForestConfig:
    """Serializable model configuration used by the ablation study."""

    name: str
    algorithm: str
    n_estimators: int
    max_features: float
    min_samples_leaf: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


MODEL_CONFIGS = (
    ForestConfig("rf_baseline", "random_forest", 50, 1.0),
    ForestConfig("et_random_thresholds", "extra_trees", 50, 1.0),
    ForestConfig("et_more_trees", "extra_trees", 100, 1.0),
    ForestConfig("et_feature_subsample", "extra_trees", 50, 0.7),
    ForestConfig("et_full", "extra_trees", 100, 0.7),
)


def build_regressor(config: ForestConfig, random_seed: int, n_jobs: int = -1):
    """Construct a deterministic regressor from an explicit configuration."""

    common = dict(
        n_estimators=config.n_estimators,
        max_features=config.max_features,
        min_samples_leaf=config.min_samples_leaf,
        random_state=random_seed,
        n_jobs=n_jobs,
    )
    if config.algorithm == "random_forest":
        return RandomForestRegressor(bootstrap=True, **common)
    if config.algorithm == "extra_trees":
        return ExtraTreesRegressor(bootstrap=False, **common)
    raise ValueError(f"Unsupported algorithm: {config.algorithm}")


def ground_truth_support(ground_truth: dict, n_targets: int) -> dict[int, set[int]]:
    """Convert dimension-centric mappings to target-centric support sets."""

    support = {target: set() for target in range(n_targets)}
    for raw_dim, info in ground_truth.items():
        target = int(info["env_var_idx"])
        if target < 0 or target >= n_targets:
            raise ValueError(f"Ground-truth target index out of range: {target}")
        support[target].add(int(raw_dim))
    return support


def support_recovery_metrics(
    importance_matrix: np.ndarray,
    support: dict[int, set[int]],
) -> dict[str, float]:
    """Evaluate feature ranking without the baseline's impossible Top-1 metric.

    Only targets with at least one planted active dimension are evaluated.
    The original mapping-level Top-3 recall is retained for comparability.
    """

    if importance_matrix.ndim != 2:
        raise ValueError("importance_matrix must have shape (n_dims, n_targets)")

    average_precisions = []
    recall_at_true_k = []
    top3_mapping_hits = []

    for target, true_dims in support.items():
        if not true_dims:
            continue
        scores = np.asarray(importance_matrix[:, target], dtype=float)
        labels = np.zeros(len(scores), dtype=int)
        labels[list(true_dims)] = 1
        average_precisions.append(average_precision_score(labels, scores))

        k = len(true_dims)
        top_k = set(np.argsort(scores)[::-1][:k])
        recall_at_true_k.append(len(top_k & true_dims) / k)

        top3 = set(np.argsort(scores)[::-1][:3])
        top3_mapping_hits.extend(int(dim in top3) for dim in true_dims)

    if not average_precisions:
        return {
            "support_map": float("nan"),
            "support_recall_at_true_k": float("nan"),
            "mapping_recall_at_3": float("nan"),
        }
    return {
        "support_map": float(np.mean(average_precisions)),
        "support_recall_at_true_k": float(np.mean(recall_at_true_k)),
        "mapping_recall_at_3": float(np.mean(top3_mapping_hits)),
    }


def evaluate_configuration(
    config: ForestConfig,
    embeddings: np.ndarray,
    targets: np.ndarray,
    train_indices: Iterable[int],
    test_indices: Iterable[int],
    variable_names: list[str],
    ground_truth: dict,
    random_seed: int,
    n_jobs: int = -1,
) -> tuple[dict, list[dict]]:
    """Fit one model per target and evaluate on a strictly held-out split."""

    x = np.asarray(embeddings, dtype=float)
    y = np.asarray(targets, dtype=float)
    train = np.asarray(list(train_indices), dtype=int)
    test = np.asarray(list(test_indices), dtype=int)

    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("Expected X=(n,d), Y=(n,m) with matching sample counts")
    if len(variable_names) != y.shape[1]:
        raise ValueError("variable_names length does not match target count")
    if len(train) == 0 or len(test) == 0 or np.intersect1d(train, test).size:
        raise ValueError("Train/test indices must be non-empty and disjoint")

    started = perf_counter()
    per_variable = []
    importance = np.zeros((x.shape[1], y.shape[1]), dtype=float)

    for target_index, variable_name in enumerate(variable_names):
        model = build_regressor(
            config,
            random_seed=random_seed * 1000 + target_index,
            n_jobs=n_jobs,
        )
        model.fit(x[train], y[train, target_index])
        pred = model.predict(x[test])
        importance[:, target_index] = model.feature_importances_
        per_variable.append(
            {
                "model": config.name,
                "target_index": target_index,
                "variable": variable_name,
                "test_r2": float(r2_score(y[test, target_index], pred)),
            }
        )

    elapsed = perf_counter() - started
    r2_values = np.array([row["test_r2"] for row in per_variable], dtype=float)
    support = ground_truth_support(ground_truth, y.shape[1])
    recovery = support_recovery_metrics(importance, support)

    aggregate = {
        "model": config.name,
        "macro_test_r2": float(np.mean(r2_values)),
        "median_test_r2": float(np.median(r2_values)),
        "worst_test_r2": float(np.min(r2_values)),
        "targets_r2_above_0_5": int(np.sum(r2_values > 0.5)),
        "fit_eval_seconds": float(elapsed),
        **recovery,
    }
    return aggregate, per_variable
