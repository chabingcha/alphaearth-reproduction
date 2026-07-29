import numpy as np

from src.improved_extra_trees import (
    MODEL_CONFIGS,
    build_regressor,
    ground_truth_support,
    support_recovery_metrics,
)


def test_model_factory_builds_both_algorithms():
    rf = build_regressor(MODEL_CONFIGS[0], random_seed=1, n_jobs=1)
    et = build_regressor(MODEL_CONFIGS[-1], random_seed=1, n_jobs=1)
    assert rf.__class__.__name__ == "RandomForestRegressor"
    assert et.__class__.__name__ == "ExtraTreesRegressor"


def test_support_metrics_are_exact_for_perfect_ranking():
    importance = np.zeros((4, 2))
    importance[1, 0] = 1.0
    importance[2, 1] = 1.0
    support = {0: {1}, 1: {2}}
    metrics = support_recovery_metrics(importance, support)
    assert metrics["support_map"] == 1.0
    assert metrics["support_recall_at_true_k"] == 1.0
    assert metrics["mapping_recall_at_3"] == 1.0


def test_ground_truth_is_inverted_by_target():
    ground_truth = {
        3: {"env_var_idx": 0},
        "5": {"env_var_idx": 1},
        7: {"env_var_idx": 1},
    }
    assert ground_truth_support(ground_truth, 3) == {
        0: {3},
        1: {5, 7},
        2: set(),
    }
