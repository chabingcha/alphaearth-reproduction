"""Train a compact multi-task Transformer on the strict GBA reproduction data.

The original paper used a much larger TabTransformer (about five million
training rows).  The GBA dataset has only 54,614 annual observations, so this
implementation keeps the same multi-task/attention idea while reducing model
capacity.  It reports both random row-wise CV (paper-comparable) and spatial
block CV, then measures per-dimension importance by permutation on a held-out
fold.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from torch import nn

from src.fetch_gba_environment_gee import VARIABLES
from src.run_gba_real_analysis import EMBEDDING_COLUMNS, block_groups, load_aligned


ROOT = Path(__file__).resolve().parent.parent


class MultiTaskEmbeddingTransformer(nn.Module):
    """Treat each scalar AlphaEarth dimension as a dimension-aware token."""

    def __init__(
        self,
        n_features: int = 64,
        n_targets: int = 26,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.value_projection = nn.Linear(1, d_model)
        self.dimension_embedding = nn.Parameter(
            torch.empty(1, n_features, d_model)
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, n_targets),
        )
        nn.init.normal_(self.dimension_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.value_projection(x.unsqueeze(-1)) + self.dimension_embedding
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0])


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_config(args: argparse.Namespace) -> dict:
    return {
        "n_features": len(EMBEDDING_COLUMNS),
        "n_targets": len(VARIABLES),
        "d_model": args.d_model,
        "n_heads": args.heads,
        "n_layers": args.layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
    }


def fit_scalers(x: np.ndarray, y: np.ndarray, train: np.ndarray) -> tuple:
    x_mean = x[train].mean(axis=0)
    x_std = x[train].std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_mean = np.nanmean(y[train], axis=0)
    y_std = np.nanstd(y[train], axis=0)
    y_std[y_std < 1e-6] = 1.0
    return x_mean, x_std, y_mean, y_std


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(target)
    safe_target = torch.nan_to_num(target)
    squared = (pred - safe_target).square()
    return squared[valid].mean()


@torch.no_grad()
def predict_batches(
    model: nn.Module,
    x: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(indices), batch_size):
        idx = torch.as_tensor(indices[start : start + batch_size], device=x.device)
        output.append(model(x[idx]).cpu().numpy())
    return np.concatenate(output, axis=0)


def variable_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    scores = np.full(y_true.shape[1], np.nan)
    for j in range(y_true.shape[1]):
        valid = np.isfinite(y_true[:, j]) & np.isfinite(y_pred[:, j])
        if valid.sum() >= 3 and np.std(y_true[valid, j]) > 0:
            scores[j] = r2_score(y_true[valid, j], y_pred[valid, j])
    return scores


def train_fold(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    fold_seed: int,
) -> tuple[dict, dict]:
    set_seed(fold_seed)
    x_mean, x_std, y_mean, y_std = fit_scalers(x_raw, y_raw, train)
    x_scaled = ((x_raw - x_mean) / x_std).astype(np.float32)
    y_scaled = ((y_raw - y_mean) / y_std).astype(np.float32)
    x = torch.as_tensor(x_scaled, device=device)
    y = torch.as_tensor(y_scaled, device=device)

    model = MultiTaskEmbeddingTransformer(**model_config(args)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_loss = math.inf
    best_state = None
    patience_left = args.patience
    history = []

    for epoch in range(args.epochs):
        model.train()
        shuffled = np.random.default_rng(fold_seed + epoch).permutation(train)
        losses = []
        for start in range(0, len(shuffled), args.batch_size):
            idx = torch.as_tensor(
                shuffled[start : start + args.batch_size], device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(x[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        test_prediction = predict_batches(model, x, test, args.batch_size)
        test_target = y_scaled[test]
        valid = np.isfinite(test_target)
        validation_loss = float(
            np.mean((test_prediction[valid] - test_target[valid]) ** 2)
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - args.min_delta:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    prediction_scaled = predict_batches(model, x, test, args.batch_size)
    prediction = prediction_scaled * y_std + y_mean
    scores = variable_r2(y_raw[test], prediction)
    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "epochs_trained": len(history),
        "best_standardized_mse": best_loss,
        "r2": {name: float(scores[j]) for j, name in enumerate(VARIABLES)},
        "mean_r2": float(np.nanmean(scores)),
        "history": history,
    }
    state = {
        "model": model,
        "x_tensor": x,
        "x_scaled": x_scaled,
        "y_scaled": y_scaled,
        "test": test,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return metrics, state


def permutation_importance(
    fold_state: dict,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    """Return a 64 x 26 normalized held-out permutation-importance matrix."""
    model = fold_state["model"]
    x_tensor = fold_state["x_tensor"]
    x_scaled = fold_state["x_scaled"]
    y_scaled = fold_state["y_scaled"]
    rng = np.random.default_rng(seed)
    test = fold_state["test"]
    if len(test) > args.importance_samples:
        test = rng.choice(test, args.importance_samples, replace=False)
    baseline = predict_batches(model, x_tensor, test, args.batch_size)
    target = y_scaled[test]
    valid = np.isfinite(target)
    baseline_mse = np.array(
        [
            np.mean((baseline[valid[:, j], j] - target[valid[:, j], j]) ** 2)
            for j in range(len(VARIABLES))
        ]
    )
    importance = np.zeros((len(EMBEDDING_COLUMNS), len(VARIABLES)), dtype=float)
    x_subset = x_scaled[test].copy()
    for dim in range(len(EMBEDDING_COLUMNS)):
        permuted = x_subset.copy()
        permuted[:, dim] = permuted[rng.permutation(len(permuted)), dim]
        permuted_tensor = torch.as_tensor(permuted, device=x_tensor.device)
        indices = np.arange(len(permuted), dtype=int)
        prediction = predict_batches(model, permuted_tensor, indices, args.batch_size)
        for j in range(len(VARIABLES)):
            column_valid = valid[:, j]
            mse = np.mean(
                (prediction[column_valid, j] - target[column_valid, j]) ** 2
            )
            importance[dim, j] = max(0.0, mse - baseline_mse[j])
    column_sums = importance.sum(axis=0, keepdims=True)
    return np.divide(
        importance,
        column_sums,
        out=np.zeros_like(importance),
        where=column_sums > 0,
    )


def aggregate_folds(folds: list[dict]) -> tuple[dict, np.ndarray]:
    matrix = np.asarray(
        [[fold["r2"][name] for name in VARIABLES] for fold in folds], dtype=float
    )
    per_variable = {}
    for j, name in enumerate(VARIABLES):
        per_variable[name] = {
            "r2_mean": float(np.nanmean(matrix[:, j])),
            "r2_std": float(np.nanstd(matrix[:, j])),
            "fold_scores": matrix[:, j].tolist(),
        }
    return per_variable, matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "data/processed/gba_aef_gee_1km_2017_2023.parquet",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=ROOT / "data/processed/gba_environment_point_10m_2017_2023.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/gba_real/transformer_results.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results/gba_real/gba_transformer.pt",
    )
    parser.add_argument("--max-samples", type=int, default=30000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--block-size-deg", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--importance-samples", type=int, default=3000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    frame = load_aligned(args.embeddings, args.environment)
    if args.max_samples and len(frame) > args.max_samples:
        frame = frame.sample(args.max_samples, random_state=args.seed).sort_index()
    frame = frame.reset_index(drop=True)
    x = frame[EMBEDDING_COLUMNS].to_numpy(dtype=np.float32)
    y = frame[VARIABLES].to_numpy(dtype=np.float32)
    groups = block_groups(frame, args.block_size_deg)
    print(
        f"Transformer data={len(frame):,}, device={device}, "
        f"spatial_groups={len(np.unique(groups))}",
        flush=True,
    )

    start_time = time.time()
    random_folds = []
    importance_state = None
    random_splitter = KFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    for fold_index, (train, test) in enumerate(random_splitter.split(x), start=1):
        metrics, state = train_fold(
            x, y, train, test, args, device, args.seed + fold_index
        )
        random_folds.append(metrics)
        if fold_index == 1:
            importance_state = state
        print(
            f"  random fold {fold_index}/{args.folds}: "
            f"mean R2={metrics['mean_r2']:.3f}, epochs={metrics['epochs_trained']}",
            flush=True,
        )
        if fold_index != 1:
            del state
        if device.type == "mps":
            torch.mps.empty_cache()

    spatial_folds = []
    spatial_splitter = GroupKFold(n_splits=args.folds)
    for fold_index, (train, test) in enumerate(
        spatial_splitter.split(x, y, groups), start=1
    ):
        metrics, state = train_fold(
            x, y, train, test, args, device, args.seed + 100 + fold_index
        )
        spatial_folds.append(metrics)
        print(
            f"  spatial fold {fold_index}/{args.folds}: "
            f"mean R2={metrics['mean_r2']:.3f}, epochs={metrics['epochs_trained']}",
            flush=True,
        )
        del state
        if device.type == "mps":
            torch.mps.empty_cache()

    assert importance_state is not None
    importance = permutation_importance(importance_state, args, args.seed + 1000)
    random_per_variable, random_matrix = aggregate_folds(random_folds)
    spatial_per_variable, spatial_matrix = aggregate_folds(spatial_folds)
    per_variable = {}
    for j, name in enumerate(VARIABLES):
        top = np.argsort(importance[:, j])[-3:][::-1]
        per_variable[name] = {
            "random_cv_r2_mean": random_per_variable[name]["r2_mean"],
            "random_cv_r2_std": random_per_variable[name]["r2_std"],
            "spatial_cv_r2_mean": spatial_per_variable[name]["r2_mean"],
            "spatial_cv_r2_std": spatial_per_variable[name]["r2_std"],
            "delta_r2": random_per_variable[name]["r2_mean"]
            - spatial_per_variable[name]["r2_mean"],
            "top3_dimensions": [EMBEDDING_COLUMNS[i] for i in top],
        }
    random_means = np.nanmean(random_matrix, axis=0)
    spatial_means = np.nanmean(spatial_matrix, axis=0)
    top10 = np.argsort(random_means)[-10:]
    output = {
        "method": "compact multi-task scalar-token Transformer",
        "paper_alignment": {
            "same": [
                "64 AlphaEarth input dimensions",
                "26 simultaneous environmental targets",
                "attention-based nonlinear model",
                "5-fold random and spatial validation",
            ],
            "different": [
                "30,000 GBA rows rather than about 5 million CONUS rows",
                "2 encoder layers / 4 heads rather than paper's 4 layers / 8 heads",
                "0.5-degree blocks rather than 2-degree blocks",
            ],
        },
        "settings": {
            **vars(args),
            "embeddings": str(args.embeddings),
            "environment": str(args.environment),
            "output": str(args.output),
            "checkpoint": str(args.checkpoint),
            "device_used": str(device),
            "rows_used": int(len(frame)),
            "spatial_groups": int(len(np.unique(groups))),
        },
        "per_variable": per_variable,
        "random_folds": random_folds,
        "spatial_folds": spatial_folds,
        "permutation_importance_matrix": importance.tolist(),
        "summary": {
            "mean_random_cv_r2": float(np.nanmean(random_means)),
            "mean_spatial_cv_r2": float(np.nanmean(spatial_means)),
            "mean_spatial_delta_r2": float(
                np.nanmean(random_means - spatial_means)
            ),
            "top10_mean_spatial_delta_r2": float(
                np.nanmean((random_means - spatial_means)[top10])
            ),
            "variables_random_r2_gt_0_9": int(np.sum(random_means > 0.9)),
            "variables_random_r2_gt_0_7": int(np.sum(random_means > 0.7)),
            "runtime_seconds": time.time() - start_time,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serializable = copy.deepcopy(output)
    serializable["settings"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in serializable["settings"].items()
    }
    args.output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model_config(args),
            "model_state_dict": importance_state["model"].state_dict(),
            "x_mean": importance_state["x_mean"],
            "x_std": importance_state["x_std"],
            "y_mean": importance_state["y_mean"],
            "y_std": importance_state["y_std"],
            "embedding_columns": EMBEDDING_COLUMNS,
            "target_columns": VARIABLES,
        },
        args.checkpoint,
    )
    print(json.dumps(output["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
