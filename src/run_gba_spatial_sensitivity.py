"""Check whether GBA spatial-CV conclusions depend on block size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from src.run_gba_real_analysis import (
    EMBEDDING_COLUMNS,
    ROOT,
    block_groups,
    load_aligned,
)


DEFAULT_VARIABLES = [
    "elevation",
    "tree_cover",
    "ndvi_mean",
    "lai_mean",
    "annual_precip",
    "impervious_surface",
    "nighttime_lights",
    "population_density",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "data" / "processed" / "gba_aef_2017_2023.parquet",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=ROOT / "data" / "processed" / "gba_environment_2017_2023.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "gba_real" / "spatial_sensitivity.json",
    )
    parser.add_argument("--max-samples", type=int, default=30000)
    parser.add_argument("--trees", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-sizes", default="0.25,0.5,1.0")
    args = parser.parse_args()

    frame = load_aligned(args.embeddings, args.environment)
    block_sizes = [float(value) for value in args.block_sizes.split(",")]
    output = {
        "settings": {
            "variables": DEFAULT_VARIABLES,
            "max_samples": args.max_samples,
            "trees": args.trees,
            "folds": args.folds,
            "seed": args.seed,
            "block_sizes_deg": block_sizes,
        },
        "results": {},
    }

    for variable in DEFAULT_VARIABLES:
        data = frame[["lon", "lat", *EMBEDDING_COLUMNS, variable]].dropna()
        if len(data) > args.max_samples:
            data = data.sample(args.max_samples, random_state=args.seed)
        x = data[EMBEDDING_COLUMNS].to_numpy(dtype=np.float32)
        y = data[variable].to_numpy(dtype=np.float32)
        output["results"][variable] = {}
        for block_size in block_sizes:
            groups = block_groups(data, block_size)
            splitter = GroupKFold(n_splits=args.folds)
            scores = []
            for train, test in splitter.split(x, y, groups):
                model = RandomForestRegressor(
                    n_estimators=args.trees,
                    min_samples_leaf=2,
                    max_features=1.0,
                    random_state=args.seed,
                    n_jobs=-1,
                ).fit(x[train], y[train])
                scores.append(float(r2_score(y[test], model.predict(x[test]))))
            output["results"][variable][str(block_size)] = {
                "groups": int(np.unique(groups).size),
                "fold_scores": scores,
                "mean_r2": float(np.mean(scores)),
                "std_r2": float(np.std(scores)),
            }
            print(
                f"{variable} {block_size:g} deg: "
                f"R2={np.mean(scores):.3f} +/- {np.std(scores):.3f}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
