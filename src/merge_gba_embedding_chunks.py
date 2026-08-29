"""Merge resumable AlphaEarth COG sampling chunks into one clean table."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def merge_chunks(pattern: str, output: Path) -> dict:
    files = [Path(path) for path in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No chunk files matched {pattern!r}")

    tables = [pd.read_parquet(path) for path in files]
    combined = pd.concat(tables, ignore_index=True)
    duplicate_rows = int(combined.duplicated(["year", "point_id"]).sum())
    combined = (
        combined.drop_duplicates(["year", "point_id"])
        .sort_values(["year", "point_id"])
        .reset_index(drop=True)
    )

    embedding_columns = [f"A{dim:02d}" for dim in range(64)]
    missing_columns = sorted(set(embedding_columns) - set(combined.columns))
    if missing_columns:
        raise ValueError(f"Missing embedding columns: {missing_columns}")
    if combined[embedding_columns].isna().any().any():
        raise ValueError("Merged embeddings contain missing values")

    norms = np.linalg.norm(combined[embedding_columns].to_numpy(), axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)

    summary = {
        "input_files": [str(path) for path in files],
        "output": str(output),
        "rows": int(len(combined)),
        "years": sorted(int(year) for year in combined["year"].unique()),
        "locations_per_year": {
            str(int(year)): int(count)
            for year, count in combined.groupby("year")["point_id"].nunique().items()
        },
        "duplicate_rows_removed": duplicate_rows,
        "embedding_norm": {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "min": float(norms.min()),
            "max": float(norms.max()),
        },
        "cities": {
            str(city): int(count)
            for city, count in combined.groupby("city")["point_id"].nunique().items()
        },
    }
    summary_path = output.with_suffix(".merge.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default="data/processed/gba_aef_20??_chunk*.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/gba_aef_2017_2023.parquet"),
    )
    args = parser.parse_args()
    merge_chunks(args.pattern, args.output)


if __name__ == "__main__":
    main()
