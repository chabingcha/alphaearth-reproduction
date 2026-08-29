"""Download GBA AlphaEarth embeddings using the paper's GEE sampling design.

Rahman (2026) describes extraction at 1 km scale with a 500 m point buffer.
This script reproduces that support directly in Earth Engine, independently
of the 640 m COG-overview dataset used for the fast regional baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POINTS = ROOT / "data" / "processed" / "gba_aef_2017_2023.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "gba_embeddings_gee_1km"
BANDS = [f"A{dimension:02d}" for dimension in range(64)]


def buffered_collection(points: pd.DataFrame, radius_m: int) -> ee.FeatureCollection:
    features = []
    for point in points.itertuples(index=False):
        geometry = ee.Geometry.Point(
            [float(point.lon), float(point.lat)]
        ).buffer(radius_m, 1)
        features.append(
            ee.Feature(geometry, {"point_id": int(point.point_id)})
        )
    return ee.FeatureCollection(features)


def download_chunk(
    year: int,
    points: pd.DataFrame,
    output: Path,
    scale_m: int,
    radius_m: int,
) -> pd.DataFrame:
    regions = buffered_collection(points, radius_m)
    image = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(regions.geometry().bounds(1))
        .mosaic()
        .select(BANDS)
    )
    samples = image.reduceRegions(
        collection=regions,
        reducer=ee.Reducer.mean(),
        scale=scale_m,
        tileScale=4,
    )
    url = samples.getDownloadURL(
        filetype="CSV",
        selectors=["point_id", *BANDS],
        filename=output.stem,
    )
    response = None
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=1800)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 4:
                raise
            delay = 2 ** attempt
            print(f"    Network retry in {delay}s ({attempt + 1}/4)", flush=True)
            time.sleep(delay)
    assert response is not None
    csv_path = output.with_suffix(".csv")
    csv_path.write_bytes(response.content)

    values = pd.read_csv(csv_path)
    if values["point_id"].duplicated().any():
        raise RuntimeError(f"Duplicate point IDs returned for {output}")
    values.insert(1, "year", year)
    table = points.merge(values, on="point_id", how="left", validate="one_to_one")
    table = table[[*points.columns, "year", *BANDS]]
    table.to_parquet(output, index=False)
    return table


def download_year(
    year: int,
    points: pd.DataFrame,
    output: Path,
    scale_m: int,
    radius_m: int,
    chunk_size: int,
    region_slug: str,
) -> dict:
    chunk_dir = output.parent / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        chunk_output = chunk_dir / f"{region_slug}_aef_gee_{year}_{start:05d}_{stop:05d}.parquet"
        expected_ids = points.iloc[start:stop]["point_id"].tolist()
        if chunk_output.exists():
            chunk = pd.read_parquet(chunk_output)
            if chunk["point_id"].tolist() != expected_ids:
                raise RuntimeError(f"Checkpoint point IDs do not match: {chunk_output}")
            print(f"  Reusing {chunk_output.name}", flush=True)
        else:
            print(f"  Downloading points {start:,}:{stop:,}", flush=True)
            chunk = download_chunk(
                year,
                points.iloc[start:stop].copy(),
                chunk_output,
                scale_m,
                radius_m,
            )
        chunks.append(chunk)

    table = pd.concat(chunks, ignore_index=True)
    if len(table) != len(points) or table["point_id"].duplicated().any():
        raise RuntimeError(f"Incomplete or duplicate data after merging {year}")
    table.to_parquet(output, index=False)
    matrix = table[BANDS].to_numpy(dtype=float)
    norms = np.linalg.norm(matrix, axis=1)
    return {
        "year": year,
        "rows": int(len(table)),
        "missing_values": int(np.isnan(matrix).sum()),
        "mean_vector_norm": float(np.nanmean(norms)),
        "min_vector_norm": float(np.nanmin(norms)),
        "max_vector_norm": float(np.nanmax(norms)),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--years", default="2017:2023")
    parser.add_argument("--scale-m", type=int, default=1000)
    parser.add_argument("--radius-m", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--region-slug", default="gba")
    args = parser.parse_args()
    if not args.project:
        raise SystemExit("Pass --project YOUR_GCP_PROJECT or set EE_PROJECT")
    ee.Initialize(project=args.project)

    start_year, end_year = (int(value) for value in args.years.split(":", 1))
    years = list(range(start_year, end_year + 1))
    points = (
        pd.read_parquet(args.points, columns=["point_id", "lon", "lat", "city"])
        .drop_duplicates("point_id")
        .sort_values("point_id")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for year in years:
        output = args.output_dir / f"{args.region_slug}_aef_gee_1km_{year}.parquet"
        print(f"Downloading paper-scale embeddings for {year}", flush=True)
        summary = download_year(
            year,
            points,
            output,
            args.scale_m,
            args.radius_m,
            args.chunk_size,
            args.region_slug,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    combined = pd.concat(
        [
            pd.read_parquet(args.output_dir / f"{args.region_slug}_aef_gee_1km_{year}.parquet")
            for year in years
        ],
        ignore_index=True,
    )
    combined_output = (
        args.output_dir.parent
        / f"{args.region_slug}_aef_gee_1km_{start_year}_{end_year}.parquet"
    )
    combined.to_parquet(combined_output, index=False)
    provenance = {
        "source": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
        "region_slug": args.region_slug,
        "project": args.project,
        "sampling_scale_m": args.scale_m,
        "point_buffer_radius_m": args.radius_m,
        "buffer_area_km2": float(np.pi * args.radius_m ** 2 / 1_000_000),
        "renormalized_after_region_mean": False,
        "year_summaries": summaries,
        "combined_output": str(combined_output),
    }
    combined_output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Saved {combined_output}", flush=True)


if __name__ == "__main__":
    main()
