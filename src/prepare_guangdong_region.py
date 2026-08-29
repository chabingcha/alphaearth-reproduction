"""Prepare the Guangdong prefecture boundary and 0.025-degree sampling grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd

from src.fetch_gba_embeddings import make_sampling_grid


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gadm", type=Path, default=ROOT / "data/raw/gadm41_CHN.gpkg"
    )
    parser.add_argument(
        "--boundary", type=Path,
        default=ROOT / "data/raw/guangdong_boundary_gadm41.geojson",
    )
    parser.add_argument(
        "--points", type=Path,
        default=ROOT / "data/processed/guangdong_sampling_points_0025deg.parquet",
    )
    parser.add_argument(
        "--summary", type=Path,
        default=ROOT / "data/processed/guangdong_sampling_points_0025deg.summary.json",
    )
    parser.add_argument("--spacing-deg", type=float, default=0.025)
    args = parser.parse_args()

    prefectures = gpd.read_file(args.gadm, layer="ADM_ADM_2")
    prefectures = prefectures[prefectures["NAME_1"] == "Guangdong"].copy()
    if len(prefectures) != 21:
        raise RuntimeError(f"Expected 21 Guangdong prefectures, found {len(prefectures)}")
    prefectures["city"] = prefectures["NAME_2"]
    boundary = prefectures[["GID_2", "city", "geometry"]].to_crs(4326)
    args.boundary.parent.mkdir(parents=True, exist_ok=True)
    boundary.to_file(args.boundary, driver="GeoJSON")

    points = make_sampling_grid(args.boundary, args.spacing_deg)
    if points["city"].isna().any():
        missing = int(points["city"].isna().sum())
        raise RuntimeError(f"{missing} sampling points were not assigned to a prefecture")
    points = points[["point_id", "lon", "lat", "city"]].sort_values("point_id")
    args.points.parent.mkdir(parents=True, exist_ok=True)
    points.to_parquet(args.points, index=False)

    equal_area = boundary.to_crs("EPSG:6933")
    west, south, east, north = boundary.total_bounds
    counts = points.groupby("city").size().sort_values(ascending=False)
    summary = {
        "region": "Guangdong Province, China",
        "administrative_scope": "21 prefecture-level cities; Hong Kong and Macau excluded",
        "gadm_version": "4.1",
        "gadm_level": 2,
        "spacing_deg": args.spacing_deg,
        "area_km2": float(equal_area.geometry.area.sum() / 1_000_000),
        "bounds_wgs84": [float(west), float(south), float(east), float(north)],
        "prefectures": int(len(boundary)),
        "points_per_year": int(len(points)),
        "rows_2017_2023": int(len(points) * 7),
        "points_per_city": {str(key): int(value) for key, value in counts.items()},
        "boundary": str(args.boundary),
        "points": str(args.points),
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
