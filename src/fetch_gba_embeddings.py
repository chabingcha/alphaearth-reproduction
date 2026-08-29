"""Fetch real AlphaEarth Foundation embeddings for the Greater Bay Area.

The source is the public Source Cooperative mirror of Google's annual
Satellite Embedding dataset.  The script samples the pre-built COG overview
levels, applies Google's documented int8 de-quantization, and writes one
analysis-ready Parquet table for 2017-2023.

No Earth Engine account or cloud credentials are required for this step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from shapely import contains_xy


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = ROOT / "data" / "raw" / "aef_index.parquet"
DEFAULT_BOUNDARY = ROOT / "data" / "raw" / "gba_boundary_gadm41.geojson"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "gba_aef_2017_2023.parquet"

SOURCE_COOP_PREFIX = "s3://us-west-2.opendata.source.coop/"
SOURCE_COOP_HTTPS = "https://data.source.coop/"


def parse_years(value: str) -> list[int]:
    if ":" in value:
        start, end = (int(part) for part in value.split(":", 1))
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def make_sampling_grid(boundary_path: Path, spacing_deg: float) -> pd.DataFrame:
    cities = gpd.read_file(boundary_path).to_crs(4326)
    region = cities.geometry.union_all()
    west, south, east, north = region.bounds

    lons = np.arange(west + spacing_deg / 2, east, spacing_deg)
    lats = np.arange(south + spacing_deg / 2, north, spacing_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()
    inside = contains_xy(region, lon_flat, lat_flat)

    points = gpd.GeoDataFrame(
        {
            "point_id": np.arange(int(inside.sum()), dtype=np.int32),
            "lon": lon_flat[inside],
            "lat": lat_flat[inside],
        },
        geometry=gpd.points_from_xy(lon_flat[inside], lat_flat[inside]),
        crs=4326,
    )
    city_lookup = gpd.sjoin(
        points,
        cities[["city", "geometry"]],
        how="left",
        predicate="within",
    )
    city_lookup = city_lookup.drop_duplicates("point_id")
    return pd.DataFrame(city_lookup.drop(columns=["geometry", "index_right"]))


def intersecting_tiles(index: pd.DataFrame, bounds: tuple[float, ...], years: list[int]) -> pd.DataFrame:
    west, south, east, north = bounds
    return index[
        index["year"].isin(years)
        & (index["wgs84_east"] >= west)
        & (index["wgs84_west"] <= east)
        & (index["wgs84_north"] >= south)
        & (index["wgs84_south"] <= north)
    ].copy()


def sample_tile(
    row: pd.Series,
    points: pd.DataFrame,
    target: np.ndarray,
    overview_factor: int,
) -> int:
    candidate = (
        target[:, 0] == -128
    ) & (
        points["lon"].to_numpy() >= row.wgs84_west
    ) & (
        points["lon"].to_numpy() <= row.wgs84_east
    ) & (
        points["lat"].to_numpy() >= row.wgs84_south
    ) & (
        points["lat"].to_numpy() <= row.wgs84_north
    )
    candidate_idx = np.flatnonzero(candidate)
    if candidate_idx.size == 0:
        return 0

    transformer = Transformer.from_crs("EPSG:4326", row.crs, always_xy=True)
    x, y = transformer.transform(
        points.loc[candidate_idx, "lon"].to_numpy(),
        points.loc[candidate_idx, "lat"].to_numpy(),
    )

    with rasterio.open(row.path) as dataset:
        rr, cc = rowcol(dataset.transform, x, y)
        rr = np.asarray(rr)
        cc = np.asarray(cc)
        in_raster = (
            (rr >= 0) & (rr < dataset.height) & (cc >= 0) & (cc < dataset.width)
        )
        if not in_raster.any():
            return 0

        out_height = max(1, dataset.height // overview_factor)
        out_width = max(1, dataset.width // overview_factor)
        overview = dataset.read(
            out_shape=(dataset.count, out_height, out_width),
            resampling=Resampling.nearest,
        )
        rr_over = np.minimum(
            (rr[in_raster] * out_height // dataset.height).astype(int),
            out_height - 1,
        )
        cc_over = np.minimum(
            (cc[in_raster] * out_width // dataset.width).astype(int),
            out_width - 1,
        )
        values = overview[:, rr_over, cc_over].T

    valid = np.all(values != -128, axis=1)
    selected_idx = candidate_idx[in_raster][valid]
    target[selected_idx] = values[valid]
    return int(valid.sum())


def dequantize(values: np.ndarray) -> np.ndarray:
    values_float = values.astype(np.float32)
    return ((values_float / 127.5) ** 2) * np.sign(values_float)


def fetch_embeddings(
    index_path: Path,
    boundary_path: Path,
    output_path: Path,
    years: list[int],
    spacing_deg: float,
    overview_factor: int,
    tile_start: int = 0,
    max_tiles: int | None = None,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = make_sampling_grid(boundary_path, spacing_deg)
    boundary = gpd.read_file(boundary_path).to_crs(4326)
    index = pd.read_parquet(
        index_path,
        columns=[
            "year", "path", "crs", "wgs84_west", "wgs84_south",
            "wgs84_east", "wgs84_north",
        ],
    )
    tiles = intersecting_tiles(index, tuple(boundary.total_bounds), years)

    print(f"GBA grid points: {len(points):,}")
    print(f"Intersecting COGs: {len(tiles):,} ({len(tiles) // len(years)} per year)")
    print(f"Sampling resolution: {overview_factor * 10} m overview")

    tables: list[pd.DataFrame] = []
    year_stats: dict[str, dict] = {}
    rasterio_env = {
        "AWS_NO_SIGN_REQUEST": "YES",
        "AWS_REGION": "us-west-2",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MULTIRANGE": "YES",
    }

    with rasterio.Env(**rasterio_env):
        for year in years:
            started = time.time()
            values_raw = np.full((len(points), 64), -128, dtype=np.int16)
            year_tiles = tiles[tiles["year"] == year].iloc[tile_start:]
            if max_tiles is not None:
                year_tiles = year_tiles.head(max_tiles)

            for tile_number, (_, row) in enumerate(year_tiles.iterrows(), start=1):
                filled = sample_tile(row, points, values_raw, overview_factor)
                print(
                    f"  {year} tile {tile_number:02d}/{len(year_tiles):02d}: "
                    f"{filled:,} new points",
                    flush=True,
                )

            valid = np.all(values_raw != -128, axis=1)
            values = dequantize(values_raw[valid])
            norms = np.linalg.norm(values, axis=1)
            table = points.loc[valid].reset_index(drop=True).copy()
            table.insert(1, "year", year)
            for dim in range(64):
                table[f"A{dim:02d}"] = values[:, dim]
            tables.append(table)

            mean_norm = float(norms.mean()) if len(norms) else None
            year_stats[str(year)] = {
                "valid_points": int(valid.sum()),
                "missing_points": int((~valid).sum()),
                "mean_embedding_norm": mean_norm,
                "elapsed_seconds": round(time.time() - started, 2),
            }
            norm_label = f"{mean_norm:.4f}" if mean_norm is not None else "n/a"
            print(
                f"  {year}: {valid.sum():,} valid, {(~valid).sum():,} missing, "
                f"mean norm={norm_label}, {time.time() - started:.1f}s",
                flush=True,
            )

    combined = pd.concat(tables, ignore_index=True)
    combined.to_parquet(output_path, index=False)

    provenance = {
        "dataset": "AlphaEarth Foundations Satellite Embedding V1 Annual",
        "producer": "Google and Google DeepMind",
        "mirror": "Source Cooperative / Taylor Geospatial",
        "license": "CC-BY-4.0",
        "source_index": "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet",
        "years": years,
        "spacing_deg": spacing_deg,
        "native_resolution_m": 10,
        "sampled_overview_resolution_m": overview_factor * 10,
        "tile_start": tile_start,
        "max_tiles": max_tiles,
        "boundary": str(boundary_path),
        "output": str(output_path),
        "rows": int(len(combined)),
        "year_stats": year_stats,
        "dequantization": "((int8 / 127.5) ** 2) * sign(int8)",
    }
    provenance_path = output_path.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Saved {len(combined):,} rows to {output_path}")
    print(f"Saved provenance to {provenance_path}")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--years", default="2017:2023")
    parser.add_argument("--spacing-deg", type=float, default=0.025)
    parser.add_argument("--overview-factor", type=int, default=64)
    parser.add_argument("--tile-start", type=int, default=0)
    parser.add_argument("--max-tiles", type=int)
    args = parser.parse_args()
    fetch_embeddings(
        args.index,
        args.boundary,
        args.output,
        parse_years(args.years),
        args.spacing_deg,
        args.overview_factor,
        args.tile_start,
        args.max_tiles,
    )


if __name__ == "__main__":
    main()
