"""Download 26 real environmental variables for GBA sample points via GEE.

The variables follow Rahman (2026) as closely as global coverage permits.
PRISM and NLCD are United-States-only, so the GBA reproduction uses
ERA5-Land for air temperature/dew point/precipitation and a 500 m
neighbourhood fraction of ESA WorldCover built-up pixels as the impervious
surface proxy.  Every substitution is recorded in the provenance output.
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
DEFAULT_EMBEDDINGS = ROOT / "data" / "processed" / "gba_aef_2017_2023.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "gba_environment"
NODATA_SENTINEL = -9999.0

VARIABLES = [
    "elevation", "slope", "aspect", "flow_accumulation",
    "clay_fraction", "organic_carbon", "soil_ph", "water_capacity",
    "ndvi_mean", "ndvi_max", "evi_mean", "lai_mean", "tree_cover", "albedo",
    "lst_daytime", "lst_nighttime", "air_temp_mean", "dew_point_temp",
    "annual_precip", "max_monthly_precip",
    "soil_moisture", "annual_runoff", "annual_et",
    "impervious_surface", "nighttime_lights", "population_density",
]
REDUCTION_BANDS = [
    variable for variable in VARIABLES if variable != "aspect"
] + ["aspect_sin", "aspect_cos"]

DATA_SOURCES = {
    "AlphaEarth points": "Source Cooperative tge-labs/aef/v1/annual",
    "elevation/slope/aspect": "USGS/SRTMGL1_003",
    "flow_accumulation": "WWF/HydroSHEDS/15ACC",
    "soil": "OpenLandMap SOL global 250 m products",
    "NDVI/EVI": "MODIS/061/MOD13A2",
    "LAI": "MODIS/061/MOD15A2H",
    "tree_cover": "UMD/hansen/global_forest_change_2025_v1_13",
    "albedo": "MODIS/061/MCD43A3",
    "LST": "MODIS/061/MOD11A2",
    "air temperature/dew point/precipitation/hydrology": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
    "impervious_surface": "ESA/WorldCover/v200 built-up fraction in 500 m neighbourhood",
    "nighttime_lights": "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG",
    "population_density": "CIESIN/GPWv411/GPW_Population_Density (2020)",
}

SUBSTITUTIONS = {
    "air_temp_mean": "ERA5-Land replaces US-only PRISM tmean",
    "dew_point_temp": "ERA5-Land replaces US-only PRISM tdmean",
    "annual_precip": "ERA5-Land replaces US-only PRISM precipitation",
    "max_monthly_precip": "ERA5-Land replaces US-only PRISM precipitation",
    "impervious_surface": "ESA WorldCover built-up fraction replaces US-only NLCD impervious percentage",
}


def _rename(image: ee.Image, name: str) -> ee.Image:
    return ee.Image(image).rename(name).toFloat()


def static_image() -> ee.Image:
    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation")
    terrain = ee.Algorithms.Terrain(elevation)
    flow = ee.Image("WWF/HydroSHEDS/15ACC").select("b1").add(1).log()

    clay = ee.Image(
        "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"
    ).select("b0")
    organic_carbon = ee.Image(
        "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02"
    ).select("b0").divide(5)
    soil_ph = ee.Image(
        "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02"
    ).select("b0").divide(10)
    water_capacity = ee.Image(
        "OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01"
    ).select("b0")

    tree_cover = ee.Image(
        "UMD/hansen/global_forest_change_2025_v1_13"
    ).select("treecover2000")
    population = (
        ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density")
        .filterDate("2020-01-01", "2021-01-01")
        .first()
        .select("population_density")
    )
    built = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").eq(50)
    built_fraction = built.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.square(radius=250, units="meters"),
    ).multiply(100)

    return ee.Image.cat([
        _rename(elevation, "elevation"),
        _rename(terrain.select("slope"), "slope"),
        _rename(terrain.select("aspect"), "aspect"),
        _rename(flow, "flow_accumulation"),
        _rename(clay, "clay_fraction"),
        _rename(organic_carbon, "organic_carbon"),
        _rename(soil_ph, "soil_ph"),
        _rename(water_capacity, "water_capacity"),
        _rename(tree_cover, "tree_cover"),
        _rename(built_fraction, "impervious_surface"),
        _rename(population, "population_density"),
    ])


def mod13_image(year: int) -> ee.Image:
    collection = ee.ImageCollection("MODIS/061/MOD13A2").filterDate(
        f"{year}-01-01", f"{year + 1}-01-01"
    )

    def mask_quality(image: ee.Image) -> ee.Image:
        return image.updateMask(image.select("SummaryQA").lte(1))

    clean = collection.map(mask_quality)
    return ee.Image.cat([
        _rename(clean.select("NDVI").mean().multiply(0.0001), "ndvi_mean"),
        _rename(clean.select("NDVI").max().multiply(0.0001), "ndvi_max"),
        _rename(clean.select("EVI").mean().multiply(0.0001), "evi_mean"),
    ])


def lai_image(year: int) -> ee.Image:
    collection = ee.ImageCollection("MODIS/061/MOD15A2H").filterDate(
        f"{year}-01-01", f"{year + 1}-01-01"
    )

    def mask_fill(image: ee.Image) -> ee.Image:
        lai = image.select("Lai_500m")
        return lai.updateMask(lai.lte(100))

    return _rename(collection.map(mask_fill).mean().multiply(0.1), "lai_mean")


def albedo_image(year: int) -> ee.Image:
    collection = ee.ImageCollection("MODIS/061/MCD43A3").filterDate(
        f"{year}-01-01", f"{year + 1}-01-01"
    )

    def mask_quality(image: ee.Image) -> ee.Image:
        value = image.select("Albedo_WSA_shortwave")
        quality = image.select("BRDF_Albedo_Band_Mandatory_Quality_shortwave")
        return value.updateMask(quality.lte(1)).updateMask(value.lt(32767))

    return _rename(collection.map(mask_quality).mean().multiply(0.001), "albedo")


def lst_image(year: int) -> ee.Image:
    collection = ee.ImageCollection("MODIS/061/MOD11A2").filterDate(
        f"{year}-01-01", f"{year + 1}-01-01"
    )

    def clean_day(image: ee.Image) -> ee.Image:
        value = image.select("LST_Day_1km")
        quality = image.select("QC_Day").bitwiseAnd(3).lte(1)
        return value.updateMask(quality).updateMask(value.gt(0))

    def clean_night(image: ee.Image) -> ee.Image:
        value = image.select("LST_Night_1km")
        quality = image.select("QC_Night").bitwiseAnd(3).lte(1)
        return value.updateMask(quality).updateMask(value.gt(0))

    return ee.Image.cat([
        _rename(collection.map(clean_day).mean().multiply(0.02).subtract(273.15), "lst_daytime"),
        _rename(collection.map(clean_night).mean().multiply(0.02).subtract(273.15), "lst_nighttime"),
    ])


def era5_image(year: int) -> ee.Image:
    collection = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate(
        f"{year}-01-01", f"{year + 1}-01-01"
    )
    return ee.Image.cat([
        _rename(collection.select("temperature_2m").mean().subtract(273.15), "air_temp_mean"),
        _rename(collection.select("dewpoint_temperature_2m").mean().subtract(273.15), "dew_point_temp"),
        _rename(collection.select("total_precipitation_sum").sum().multiply(1000), "annual_precip"),
        _rename(collection.select("total_precipitation_sum").max().multiply(1000), "max_monthly_precip"),
        _rename(collection.select("volumetric_soil_water_layer_1").mean(), "soil_moisture"),
        _rename(collection.select("runoff_sum").sum().multiply(1000), "annual_runoff"),
        _rename(collection.select("total_evaporation_sum").sum().multiply(-1000), "annual_et"),
    ])


def nightlights_image(year: int) -> ee.Image:
    collection = ee.ImageCollection(
        "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
    ).filterDate(f"{year}-01-01", f"{year + 1}-01-01")

    def mask_coverage(image: ee.Image) -> ee.Image:
        return image.select("avg_rad").updateMask(image.select("cf_cvg").gt(0))

    return _rename(collection.map(mask_coverage).mean(), "nighttime_lights")


def environmental_image(year: int) -> ee.Image:
    return ee.Image.cat([
        static_image(),
        mod13_image(year),
        lai_image(year),
        albedo_image(year),
        lst_image(year),
        era5_image(year),
        nightlights_image(year),
    ]).select(VARIABLES)


def point_collection(points: pd.DataFrame) -> ee.FeatureCollection:
    features = []
    for point in points.itertuples(index=False):
        features.append(
            ee.Feature(
                ee.Geometry.Point([float(point.lon), float(point.lat)]),
                {"point_id": int(point.point_id)},
            )
        )
    return ee.FeatureCollection(features)


def reduction_image(year: int) -> ee.Image:
    """Prepare bands for footprint means, including circular aspect terms."""
    image = environmental_image(year)
    aspect_radians = image.select("aspect").multiply(np.pi / 180)
    return ee.Image.cat([
        image.select([variable for variable in VARIABLES if variable != "aspect"]),
        aspect_radians.sin().rename("aspect_sin"),
        aspect_radians.cos().rename("aspect_cos"),
    ]).select(REDUCTION_BANDS).resample("bilinear").reproject(
        crs="EPSG:3857",
        scale=30,
    )


def footprint_collection(
    points: pd.DataFrame,
    footprint_m: int,
) -> ee.FeatureCollection:
    features = []
    for point in points.itertuples(index=False):
        footprint = (
            ee.Geometry.Point([float(point.lon), float(point.lat)])
            .buffer(footprint_m / 2, 1)
            .bounds(1)
        )
        features.append(
            ee.Feature(footprint, {"point_id": int(point.point_id)})
        )
    return ee.FeatureCollection(features)


def download_chunk(
    year: int,
    points: pd.DataFrame,
    output: Path,
    footprint_m: int,
    sampling_mode: str,
) -> pd.DataFrame:
    if sampling_mode == "point_10m":
        samples = reduction_image(year).unmask(NODATA_SENTINEL).sampleRegions(
            collection=point_collection(points),
            properties=["point_id"],
            scale=10,
            projection="EPSG:3857",
            geometries=False,
            tileScale=8,
        )
    else:
        samples = reduction_image(year).reduceRegions(
            collection=footprint_collection(points, footprint_m),
            reducer=ee.Reducer.mean(),
            crs="EPSG:3857",
            scale=30,
            tileScale=8,
        )
    url = samples.getDownloadURL(
        filetype="CSV",
        selectors=["point_id", *REDUCTION_BANDS],
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
            print(
                f"    Download interrupted; retrying in {delay}s "
                f"({attempt + 1}/4)",
                flush=True,
            )
            time.sleep(delay)
    assert response is not None
    csv_path = output.with_suffix(".csv")
    csv_path.write_bytes(response.content)

    table = pd.read_csv(csv_path).replace(NODATA_SENTINEL, np.nan)
    if table["point_id"].duplicated().any():
        raise RuntimeError(f"Earth Engine returned duplicate point IDs for {output}")
    table["aspect"] = (
        np.degrees(np.arctan2(table.pop("aspect_sin"), table.pop("aspect_cos")))
        % 360
    )
    table.insert(1, "year", year)
    table = points.merge(table, on="point_id", how="left", validate="one_to_one")
    table = table[[*points.columns, "year", *VARIABLES]]
    table.to_parquet(output, index=False)
    return table


def download_year(
    year: int,
    points: pd.DataFrame,
    output: Path,
    footprint_m: int,
    chunk_size: int,
    sampling_mode: str,
    region_slug: str,
) -> dict:
    chunk_dir = output.parent / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    chunk_outputs = []
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        chunk_output = chunk_dir / f"{region_slug}_environment_{year}_{start:05d}_{stop:05d}.parquet"
        chunk_outputs.append(str(chunk_output))
        if chunk_output.exists():
            chunk = pd.read_parquet(chunk_output)
            expected_ids = points.iloc[start:stop]["point_id"].tolist()
            if chunk["point_id"].tolist() != expected_ids:
                raise RuntimeError(f"Checkpoint point IDs do not match: {chunk_output}")
            print(f"  Reusing {chunk_output.name}")
        else:
            print(f"  Downloading points {start:,}:{stop:,}")
            chunk = download_chunk(
                year,
                points.iloc[start:stop].copy(),
                chunk_output,
                footprint_m,
                sampling_mode,
            )
        chunks.append(chunk)

    table = pd.concat(chunks, ignore_index=True)
    if len(table) != len(points) or table["point_id"].duplicated().any():
        raise RuntimeError(f"Incomplete or duplicate data after merging {year} chunks")
    table.to_parquet(output, index=False)
    missing = table[VARIABLES].isna().mean().sort_values(ascending=False)
    return {
        "year": year,
        "rows": int(len(table)),
        "footprint_m": footprint_m if sampling_mode == "footprint_mean" else None,
        "footprint_area_km2": (
            (footprint_m ** 2) / 1_000_000
            if sampling_mode == "footprint_mean"
            else None
        ),
        "sampling_mode": sampling_mode,
        "chunk_size": chunk_size,
        "chunks": chunk_outputs,
        "download_url_host": "earthengine.googleapis.com",
        "missing_fraction": {key: float(value) for key, value in missing.items()},
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    parser.add_argument("--authenticate", action="store_true")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--years", default="2017:2023")
    parser.add_argument("--footprint-m", type=int, default=640)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--region-slug", default="gba")
    parser.add_argument(
        "--sampling-mode",
        choices=["footprint_mean", "point_10m"],
        default="footprint_mean",
    )
    args = parser.parse_args()

    if args.authenticate:
        ee.Authenticate(auth_mode="localhost")
    if not args.project:
        raise SystemExit("Pass --project YOUR_GCP_PROJECT or set EE_PROJECT")
    ee.Initialize(project=args.project)

    start, end = (int(part) for part in args.years.split(":", 1))
    years = list(range(start, end + 1))
    embedding_table = pd.read_parquet(
        args.embeddings, columns=["point_id", "lon", "lat", "city"]
    ).drop_duplicates("point_id").sort_values("point_id")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for year in years:
        output = args.output_dir / f"{args.region_slug}_environment_{year}.parquet"
        print(f"Downloading {year} environmental variables...", flush=True)
        summary = download_year(
            year,
            embedding_table,
            output,
            footprint_m=args.footprint_m,
            chunk_size=args.chunk_size,
            sampling_mode=args.sampling_mode,
            region_slug=args.region_slug,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    combined = pd.concat(
        [pd.read_parquet(args.output_dir / f"{args.region_slug}_environment_{year}.parquet") for year in years],
        ignore_index=True,
    )
    combined_output = (
        args.output_dir.parent
        / f"{args.output_dir.name}_{start}_{end}.parquet"
    )
    combined.to_parquet(combined_output, index=False)

    provenance = {
        "project": args.project,
        "region_slug": args.region_slug,
        "variables": VARIABLES,
        "data_sources": DATA_SOURCES,
        "substitutions": SUBSTITUTIONS,
        "spatial_support": {
            "sampling_mode": args.sampling_mode,
            "embedding_overview_m": (
                args.footprint_m if args.sampling_mode == "footprint_mean" else None
            ),
            "environmental_summary": (
                "mean over a square neighbourhood"
                if args.sampling_mode == "footprint_mean"
                else "point value after bilinear reprojection to a 10 m grid"
            ),
            "area_km2": (
                (args.footprint_m ** 2) / 1_000_000
                if args.sampling_mode == "footprint_mean"
                else None
            ),
        },
        "year_summaries": summaries,
        "combined_output": str(combined_output),
    }
    provenance_path = combined_output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Saved combined environment table to {combined_output}")


if __name__ == "__main__":
    main()
