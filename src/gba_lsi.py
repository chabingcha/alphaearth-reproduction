"""FAISS retrieval and evidence-grounded GBA Land Surface Intelligence RAG."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.build_gba_dimension_dictionary import VARIABLE_METADATA
from src.fetch_gba_environment_gee import VARIABLES
from src.run_gba_real_analysis import EMBEDDING_COLUMNS, load_aligned


INTENT_SPECS = {
    "flood_risk": {
        "label": "洪涝风险",
        "variables": ["annual_precip", "max_monthly_precip", "annual_runoff", "flow_accumulation", "impervious_surface", "slope"],
        "keywords": ["洪", "涝", "积水", "flood"],
        "advice": "该结果只表示环境敏感性，工程排水能力、潮位和实时降雨仍需另行核实。",
    },
    "drought_vulnerability": {
        "label": "干旱脆弱性",
        "variables": ["annual_precip", "soil_moisture", "annual_et", "ndvi_mean", "water_capacity"],
        "keywords": ["干旱", "缺水", "drought"],
        "advice": "可将偏低的土壤湿度与降水作为巡查线索，不应代替现场墒情或供水数据。",
    },
    "vegetation_health": {
        "label": "植被健康",
        "variables": ["ndvi_mean", "ndvi_max", "evi_mean", "lai_mean", "tree_cover"],
        "keywords": ["植被", "绿地", "生态", "ndvi", "vegetation"],
        "advice": "年度指标不能识别短期病虫害，异常区域应结合季节序列和现场调查。",
    },
    "agricultural_suitability": {
        "label": "农业适宜性",
        "variables": ["slope", "clay_fraction", "organic_carbon", "soil_ph", "water_capacity", "annual_precip", "air_temp_mean"],
        "keywords": ["农业", "种植", "耕地", "作物", "agricultur"],
        "advice": "作物适宜性还取决于具体品种、灌溉、盐分和管理条件，本结果仅用于区域初筛。",
    },
    "climate_characterization": {
        "label": "气候特征",
        "variables": ["air_temp_mean", "dew_point_temp", "annual_precip", "max_monthly_precip", "lst_daytime", "lst_nighttime"],
        "keywords": ["气候", "气温", "降水", "温度", "climate"],
        "advice": "这里使用年度汇总值，不能代替逐日极端天气或未来气候情景。",
    },
    "terrain_analysis": {
        "label": "地形分析",
        "variables": ["elevation", "slope", "aspect", "flow_accumulation"],
        "keywords": ["地形", "高程", "坡度", "坡向", "terrain"],
        "advice": "1公里嵌入支持尺度会平滑局部陡坎，工程设计应使用更高分辨率DEM。",
    },
    "hydrology": {
        "label": "水文条件",
        "variables": ["soil_moisture", "annual_runoff", "annual_et", "flow_accumulation", "annual_precip"],
        "keywords": ["水文", "径流", "蒸散", "土壤湿度", "hydrolog"],
        "advice": "ERA5-Land水文量空间分辨率较粗，适合区域背景判断而非街区级定量设计。",
    },
    "urban_development": {
        "label": "城市发展",
        "variables": ["impervious_surface", "nighttime_lights", "population_density", "lst_daytime", "tree_cover"],
        "keywords": ["城市", "建成", "人口", "灯光", "热岛", "urban"],
        "advice": "人口和灯光的空间外推表现较弱，应将其视作辅助证据而非精确社会统计。",
    },
    "location_comparison": {
        "label": "相似地点比较",
        "variables": ["elevation", "ndvi_mean", "air_temp_mean", "annual_precip", "impervious_surface", "nighttime_lights"],
        "keywords": ["比较", "类似", "相似", "analog", "compar"],
        "advice": "嵌入相似表示综合地表外观相近，不等于行政、经济或灾害机制完全相同。",
    },
    "general_profile": {
        "label": "综合地表画像",
        "variables": ["elevation", "ndvi_mean", "tree_cover", "air_temp_mean", "annual_precip", "impervious_surface", "nighttime_lights"],
        "keywords": ["综合", "概况", "画像", "环境", "general"],
        "advice": "结论来自年度遥感嵌入与区域数据，适合筛查和比较，不替代现场观测。",
    },
}


def classify_intent(question: str) -> str:
    lowered = question.lower()
    for intent, spec in INTENT_SPECS.items():
        if any(keyword in lowered for keyword in spec["keywords"]):
            return intent
    return "general_profile"


def haversine_km(lat1: float, lon1: float, lat2, lon2):
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32).copy()
    faiss.normalize_L2(vectors)
    return vectors


def build_lsi_assets(
    embeddings_path: Path,
    environment_path: Path,
    dictionary_path: Path,
    output_dir: Path,
    nlist: int | None = None,
    nprobe: int = 64,
) -> dict:
    frame = load_aligned(embeddings_path, environment_path).reset_index(drop=True)
    vectors = normalize_vectors(frame[EMBEDDING_COLUMNS].to_numpy(dtype=np.float32))
    nlist = nlist or int(np.sqrt(len(vectors)))
    nlist = max(1, min(nlist, len(vectors)))
    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIVFFlat(
        quantizer, vectors.shape[1], nlist, faiss.METRIC_INNER_PRODUCT
    )
    started = time.time()
    index.train(vectors)
    index.add(vectors)
    index.nprobe = min(nprobe, nlist)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "gba_aef_ivfflat.faiss"
    metadata_path = output_dir / "gba_lsi_metadata.parquet"
    manifest_path = output_dir / "manifest.json"
    faiss.write_index(index, str(index_path))
    frame[["point_id", "lon", "lat", "city", "year", *VARIABLES]].to_parquet(
        metadata_path, index=False
    )
    embedding_mean = frame[EMBEDDING_COLUMNS].mean().to_numpy(dtype=float)
    embedding_std = frame[EMBEDDING_COLUMNS].std().to_numpy(dtype=float)
    dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
    manifest = {
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "embeddings_path": str(embeddings_path),
        "environment_path": str(environment_path),
        "dictionary_path": str(dictionary_path),
        "index_type": "IndexIVFFlat",
        "metric": "cosine similarity (inner product over L2-normalized vectors)",
        "n_vectors": int(index.ntotal),
        "dimension": int(vectors.shape[1]),
        "nlist": int(nlist),
        "nprobe": int(index.nprobe),
        "build_seconds": time.time() - started,
        "years": sorted(int(value) for value in frame["year"].unique()),
        "embedding_mean": embedding_mean.tolist(),
        "embedding_std": embedding_std.tolist(),
        "dictionary_summary": dictionary["summary"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class GBALandSurfaceIntelligence:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.index = faiss.read_index(self.manifest["index_path"])
        self.index.nprobe = self.manifest["nprobe"]
        self.metadata = pd.read_parquet(self.manifest["metadata_path"])
        embedding_frame = pd.read_parquet(
            self.manifest["embeddings_path"],
            columns=["point_id", "year", *EMBEDDING_COLUMNS],
        )
        ordered = self.metadata[["point_id", "year"]].merge(
            embedding_frame, on=["point_id", "year"], how="left", validate="one_to_one"
        )
        self.vectors_raw = ordered[EMBEDDING_COLUMNS].to_numpy(dtype=np.float32)
        self.vectors = normalize_vectors(self.vectors_raw)
        self.dictionary = json.loads(
            Path(self.manifest["dictionary_path"]).read_text(encoding="utf-8")
        )
        self.embedding_mean = np.asarray(self.manifest["embedding_mean"], dtype=float)
        self.embedding_std = np.asarray(self.manifest["embedding_std"], dtype=float)
        self.embedding_std[self.embedding_std < 1e-8] = 1.0
        self._year_values = {
            int(year): group.index.to_numpy()
            for year, group in self.metadata.groupby("year")
        }

    def nearest_record(self, lat: float, lon: float, year: int) -> int:
        indices = self._year_values.get(int(year))
        if indices is None:
            raise ValueError(f"Year {year} is not available")
        rows = self.metadata.loc[indices]
        distances = haversine_km(lat, lon, rows["lat"].to_numpy(), rows["lon"].to_numpy())
        return int(indices[int(np.argmin(distances))])

    def retrieve(
        self,
        lat: float,
        lon: float,
        year: int = 2023,
        k: int = 10,
        exclude_same_point: bool = True,
        min_distance_km: float = 1.0,
    ) -> dict:
        query_index = self.nearest_record(lat, lon, year)
        query_row = self.metadata.iloc[query_index]
        search_k = min(len(self.metadata), max(k * 20, 200))
        similarities, indices = self.index.search(
            self.vectors[query_index : query_index + 1], search_k
        )
        matches = []
        for similarity, index in zip(similarities[0], indices[0]):
            if index < 0:
                continue
            row = self.metadata.iloc[int(index)]
            distance = float(haversine_km(lat, lon, row["lat"], row["lon"]))
            if exclude_same_point and int(row["point_id"]) == int(query_row["point_id"]):
                continue
            if distance < min_distance_km:
                continue
            matches.append(
                {
                    "row_index": int(index),
                    "point_id": int(row["point_id"]),
                    "city": str(row["city"]),
                    "year": int(row["year"]),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "similarity": float(similarity),
                    "distance_km": distance,
                    "environment": {
                        variable: (
                            float(row[variable]) if np.isfinite(row[variable]) else None
                        )
                        for variable in VARIABLES
                    },
                }
            )
            if len(matches) >= k:
                break
        return {
            "requested_location": {"lat": lat, "lon": lon, "year": int(year)},
            "query_index": query_index,
            "nearest_grid_point": {
                "point_id": int(query_row["point_id"]),
                "city": str(query_row["city"]),
                "year": int(query_row["year"]),
                "lat": float(query_row["lat"]),
                "lon": float(query_row["lon"]),
                "distance_km": float(
                    haversine_km(lat, lon, query_row["lat"], query_row["lon"])
                ),
                "environment": {
                    variable: (
                        float(query_row[variable])
                        if np.isfinite(query_row[variable])
                        else None
                    )
                    for variable in VARIABLES
                },
            },
            "similar_locations": matches,
        }

    def percentile(self, variable: str, value: float, year: int) -> float:
        indices = self._year_values[int(year)]
        values = self.metadata.loc[indices, variable].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not np.isfinite(value) or len(values) == 0:
            return float("nan")
        return float(100.0 * np.mean(values <= value))

    def active_dimensions(self, query_index: int, variables: list[str], k: int = 3) -> list[dict]:
        z = (self.vectors_raw[query_index] - self.embedding_mean) / self.embedding_std
        candidates = []
        variable_set = set(variables)
        for dim, name in enumerate(EMBEDDING_COLUMNS):
            info = self.dictionary["dimensions"][name]
            if info["primary_variable"] in variable_set:
                candidates.append(
                    {
                        "dimension": name,
                        "z_score": float(z[dim]),
                        "primary_variable": info["primary_variable"],
                        "spearman_rho": info["spearman_rho"],
                        "agreement_count": info["agreement_count"],
                        "robust": info["robust"],
                    }
                )
        candidates.sort(key=lambda item: abs(item["z_score"]), reverse=True)
        return candidates[:k]

    def answer(
        self,
        question: str,
        lat: float,
        lon: float,
        year: int = 2023,
        intent: str | None = None,
        k: int = 5,
    ) -> dict:
        intent = intent or classify_intent(question)
        if intent not in INTENT_SPECS:
            raise ValueError(f"Unknown intent: {intent}")
        spec = INTENT_SPECS[intent]
        retrieval = self.retrieve(lat, lon, year=year, k=k)
        profile = retrieval["nearest_grid_point"]["environment"]
        evidence = []
        lines = []
        for variable in spec["variables"]:
            value = profile[variable]
            if value is None:
                continue
            percentile = self.percentile(variable, value, year)
            display, unit = VARIABLE_METADATA[variable]
            level = "偏高" if percentile >= 67 else "偏低" if percentile <= 33 else "中等"
            decimals = 3 if abs(value) < 10 else 1
            lines.append(
                f"- {display}：{value:.{decimals}f} {unit}，位于当年大湾区第{percentile:.0f}百分位（{level}）。"
            )
            evidence.append(
                {
                    "variable": variable,
                    "display_name_zh": display,
                    "value": value,
                    "unit": unit,
                    "percentile": percentile,
                    "level": level,
                    "source": "nearest_grid_point_environment",
                }
            )
        analogs = retrieval["similar_locations"][:3]
        analog_text = "；".join(
            f"{item['city']}({item['lat']:.3f},{item['lon']:.3f})，相似度{item['similarity']:.3f}"
            for item in analogs
        )
        active = self.active_dimensions(
            retrieval["query_index"], spec["variables"], k=3
        )
        dimension_text = "；".join(
            f"{item['dimension']}→{VARIABLE_METADATA[item['primary_variable']][0]}"
            f"(z={item['z_score']:+.2f}, {item['agreement_count']}种方法一致)"
            for item in active
        ) or "没有满足当前主题的高置信维度"
        grid = retrieval["nearest_grid_point"]
        answer_text = (
            f"## {spec['label']}\n\n"
            f"数据定位：{year}年，{grid['city']}网格点 "
            f"({grid['lat']:.4f}°N, {grid['lon']:.4f}°E)，距输入坐标"
            f"{grid['distance_km']:.2f} km。\n\n"
            "关键证据：\n" + "\n".join(lines) + "\n\n"
            f"AlphaEarth维度解释：{dimension_text}。\n\n"
            f"相似地点：{analog_text}。这些地点仅表示嵌入空间中的地表相似性。\n\n"
            f"使用提示：{spec['advice']}"
        )
        return {
            "question": question,
            "intent": intent,
            "intent_label": spec["label"],
            "answer": answer_text,
            "evidence": evidence,
            "active_dimensions": active,
            "retrieval": retrieval,
            "limitations": [spec["advice"]],
            "generator": "deterministic evidence-grounded RAG renderer",
        }


def parse_location(text: str) -> tuple[float, float] | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None
