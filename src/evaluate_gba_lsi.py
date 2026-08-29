"""Evaluate 360 real-data GBA LSI question/answer cycles.

No external LLM credential is assumed.  The executed evaluator therefore
checks evidence consistency, numerical accuracy, coverage, structure and
decision-use caveats deterministically.  The output is deliberately labelled
as an evidence-based automatic evaluation, not as LLM-as-Judge.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.gba_lsi import GBALandSurfaceIntelligence, INTENT_SPECS


ROOT = Path(__file__).resolve().parent.parent
QUESTION_TEMPLATES = {
    "flood_risk": "这个位置的洪涝环境敏感性如何？",
    "drought_vulnerability": "这个位置是否存在干旱或缺水脆弱性？",
    "vegetation_health": "这个位置的植被健康和覆盖状况如何？",
    "agricultural_suitability": "这个位置从环境条件看是否适合农业种植？",
    "climate_characterization": "请概括这个位置的气候和温度降水特征。",
    "terrain_analysis": "请分析这个位置的地形、高程和坡度条件。",
    "hydrology": "这个位置的水文、径流和土壤湿度条件如何？",
    "urban_development": "这个位置的城市化、人口和热环境有什么特点？",
    "location_comparison": "请把这个位置与AlphaEarth中相似地点进行比较。",
    "general_profile": "请给出这个位置的综合地表环境画像。",
}


def stratified_locations(metadata: pd.DataFrame, year: int, n: int, seed: int) -> pd.DataFrame:
    frame = metadata[metadata["year"] == year]
    cities = sorted(frame["city"].unique())
    base, remainder = divmod(n, len(cities))
    selected = []
    for city_index, city in enumerate(cities):
        count = base + (city_index < remainder)
        group = frame[frame["city"] == city]
        selected.append(
            group.sample(min(count, len(group)), random_state=seed + city_index)
        )
    result = pd.concat(selected).sort_values(["city", "point_id"])
    if len(result) != n:
        remaining = frame.drop(index=result.index)
        result = pd.concat(
            [result, remaining.sample(n - len(result), random_state=seed + 999)]
        )
    return result.reset_index(drop=True)


def score_response(engine: GBALandSurfaceIntelligence, response: dict) -> dict:
    retrieval = response["retrieval"]
    profile = retrieval["nearest_grid_point"]["environment"]
    year = retrieval["nearest_grid_point"]["year"]
    spec = INTENT_SPECS[response["intent"]]
    evidence = response["evidence"]

    exact = []
    percentile_exact = []
    for item in evidence:
        expected = profile[item["variable"]]
        exact.append(
            expected is not None
            and np.isclose(item["value"], expected, rtol=1e-7, atol=1e-7)
        )
        recalculated = engine.percentile(item["variable"], item["value"], year)
        percentile_exact.append(
            np.isclose(item["percentile"], recalculated, rtol=0, atol=1e-9)
        )
    accuracy_fraction = float(np.mean(exact + percentile_exact)) if evidence else 0.0
    scientific_accuracy = 1.0 + 4.0 * accuracy_fraction

    expected_available = sum(profile[variable] is not None for variable in spec["variables"])
    evidence_coverage = len(evidence) / max(1, expected_available)
    has_sources = all(item["source"] for item in evidence)
    has_retrieval = len(retrieval["similar_locations"]) >= 3
    grounding_components = [
        min(1.0, evidence_coverage),
        float(has_sources),
        float(has_retrieval),
        float(len(response["active_dimensions"]) > 0),
    ]
    grounding = 1.0 + 4.0 * float(np.mean(grounding_components))

    answer = response["answer"]
    required_markers = [
        f"## {response['intent_label']}",
        "数据定位：",
        "关键证据：",
        "AlphaEarth维度解释：",
        "相似地点：",
        "使用提示：",
    ]
    marker_fraction = np.mean([marker in answer for marker in required_markers])
    length_ok = 250 <= len(answer) <= 2200
    no_invalid_text = not any(token in answer.lower() for token in ["nan", "inf", "none"])
    coherence = 1.0 + 4.0 * float(
        np.mean([marker_fraction, float(length_ok), float(no_invalid_text)])
    )

    completeness_components = [
        min(1.0, evidence_coverage),
        float(len(response["active_dimensions"]) > 0),
        float(has_retrieval),
        float(bool(response["limitations"])),
    ]
    completeness = 1.0 + 4.0 * float(np.mean(completeness_components))

    utility_components = [
        float("使用提示：" in answer),
        float(bool(response["limitations"])),
        float("不" in spec["advice"] or "仅" in spec["advice"]),
        float(len(evidence) >= 3),
    ]
    practical_utility = 1.0 + 4.0 * float(np.mean(utility_components))

    scores = {
        "grounding": float(np.clip(grounding, 1, 5)),
        "coherence": float(np.clip(coherence, 1, 5)),
        "scientific_accuracy": float(np.clip(scientific_accuracy, 1, 5)),
        "completeness": float(np.clip(completeness, 1, 5)),
        "practical_utility": float(np.clip(practical_utility, 1, 5)),
    }
    scores["overall"] = float(np.mean(list(scores.values())))
    diagnostics = {
        "evidence_count": len(evidence),
        "expected_evidence_count": expected_available,
        "numeric_accuracy_fraction": accuracy_fraction,
        "retrieved_locations": len(retrieval["similar_locations"]),
        "active_dimension_count": len(response["active_dimensions"]),
        "answer_characters": len(answer),
    }
    return {"scores": scores, "diagnostics": diagnostics}


def retrieval_benchmark(
    engine: GBALandSurfaceIntelligence, locations: pd.DataFrame, year: int
) -> dict:
    recalls = []
    latencies = []
    analog_distances = []
    cross_city = []
    for _, location in locations.iterrows():
        query_index = engine.nearest_record(
            float(location["lat"]), float(location["lon"]), year
        )
        exact_scores = engine.vectors @ engine.vectors[query_index]
        exact = np.argsort(exact_scores)[-10:][::-1]
        started = time.perf_counter()
        _, approximate = engine.index.search(
            engine.vectors[query_index : query_index + 1], 10
        )
        latencies.append((time.perf_counter() - started) * 1000)
        recalls.append(len(set(exact) & set(approximate[0])) / 10)
        retrieval = engine.retrieve(
            float(location["lat"]), float(location["lon"]), year=year, k=5
        )
        source_city = retrieval["nearest_grid_point"]["city"]
        for analog in retrieval["similar_locations"]:
            analog_distances.append(analog["distance_km"])
            cross_city.append(analog["city"] != source_city)
    return {
        "exact_recall_at_10_mean": float(np.mean(recalls)),
        "exact_recall_at_10_min": float(np.min(recalls)),
        "raw_faiss_search_latency_ms_mean": float(np.mean(latencies)),
        "raw_faiss_search_latency_ms_p95": float(np.quantile(latencies, 0.95)),
        "filtered_analog_distance_km_mean": float(np.mean(analog_distances)),
        "filtered_analogs_cross_city_fraction": float(np.mean(cross_city)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/gba_real/lsi/manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/gba_real/lsi/evaluation",
    )
    parser.add_argument("--locations", type=int, default=36)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = GBALandSurfaceIntelligence(args.manifest)
    locations = stratified_locations(
        engine.metadata, args.year, args.locations, args.seed
    )
    records = []
    answer_records = []
    cycle = 0
    for location_index, location in locations.iterrows():
        for intent, question in QUESTION_TEMPLATES.items():
            cycle += 1
            started = time.perf_counter()
            response = engine.answer(
                question,
                lat=float(location["lat"]),
                lon=float(location["lon"]),
                year=args.year,
                intent=intent,
                k=5,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            evaluation = score_response(engine, response)
            record = {
                "cycle": cycle,
                "location_index": int(location_index),
                "point_id": int(location["point_id"]),
                "city": str(location["city"]),
                "year": args.year,
                "lat": float(location["lat"]),
                "lon": float(location["lon"]),
                "intent": intent,
                "question": question,
                "latency_ms": latency_ms,
                **evaluation["scores"],
                **evaluation["diagnostics"],
            }
            records.append(record)
            answer_records.append(
                {
                    "cycle": cycle,
                    "point_id": int(location["point_id"]),
                    "city": str(location["city"]),
                    "response": response,
                    "evaluation": evaluation,
                    "latency_ms": latency_ms,
                }
            )
            if cycle % 50 == 0:
                print(f"  evaluated {cycle} cycles", flush=True)

    detail = pd.DataFrame(records)
    retrieval_metrics = retrieval_benchmark(engine, locations, args.year)
    score_columns = [
        "grounding", "coherence", "scientific_accuracy", "completeness",
        "practical_utility", "overall",
    ]
    summary = {
        "evaluation_type": "deterministic evidence-based automatic evaluation",
        "not_directly_comparable_to": (
            "the paper's four-model rotating LLM-as-Judge evaluation"
        ),
        "cycles": int(len(detail)),
        "locations": int(detail["point_id"].nunique()),
        "intents": int(detail["intent"].nunique()),
        "year": args.year,
        "scores": {
            column: {
                "mean": float(detail[column].mean()),
                "std": float(detail[column].std(ddof=0)),
                "min": float(detail[column].min()),
                "max": float(detail[column].max()),
            }
            for column in score_columns
        },
        "latency_ms": {
            "mean": float(detail["latency_ms"].mean()),
            "p50": float(detail["latency_ms"].quantile(0.5)),
            "p95": float(detail["latency_ms"].quantile(0.95)),
        },
        "numeric_accuracy_fraction": float(
            detail["numeric_accuracy_fraction"].mean()
        ),
        "retrieval_benchmark": retrieval_metrics,
        "per_intent": {
            intent: {
                column: float(group[column].mean()) for column in score_columns
            }
            for intent, group in detail.groupby("intent")
        },
        "paper_reference": {
            "judge": "4 rotating LLMs",
            "cycles": 360,
            "grounding_mean": 3.93,
            "coherence_mean": 4.25,
            "scientific_accuracy_mean": 3.62,
            "completeness_mean": 3.48,
            "practical_utility_mean": 3.42,
            "overall_mean": 3.74,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "evaluation_cycles.csv", index=False)
    detail.to_parquet(args.output_dir / "evaluation_cycles.parquet", index=False)
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for record in answer_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
