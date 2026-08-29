"""Build the real-data GBA FAISS index and generate demonstration RAG answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.gba_lsi import GBALandSurfaceIntelligence, build_lsi_assets


ROOT = Path(__file__).resolve().parent.parent


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
        "--dictionary",
        type=Path,
        default=ROOT / "results/gba_real/dimension_dictionary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/gba_real/lsi",
    )
    parser.add_argument("--nlist", type=int)
    parser.add_argument("--nprobe", type=int, default=64)
    args = parser.parse_args()

    manifest = build_lsi_assets(
        args.embeddings,
        args.environment,
        args.dictionary,
        args.output_dir,
        nlist=args.nlist,
        nprobe=args.nprobe,
    )
    engine = GBALandSurfaceIntelligence(args.output_dir / "manifest.json")
    demos = {
        "guangzhou_flood": engine.answer(
            "广州这个位置的洪涝环境敏感性如何？",
            lat=23.1291,
            lon=113.2644,
            year=2023,
        ),
        "shenzhen_urban": engine.answer(
            "深圳这个位置的城市化和热环境有什么特点？",
            lat=22.5431,
            lon=114.0579,
            year=2023,
        ),
        "zhaoqing_vegetation": engine.answer(
            "肇庆这个位置的植被健康状况如何？",
            lat=23.0472,
            lon=112.4651,
            year=2023,
        ),
    }
    demo_path = args.output_dir / "demo_answers.json"
    demo_path.write_text(json.dumps(demos, indent=2), encoding="utf-8")
    (args.output_dir / "demo_answers.md").write_text(
        "\n\n---\n\n".join(item["answer"] for item in demos.values()),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "manifest.json"),
                "demo_answers": str(demo_path),
                "n_vectors": manifest["n_vectors"],
                "nlist": manifest["nlist"],
                "nprobe": manifest["nprobe"],
                "build_seconds": manifest["build_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
