# AlphaEarth Embedding Interpretability — GBA and Guangdong Reproduction

Real-data reproduction and regional extension of:

> Rahman, M. (2026). *Physically Interpretable AlphaEarth Foundation Model Embeddings Enable LLM-Based Land Surface Intelligence.* arXiv:2602.10354v1.

The project first reproduces the experiment over the Guangdong–Hong Kong–Macao Greater Bay Area (GBA), then expands the same 2017–2023 design to Guangdong Province. The GBA dataset contains 54,614 annual observations at 7,802 locations; the Guangdong dataset contains 175,875 observations at 25,125 locations across 21 prefecture-level cities. Both align 64-dimensional AlphaEarth embeddings with 26 real environmental variables.

## Implemented modules

- 64×26 Spearman correlation analysis.
- Random Forest random CV, spatial block CV and permutation importance.
- Seven-year temporal stability and coordinate-only controls.
- Compact multi-task scalar-token Transformer with random/spatial CV.
- Three-method dimension dictionary (Spearman + RF + Transformer).
- FAISS IndexIVFFlat over all 54,614 annual embeddings.
- Evidence-grounded RAG-style environmental answers for ten intents.
- 360-cycle deterministic evidence evaluation and retrieval benchmark.
- Optional resumable four-model LLM-as-Judge runner for configured APIs.
- Quality audit with point-group CV, location learning curves, city holdouts,
  three random seeds, feature baselines and target-transform sensitivity.

The earlier synthetic-data reproduction remains available through `quick_run.py` and `main.py`. Real-data outputs are under `results/gba_real/` and `results/guangdong_real/`.

## Main results

| Metric | Original paper (CONUS) | GBA | Guangdong |
|---|---:|---:|---:|
| Rows / valid annual locations | ~12.1M / ~1.73M per year | 54,614 / 7,802 | 175,875 / 25,125 |
| RF mean random CV R² | — | 0.662 | 0.651 |
| RF variables R² > 0.9 | 12/26 | 1/26 | 1/26 |
| RF variables R² > 0.7 | 20/26 | 17/26 | 14/26 |
| RF top-10 mean spatial ΔR² | 0.009 | 0.291 | 0.067 |
| Transformer mean random CV R² | — | 0.414 | 0.448 |
| Transformer mean spatial ΔR² | 0.017 | 0.213 | 0.087 |
| Mean temporal stability | 0.963 | 0.869 | 0.866 |
| Spearman–RF convergence | 0.450 | 0.215 | 0.231 |

The compact Transformer is not a code-level reproduction of the paper's five-million-row model. Its lower performance is retained as a regional small-sample result rather than presented as an equivalent benchmark.

## Reproduce the real-data analyses

Create an environment and install dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The strict real-data Parquet files are expected at:

```text
data/processed/gba_aef_gee_1km_2017_2023.parquet
data/processed/gba_environment_point_10m_2017_2023.parquet
data/processed/guangdong_aef_gee_1km_2017_2023.parquet
data/processed/guangdong_environment_point_10m_2017_2023.parquet
```

Run the extended stages:

```bash
.venv/bin/python -m src.run_gba_transformer
.venv/bin/python -m src.build_gba_dimension_dictionary
.venv/bin/python -m src.build_gba_lsi
.venv/bin/python -m src.evaluate_gba_lsi
.venv/bin/python -m src.generate_gba_extended_report
.venv/bin/python -m src.run_gba_quality_audit
.venv/bin/python -m src.generate_gba_quality_report
```

The Transformer defaults to Apple MPS when available, then CUDA, then CPU.

Run the completed Guangdong core analyses with explicit paths:

```bash
.venv/bin/python -m src.run_gba_real_analysis \
  --embeddings data/processed/guangdong_aef_gee_1km_2017_2023.parquet \
  --environment data/processed/guangdong_environment_point_10m_2017_2023.parquet \
  --output results/guangdong_real/analysis_results_paper_method.json \
  --max-rf-samples 30000 --trees 50 --folds 5 --block-size-deg 0.5 \
  --embedding-scale-m 1000 --support-area-km2 0.7853981634

.venv/bin/python -m src.run_gba_transformer \
  --embeddings data/processed/guangdong_aef_gee_1km_2017_2023.parquet \
  --environment data/processed/guangdong_environment_point_10m_2017_2023.parquet \
  --output results/guangdong_real/transformer_results.json \
  --checkpoint results/guangdong_real/guangdong_transformer.pt \
  --max-samples 30000 --folds 5 --epochs 12 --patience 3 \
  --batch-size 1024 --importance-samples 3000
```

## Optional four-model LLM-as-Judge

The credential-free evaluation verifies every generated numeric claim against the source rows. It is not directly comparable to the paper's four-LLM judgment score.

To run actual rotating LLM judges, copy and edit:

```text
results/gba_real/lsi/evaluation/judge_profiles.example.json
```

Set the four API-key environment variables named in that file, then run:

```bash
.venv/bin/python -m src.run_gba_llm_judge \
  --profiles path/to/judge_profiles.json
```

The runner uses OpenAI-compatible `/chat/completions` endpoints, rotates the four profiles, and resumes from its JSONL checkpoint.

## Reports and artifacts

- `results/gba_real/REPORT.md` — strict Spearman/RF reproduction.
- `results/gba_real/EXTENDED_REPORT.md` — Transformer, dictionary, FAISS/RAG and evaluation.
- `results/gba_real/QUALITY_AUDIT_REPORT.md` — leakage, learning curves, baselines and robustness.
- `results/gba_real/transformer_results.json` — full Transformer fold results.
- `results/gba_real/dimension_dictionary.json` — three-method knowledge base.
- `results/gba_real/lsi/demo_answers.md` — real answer examples.
- `results/gba_real/lsi/evaluation/evaluation_summary.json` — 360-cycle evaluation.
- `results/guangdong_real/analysis_results_paper_method.json` — Guangdong Spearman/RF results.
- `results/guangdong_real/transformer_results.json` — Guangdong compact Transformer results.
- `results/guangdong_real/guangdong_sampling_coverage.png` — 21-city sampling map.

The Guangdong quality audit and 0.5°/1°/2° spatial-sensitivity run are not yet complete and are therefore not reported as finished artifacts.

## Important differences from the paper

- GBA has 54,614 rows and Guangdong has 175,875 rows; the paper uses about 12.1 million CONUS rows.
- The compact Transformer uses 30,000 rows, 2 layers and 4 heads; the paper reports about 5 million rows, 4 layers, 8 heads and 60 epochs.
- GBA spatial validation uses 0.5° blocks because 2° blocks produce only five severely unbalanced groups.
- ERA5-Land replaces US-only PRISM variables, and ESA WorldCover replaces US-only NLCD impervious surface.
- No external LLM judgments or human-expert judgments are claimed unless their separate output files exist.
