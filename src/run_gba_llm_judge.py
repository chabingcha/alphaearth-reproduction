"""Optional four-model rotating LLM-as-Judge runner for GBA LSI answers.

This runner is not used by the credential-free reproduction.  It accepts four
OpenAI-compatible endpoint profiles, rotates them across the 360 saved answer
cycles, checkpoints every judgment, and produces the same five 1--5 criteria
used in the paper.  API keys are read only from named environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import requests


ROOT = Path(__file__).resolve().parent.parent
CRITERIA = [
    "grounding",
    "coherence",
    "scientific_accuracy",
    "completeness",
    "practical_utility",
]
SYSTEM_PROMPT = """You are evaluating a geospatial land-surface answer.
Score each criterion from 1 (poor) to 5 (excellent):
grounding, coherence, scientific_accuracy, completeness, practical_utility.
Use only the supplied structured evidence to verify claims. Do not reward
unsupported precision. Return JSON only, with the five numeric keys and a
brief `rationale` string."""


def load_profiles(path: Path) -> list[dict]:
    profiles = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profiles, list) or len(profiles) != 4:
        raise ValueError("Judge profile file must contain exactly four profiles")
    required = {"name", "base_url", "model", "api_key_env"}
    for profile in profiles:
        missing = required - set(profile)
        if missing:
            raise ValueError(f"Profile {profile!r} is missing {sorted(missing)}")
        if not os.environ.get(profile["api_key_env"]):
            raise RuntimeError(
                f"Missing API key environment variable: {profile['api_key_env']}"
            )
    return profiles


def judge_payload(record: dict) -> str:
    response = record["response"]
    compact = {
        "question": response["question"],
        "intent": response["intent"],
        "answer": response["answer"],
        "structured_evidence": response["evidence"],
        "dimension_evidence": response["active_dimensions"],
        "limitations": response["limitations"],
    }
    return json.dumps(compact, ensure_ascii=False)


def parse_json(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    for criterion in CRITERIA:
        value[criterion] = float(np.clip(float(value[criterion]), 1, 5))
    return value


def call_judge(profile: dict, user_content: str, timeout: int) -> dict:
    url = profile["base_url"].rstrip("/") + "/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {os.environ[profile['api_key_env']]}",
            "Content-Type": "application/json",
        },
        json={
            "model": profile["model"],
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    return parse_json(body["choices"][0]["message"]["content"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answers",
        type=Path,
        default=ROOT / "results/gba_real/lsi/evaluation/answers.jsonl",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        required=True,
        help="JSON list of exactly four OpenAI-compatible judge profiles",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/gba_real/lsi/evaluation/llm_judgments.jsonl",
    )
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    profiles = load_profiles(args.profiles)
    records = [json.loads(line) for line in args.answers.read_text(encoding="utf-8").splitlines() if line]
    completed = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line:
                item = json.loads(line)
                completed[int(item["cycle"])] = item
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for record in records:
            cycle = int(record["cycle"])
            if cycle in completed:
                continue
            profile = profiles[(cycle - 1) % len(profiles)]
            error = None
            for attempt in range(args.retries):
                try:
                    started = time.perf_counter()
                    scores = call_judge(profile, judge_payload(record), args.timeout)
                    item = {
                        "cycle": cycle,
                        "judge": profile["name"],
                        "model": profile["model"],
                        "scores": scores,
                        "latency_seconds": time.perf_counter() - started,
                    }
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed[cycle] = item
                    break
                except Exception as exc:  # network/provider errors need retry
                    error = exc
                    time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Cycle {cycle} failed after retries: {error}")
            if cycle % 20 == 0:
                print(f"judged {cycle}/{len(records)}", flush=True)

    all_items = [completed[cycle] for cycle in sorted(completed)]
    summary_path = args.summary or args.output.with_name("llm_judge_summary.json")
    summary = {
        "evaluation_type": "four-model rotating LLM-as-Judge",
        "cycles": len(all_items),
        "judges": sorted({item["judge"] for item in all_items}),
        "scores": {
            criterion: {
                "mean": float(np.mean([item["scores"][criterion] for item in all_items])),
                "std": float(np.std([item["scores"][criterion] for item in all_items])),
            }
            for criterion in CRITERIA
        },
    }
    summary["overall_mean"] = float(
        np.mean([value["mean"] for value in summary["scores"].values()])
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
