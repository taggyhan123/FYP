#!/usr/bin/env python3
"""Black-box exact-prefix sanity probe for an OpenAI-compatible vLLM server.

Covers Task B checks 1-5 of the research brief. Each trial resets the prefix
cache, then walks a fixed scenario sequence so that reuse is attributable to a
known prompt edit rather than to residual cache state.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.vllm_client import (
    KvUsageSampler,
    fetch_text,
    metric_delta,
    parse_prometheus,
    request_json,
    reset_prefix_cache,
    response_projection,
    served_model,
    server_cache_config,
)

# Two-sided 95% Student-t critical values by degrees of freedom.
T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042,
}


def functions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_temperature",
                "description": "Convert a temperature between Celsius and Fahrenheit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "target_unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["value", "target_unit"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "save_note",
                "description": "Save a short text note.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
    ]


def scenarios() -> list[tuple[str, list[dict[str, Any]]]]:
    original = functions()
    changed = json.loads(json.dumps(original))
    changed[1]["function"]["description"] += " Return an exact numeric result."
    reordered = [original[1], original[0], original[2]]
    return [
        ("original_cold", original),
        ("original_identical", original),
        ("changed_second_tool", changed),
        ("reordered_first_two", reordered),
        ("original_restored", original),
    ]


def payload(model: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Singapore? Use a tool.",
            }
        ],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "seed": 0,
        "max_tokens": 48,
    }


def run_scenario(
    base_url: str,
    model: str,
    scenario: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    before = parse_prometheus(fetch_text(f"{base_url}/metrics"))
    with KvUsageSampler(base_url) as sampler:
        start = time.perf_counter()
        response = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            payload(model, tools),
        )
        wall_seconds = time.perf_counter() - start
    after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
    return {
        "scenario": scenario,
        "wall_seconds": round(wall_seconds, 6),
        **response_projection(response),
        "metric_delta": metric_delta(before, after),
        "peak_gauges": {k: round(v, 6) for k, v in sorted(sampler.peak.items())},
        "gauge_samples": sampler.samples,
    }


def describe(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = statistics.fmean(values)
    row: dict[str, Any] = {
        "n": n,
        "mean": round(mean, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }
    if n > 1:
        stdev = statistics.stdev(values)
        critical = T_95.get(n - 1, 1.96)
        row["stdev"] = round(stdev, 6)
        row["ci95_half_width"] = round(critical * stdev / (n**0.5), 6)
    return row


def summarize(trials: list[list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for index, scenario in enumerate(name for name, _ in scenarios()):
        rows = [trial[index] for trial in trials]
        cached = [
            row["metric_delta"].get("vllm:prompt_tokens_cached", 0.0) for row in rows
        ]
        prompt_tokens = [row["usage"].get("prompt_tokens", 0) for row in rows]
        ttft = [
            row["metric_delta"].get("vllm:time_to_first_token_seconds_sum", 0.0)
            for row in rows
        ]
        prefill = [
            row["metric_delta"].get("vllm:request_prefill_time_seconds_sum", 0.0)
            for row in rows
        ]
        itl_sum = [
            row["metric_delta"].get("vllm:inter_token_latency_seconds_sum", 0.0)
            for row in rows
        ]
        itl_count = [
            row["metric_delta"].get("vllm:inter_token_latency_seconds_count", 0.0)
            for row in rows
        ]
        peak_kv = [
            row["peak_gauges"].get("vllm:kv_cache_usage_perc", 0.0) for row in rows
        ]
        mean_itl = [
            (total / count) if count else 0.0
            for total, count in zip(itl_sum, itl_count)
        ]
        distinct_prompt_tokens = sorted(set(prompt_tokens))
        summary[scenario] = {
            "prompt_tokens": distinct_prompt_tokens,
            "cached_prompt_tokens": describe(cached),
            "reuse_fraction": round(
                statistics.fmean(cached) / statistics.fmean(prompt_tokens), 4
            )
            if statistics.fmean(prompt_tokens)
            else 0.0,
            "ttft_seconds": describe(ttft),
            "prefill_seconds": describe(prefill),
            "mean_inter_token_latency_seconds": describe(mean_itl),
            "peak_kv_cache_usage_perc": describe(peak_kv),
            "outputs_identical_across_trials": len(
                {row["content"] for row in rows}
            )
            == 1,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", help="Defaults to the first served model.")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pause-seconds", type=float, default=0.25)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model = served_model(base_url, args.model)
    cache_config = server_cache_config(base_url)

    trials: list[list[dict[str, Any]]] = []
    resets: list[bool] = []
    for _ in range(args.repeats):
        resets.append(reset_prefix_cache(base_url))
        rows = []
        for scenario, tools in scenarios():
            rows.append(run_scenario(base_url, model, scenario, tools))
            time.sleep(args.pause_seconds)
        trials.append(rows)

    reset_ok = all(resets)
    output = {
        "format_version": 2,
        "run_label": args.run_label,
        "server": base_url,
        "model": model,
        "server_cache_config": cache_config,
        "repeats": args.repeats,
        "prefix_cache_reset_between_trials": reset_ok,
        "summary": summarize(trials),
        "trials": [
            {"trial": index, "results": rows} for index, rows in enumerate(trials)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    enabled = cache_config.get("enable_prefix_caching")
    print(f"Wrote {args.output}")
    print(f"server enable_prefix_caching={enabled}")
    if not reset_ok:
        print(
            "WARNING: /reset_prefix_cache unavailable "
            "(start the server with VLLM_SERVER_DEV_MODE=1); "
            "'original_cold' is only cold in the first trial.",
            file=sys.stderr,
        )
    for scenario, row in output["summary"].items():
        print(
            f"  {scenario:<22} cached={row['cached_prompt_tokens']['mean']:>7.1f} "
            f"reuse={row['reuse_fraction']:>6.1%} "
            f"ttft={row['ttft_seconds']['mean']:.4f}s"
        )


if __name__ == "__main__":
    main()
