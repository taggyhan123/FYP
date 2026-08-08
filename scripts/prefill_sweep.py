#!/usr/bin/env python3
"""Find the tool-catalog size at which prefix caching starts to pay for itself.

Task B established that at a 303-token prompt, 95% prefix reuse produced no
measurable TTFT benefit: prefill is dominated by fixed per-request overhead at
that size. Every ordering experiment therefore needs to run above some catalog
size, and this script measures where that is.

For each menu size the same request is issued cold (prefix cache reset) and then
warm (identical repeat). The cold/warm gap is the prefix-cache benefit. The
crossover is the smallest menu whose cold and warm 95% intervals no longer
overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.analysis import load_processed
from tatm.prompting import build_menu, openai_tool
from tatm.stats import describe, separated
from tatm.vllm_client import (
    fetch_text,
    metric_delta,
    parse_prometheus,
    request_json,
    require_prefix_cache_reset,
    served_model,
    server_cache_config,
)

DEFAULT_SIZES = (1, 4, 16, 64, 128, 200)
QUERY = "What is the weather in Singapore? Use a tool."


def request_payload(
    model: str,
    tools: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": QUERY}],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
    }


def timed_request(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    before = parse_prometheus(fetch_text(f"{base_url}/metrics"))
    started = time.perf_counter()
    response = request_json("POST", f"{base_url}/v1/chat/completions", payload)
    wall = time.perf_counter() - started
    after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
    delta = metric_delta(before, after)
    return {
        "wall_seconds": wall,
        "ttft": delta.get("vllm:time_to_first_token_seconds_sum", 0.0),
        "prefill": delta.get("vllm:request_prefill_time_seconds_sum", 0.0),
        "cached_tokens": delta.get("vllm:prompt_tokens_cached", 0.0),
        "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Menu sizes in tools.",
    )
    parser.add_argument(
        "--max-schema-tokens",
        type=int,
        default=300,
        help="Exclude outlier schemas so menu size tracks token count.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model = served_model(base_url, args.model)
    cache_config = server_cache_config(base_url)

    tools, _tasks = load_processed(args.processed_dir)
    pool = sorted(
        tool_id
        for tool_id, tool in tools.items()
        if 0 < tool.schema_tokens <= args.max_schema_tokens
    )
    if not pool:
        raise SystemExit("No tools within the schema-token limit.")
    gold = (pool[0],)

    # Burn in one request: the server's first-ever request is materially slower
    # and would otherwise land in the smallest menu size and skew it.
    warmup_menu = build_menu(gold, pool, min(args.sizes), seed=7)
    timed_request(
        base_url,
        request_payload(
            model,
            [openai_tool(tools[t]) for t in warmup_menu],
            args.max_tokens,
        ),
    )

    rows: list[dict[str, Any]] = []
    for size in sorted(args.sizes):
        menu = build_menu(gold, pool, size, seed=7)
        payload = request_payload(
            model,
            [openai_tool(tools[t]) for t in menu],
            args.max_tokens,
        )
        canonical_tokens = sum(tools[t].schema_tokens for t in menu)
        cold: list[float] = []
        warm: list[float] = []
        cold_cached: list[float] = []
        warm_cached: list[float] = []
        prompt_tokens = 0
        for _ in range(args.repeats):
            try:
                require_prefix_cache_reset(base_url)
            except RuntimeError as error:
                raise SystemExit(str(error)) from error
            first = timed_request(base_url, payload)
            second = timed_request(base_url, payload)
            cold.append(first["ttft"])
            warm.append(second["ttft"])
            cold_cached.append(first["cached_tokens"])
            warm_cached.append(second["cached_tokens"])
            prompt_tokens = second["prompt_tokens"]

        cold_stats = describe(cold)
        warm_stats = describe(warm)
        rows.append(
            {
                "menu_tools": len(menu),
                "canonical_tool_tokens": canonical_tokens,
                "prompt_tokens": prompt_tokens,
                "cold_ttft_seconds": cold_stats,
                "warm_ttft_seconds": warm_stats,
                "cold_cached_tokens": describe(cold_cached),
                "warm_cached_tokens": describe(warm_cached),
                "warm_faster": warm_stats["mean"] < cold_stats["mean"],
                "intervals_separated": separated(cold_stats, warm_stats),
            }
        )
        print(
            f"  {len(menu):>4} tools  {prompt_tokens:>6} prompt tok  "
            f'cold {cold_stats["mean"] * 1000:>7.1f}ms  '
            f'warm {warm_stats["mean"] * 1000:>7.1f}ms  '
            f'cached {warm_cached[-1]:>6.0f}  '
            f'{"SEPARATED" if rows[-1]["intervals_separated"] else ""}'
        )

    crossover = next(
        (
            row["menu_tools"]
            for row in rows
            if row["intervals_separated"] and row["warm_faster"]
        ),
        None,
    )
    output = {
        "format_version": 1,
        "run_label": args.run_label,
        "server": base_url,
        "model": model,
        "server_cache_config": cache_config,
        "repeats": args.repeats,
        "crossover_menu_tools": crossover,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {args.output}")
    print(f"server enable_prefix_caching={cache_config.get('enable_prefix_caching')}")
    if crossover is None:
        print(
            "No crossover: warm and cold TTFT never separated. Prefix caching "
            "buys nothing measurable at any tested menu size."
        )
    else:
        print(f"Crossover at {crossover} tools.")


if __name__ == "__main__":
    main()
