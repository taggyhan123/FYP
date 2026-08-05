#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.io import read_jsonl
from tatm.measurement import (
    REPLAY_CONDITION_ROLES,
    project_request_measurement,
    selected_tool_text_condition,
    summarize_request_measurements,
)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a generated TATM workload against vLLM."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument(
        "--condition-role",
        choices=REPLAY_CONDITION_ROLES,
        default="ordering_candidate",
        help=(
            "Label this replay as an ordering candidate or as the predeclared "
            "ordinary selected-tool text-prefill fallback. The fallback "
            "requires an original-order input workload."
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument(
        "--tool-choice",
        choices=("auto", "required", "none"),
        default="auto",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Set chat_template_kwargs.enable_thinking=false. Qwen3 emits a "
            "<think> block by default, which can consume the whole "
            "--max-tokens budget before any tool call is produced."
        ),
    )
    parser.add_argument(
        "--reset-before",
        action="store_true",
        help=(
            "Reset APC once immediately before this replay. Requires the "
            "server to run with VLLM_SERVER_DEV_MODE=1."
        ),
    )
    parser.add_argument(
        "--allow-counter-mismatch",
        action="store_true",
        help=(
            "Write a contaminated/mismatched run without exiting non-zero. "
            "Never include such a run in the comparison table."
        ),
    )
    parser.add_argument(
        "--sample-kv-usage",
        action="store_true",
        help=(
            "Poll cache occupancy throughout the replay. Use a separate memory "
            "run when latency perturbation from metric scraping is a concern."
        ),
    )
    parser.add_argument("--kv-sample-interval", type=float, default=0.01)
    parser.add_argument(
        "--require-peak-kv-usage",
        type=float,
        help=(
            "Fail after writing results unless sampled KV occupancy reaches this "
            "0-1 fraction; useful for proving a memory-pressure run was nontrivial."
        ),
    )
    args = parser.parse_args()

    if args.kv_sample_interval <= 0:
        parser.error("--kv-sample-interval must be > 0")
    if args.require_peak_kv_usage is not None:
        if not args.sample_kv_usage:
            parser.error("--require-peak-kv-usage requires --sample-kv-usage")
        if not 0 <= args.require_peak_kv_usage <= 1:
            parser.error("--require-peak-kv-usage must be between 0 and 1")

    workload = list(read_jsonl(args.input))
    if args.limit is not None:
        workload = workload[: args.limit]
    if not workload:
        raise SystemExit("Empty workload.")
    try:
        execution_condition = selected_tool_text_condition(
            args.condition_role,
            [str(record.get("ordering")) for record in workload],
        )
    except ValueError as error:
        parser.error(str(error))

    base_url = args.base_url.rstrip("/")
    model = served_model(base_url, args.model)
    cache_config = server_cache_config(base_url)

    if args.reset_before and not reset_prefix_cache(base_url):
        raise SystemExit(
            "vLLM prefix-cache reset failed; verify VLLM_SERVER_DEV_MODE=1"
        )

    results = []
    aggregate_before = parse_prometheus(fetch_text(f"{base_url}/metrics"))
    started = time.perf_counter()
    sampler_context = (
        KvUsageSampler(base_url, interval_seconds=args.kv_sample_interval)
        if args.sample_kv_usage
        else nullcontext(None)
    )
    with sampler_context as sampler:
        for index, record in enumerate(workload):
            payload = {
                "model": model,
                "messages": record["messages"],
                "tools": record["tools"],
                "tool_choice": args.tool_choice,
                "temperature": 0,
                "seed": 0,
                "max_tokens": args.max_tokens,
            }
            if args.disable_thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            before = parse_prometheus(fetch_text(f"{base_url}/metrics"))
            request_started = time.perf_counter()
            response = request_json(
                "POST",
                f"{base_url}/v1/chat/completions",
                payload,
            )
            wall_seconds = time.perf_counter() - request_started
            after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
            projection = response_projection(response)
            delta = metric_delta(before, after)
            measurement = project_request_measurement(delta, projection["usage"])
            measurement["wall_seconds"] = round(wall_seconds, 6)
            results.append(
                {
                    "index": index,
                    "case_id": record.get("case_id", record["task_id"]),
                    "task_id": record["task_id"],
                    "source": record["source"],
                    "domain": record["domain"],
                    "evidence_type": record["evidence_type"],
                    "ordering": record["ordering"],
                    "base_ordering": record.get("base_ordering"),
                    "menu_seed": record.get("menu_seed"),
                    "tool_ids": record["tool_ids"],
                    "canonical_tool_tokens": record["canonical_tool_tokens"],
                    "wall_seconds": round(wall_seconds, 6),
                    **projection,
                    "metric_delta": delta,
                    "measurement": measurement,
                }
            )
            if args.pause_seconds:
                time.sleep(args.pause_seconds)

    aggregate_after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
    aggregate_delta = metric_delta(
        aggregate_before,
        aggregate_after,
    )
    prompt_tokens_from_responses = sum(
        int((result.get("usage") or {}).get("prompt_tokens", 0))
        for result in results
    )
    query_tokens = aggregate_delta.get("vllm:prefix_cache_queries")
    cached_tokens = aggregate_delta.get("vllm:prompt_tokens_cached")
    computed_tokens = aggregate_delta.get(
        "vllm:request_prefill_kv_computed_tokens_sum"
    )
    validation = {
        "query_counter_matches_response_prompt_tokens": (
            query_tokens == prompt_tokens_from_responses
        ),
        "cached_plus_computed_matches_queries": (
            query_tokens is not None
            and cached_tokens is not None
            and computed_tokens is not None
            and cached_tokens + computed_tokens == query_tokens
        ),
    }
    counters_clean = all(validation.values())
    kv_usage = sampler.summary() if sampler is not None else None
    capacity_tokens = 0
    try:
        capacity_tokens = int(cache_config.get("block_size") or 0) * int(
            cache_config.get("num_gpu_blocks") or 0
        )
    except (TypeError, ValueError):
        capacity_tokens = 0
    peak_usage = None
    if kv_usage is not None:
        peak_usage = kv_usage["peak"].get(
            "vllm:kv_cache_usage_perc",
            kv_usage["peak"].get("vllm:gpu_cache_usage_perc"),
        )
    pressure = {
        "capacity_tokens": capacity_tokens or None,
        "prompt_token_volume": prompt_tokens_from_responses,
        "prompt_volume_over_capacity": round(
            prompt_tokens_from_responses / capacity_tokens, 6
        )
        if capacity_tokens
        else None,
        "peak_kv_usage_fraction": peak_usage,
        "estimated_peak_resident_tokens": round(peak_usage * capacity_tokens)
        if peak_usage is not None and capacity_tokens
        else None,
        "required_peak_fraction": args.require_peak_kv_usage,
        "requirement_met": (
            peak_usage is not None
            and args.require_peak_kv_usage is not None
            and peak_usage >= args.require_peak_kv_usage
        )
        if args.require_peak_kv_usage is not None
        else None,
    }
    output = {
        "format_version": 3,
        "engine": "vllm",
        "run_label": args.run_label,
        "execution_condition": execution_condition,
        "server": base_url,
        "model": model,
        "input": args.input.as_posix(),
        "request_count": len(results),
        "wall_seconds": round(time.perf_counter() - started, 6),
        "cache_reset_before": args.reset_before,
        "server_cache_config": cache_config,
        "aggregate_metric_delta": aggregate_delta,
        "counter_validation": {
            **validation,
            "clean": counters_clean,
            "response_prompt_tokens": prompt_tokens_from_responses,
        },
        "direct_measurements_by_reuse_bucket": summarize_request_measurements(
            [result["measurement"] for result in results]
        ),
        "kv_usage_sampling": kv_usage,
        "memory_pressure": pressure,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(results)} results to {args.output}")
    print(json.dumps(output["counter_validation"], indent=2, sort_keys=True))
    if not counters_clean and not args.allow_counter_mismatch:
        raise SystemExit(
            "vLLM counters do not match this replay; preserve and quarantine "
            "the result, then rerun with no other traffic"
        )
    if args.require_peak_kv_usage is not None and not pressure["requirement_met"]:
        raise SystemExit(
            "The replay did not reach the required KV-cache pressure; increase "
            "request/menu volume or reduce the served cache capacity."
        )


if __name__ == "__main__":
    main()
