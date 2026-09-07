#!/usr/bin/env python3
"""Measure whether request ordering changes reuse on the live vLLM cache.

Every earlier GPU experiment resets the prefix cache before each trial, which
makes them clean and independent but structurally unable to answer the
locality question: whether request N still finds request K's tools cached
depends on *everything sent in between*, which a per-trial reset erases.

This sends one continuous session per replay condition -- no resets between
requests, only once at the start of each condition -- and compares aggregate
reuse across two replay orders built from the exact same task multiset
(`empirical` vs `session_bursty` from `replay_workloads`; `uniform`/`skewed`
resample with replacement and so differ in content, not just order, and are
not a fair pair for isolating locality).

Padding each task to a real menu size is required for the same reason as
Task E: unpadded benchmark menus (median 1 tool) sit below the measurement
floor. Total rendered token volume exceeding nominal cache capacity is not by
itself proof of eviction because reuse and request lifetimes affect residency.
This version therefore samples live KV occupancy and can require a predeclared
pressure threshold before a run is accepted as memory-pressure evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.analysis import (
    bounded_trie_metrics,
    deduplicated_existing_ids,
    load_processed,
    replay_workloads,
)
from tatm.measurement import (
    project_request_measurement,
    summarize_request_measurements,
)
from tatm.prompting import build_menu, order_tool_ids, workload_record
from tatm.baselines import OnlineFrequencyPlanner, OnlinePairTriplePlanner
from tatm.tooltrie import ToolTrie
from tatm.tooltrie_weighted import WeightedToolTrie
from tatm.vllm_client import (
    KvUsageSampler,
    fetch_text,
    metric_delta,
    parse_prometheus,
    request_json,
    reset_prefix_cache,
    served_model,
    server_cache_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare live-cache reuse across replay orderings of the same requests."
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument(
        "--partition", choices=("toolret", "bfcl"), default="bfcl"
    )
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--menu-size", type=int, default=64)
    parser.add_argument(
        "--ordering",
        choices=(
            "original",
            "alphabetical",
            "random",
            "frequency",
            "schema_cost_weighted",
            "fp_tree_global",
            "tooltrie",
            "tooltrie_weighted",
            "frequency_online",
            "pair_triple_online",
        ),
        default="alphabetical",
        help=(
            "Intra-menu tool order. The first six are fixed permutations held "
            "constant across replay conditions. 'tooltrie', 'tooltrie_weighted', "
            "'frequency_online' and 'pair_triple_online' plan causally from "
            "earlier requests and so differ per condition."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--replay-seed", type=int, default=2026)
    parser.add_argument("--max-schema-tokens", type=int, default=300)
    parser.add_argument(
        "--support-mode",
        choices=("disjoint", "all", "evaluation"),
        default="disjoint",
        help=(
            "Gold-label split used to fit frequency-based ordering. Disjoint is "
            "the fair default; it does not affect original/alphabetical/random."
        ),
    )
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument(
        "--condition",
        dest="conditions",
        action="append",
        choices=("empirical", "uniform", "skewed", "session_bursty"),
        help=(
            "Replay condition to run; repeat the flag for several. Defaults to "
            "empirical and session_bursty."
        ),
    )
    parser.add_argument("--kv-sample-interval", type=float, default=0.01)
    parser.add_argument(
        "--require-peak-kv-usage",
        type=float,
        help="Require every condition to reach this 0-1 KV occupancy fraction.",
    )
    parser.add_argument(
        "--allow-warm-start",
        action="store_true",
        help="Continue when /reset_prefix_cache is unavailable and mark the run.",
    )
    args = parser.parse_args()

    if args.kv_sample_interval <= 0:
        parser.error("--kv-sample-interval must be > 0")
    if args.offset < 0:
        parser.error("--offset must be >= 0")
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.require_peak_kv_usage is not None and not (
        0 <= args.require_peak_kv_usage <= 1
    ):
        parser.error("--require-peak-kv-usage must be between 0 and 1")

    base_url = args.base_url.rstrip("/")
    model = served_model(base_url, args.model)
    cache_config = server_cache_config(base_url)
    try:
        block_size = int(cache_config.get("block_size") or 0)
        num_gpu_blocks = int(cache_config.get("num_gpu_blocks") or 0)
    except (TypeError, ValueError) as error:
        raise SystemExit(
            "Could not read a valid block_size and num_gpu_blocks from "
            "vllm:cache_config_info."
        ) from error
    capacity_tokens = block_size * num_gpu_blocks
    if block_size < 1 or num_gpu_blocks < 1 or capacity_tokens < 1:
        raise SystemExit(
            "Could not read block_size and num_gpu_blocks from "
            "vllm:cache_config_info; "
            "is the server running with the expected vLLM version?"
        )

    tools, tasks = load_processed(args.processed_dir)
    evidence = "gold_relevance" if args.partition == "toolret" else "exposed_menu"
    partition_tasks = [t for t in tasks if t.evidence_type == evidence]
    selected_tasks = partition_tasks[args.offset : args.offset + args.limit]
    if not selected_tasks:
        raise SystemExit("The requested benchmark slice is empty.")
    selected_ids = {task.task_id for task in selected_tasks}
    if args.support_mode == "disjoint":
        support_tasks = [
            task for task in partition_tasks if task.task_id not in selected_ids
        ]
        support_provenance = "disjoint_benchmark_gold_or_exposed_labels"
    elif args.support_mode == "all":
        support_tasks = partition_tasks
        support_provenance = "all_benchmark_labels_including_evaluation"
    else:
        support_tasks = selected_tasks
        support_provenance = "evaluation_labels"

    distractor_pool = sorted(
        tool_id
        for tool_id, tool in tools.items()
        if 0 < tool.schema_tokens <= args.max_schema_tokens
    )

    support: Counter[str] = Counter()
    for task in support_tasks:
        support.update(deduplicated_existing_ids(task, tools))

    replays = replay_workloads(selected_tasks, tools, seed=args.replay_seed)
    selected_conditions = args.conditions or ["empirical", "session_bursty"]
    conditions = {name: replays[name] for name in selected_conditions}
    empirical_multiset = Counter(task.task_id for task in replays["empirical"])

    def build_sequence(order: list) -> list[tuple[str, ...]]:
        # Every fixed ordering is a stateless permutation applied per task.
        # ToolTrie is causal: it plans from paths observed strictly earlier and
        # observes only afterwards, so it needs a planner that persists across
        # the regime. A fresh planner per regime matches the cache reset that
        # precedes each one — a warm planner against a cold cache would
        # misrepresent both.
        if args.ordering == "tooltrie":
            planner = ToolTrie(
                tools,
                fallback="alphabetical",
                recency_window=128,
                capacity_tokens=capacity_tokens,
            )
        elif args.ordering == "tooltrie_weighted":
            planner = WeightedToolTrie(
                tools,
                fallback="alphabetical",
                recency_window=128,
                capacity_tokens=capacity_tokens,
            )
        elif args.ordering == "frequency_online":
            planner = OnlineFrequencyPlanner(tools)
        elif args.ordering == "pair_triple_online":
            planner = OnlinePairTriplePlanner(tools)
        else:
            planner = None
        sequences = []
        for task in order:
            tool_ids = deduplicated_existing_ids(task, tools)
            if args.menu_size:
                tool_ids = build_menu(
                    tool_ids, distractor_pool, args.menu_size, seed=args.random_seed
                )
            if planner is None:
                ordered = order_tool_ids(
                    tool_ids, tools, support, args.ordering, random_seed=args.random_seed
                )
            else:
                ordered = planner.plan(tool_ids).ordered_ids
                planner.observe(ordered)
            sequences.append(ordered)
        return sequences

    condition_results = {}
    for name, order in conditions.items():
        sequences = build_sequence(order)
        predicted = bounded_trie_metrics(
            sequences, tools, block_size=block_size, capacity_tokens=capacity_tokens
        )

        reset_ok = reset_prefix_cache(base_url)
        if not reset_ok and not args.allow_warm_start:
            raise SystemExit(
                "Prefix-cache reset failed. Start vLLM with "
                "VLLM_SERVER_DEV_MODE=1 or pass --allow-warm-start."
            )
        session_before = parse_prometheus(fetch_text(f"{base_url}/metrics"))
        started = time.perf_counter()
        per_request = []
        with KvUsageSampler(
            base_url, interval_seconds=args.kv_sample_interval
        ) as sampler:
            for index, (task, tool_ids) in enumerate(zip(order, sequences)):
                record = workload_record(task, tool_ids, tools, args.ordering)
                payload = {
                    "model": model,
                    "messages": record["messages"],
                    "tools": record["tools"],
                    "tool_choice": "auto",
                    "temperature": 0,
                    "seed": 0,
                    "max_tokens": 1,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                before = parse_prometheus(fetch_text(f"{base_url}/metrics"))
                request_started = time.perf_counter()
                response = request_json(
                    "POST", f"{base_url}/v1/chat/completions", payload
                )
                wall_seconds = time.perf_counter() - request_started
                after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
                delta = metric_delta(before, after)
                measurement = project_request_measurement(
                    delta, response.get("usage", {})
                )
                measurement["wall_seconds"] = round(wall_seconds, 6)
                per_request.append(
                    {
                        "index": index,
                        "task_id": task.task_id,
                        "canonical_tool_tokens": record["canonical_tool_tokens"],
                        "measurement": measurement,
                    }
                )
                if args.pause_seconds:
                    time.sleep(args.pause_seconds)

        session_after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
        session_delta = metric_delta(session_before, session_after)

        total_canonical_tool_tokens = sum(
            r["canonical_tool_tokens"] for r in per_request
        )
        total_prompt_tokens = sum(
            int(r["measurement"]["prompt_tokens"] or 0) for r in per_request
        )
        total_cached_tokens = sum(
            float(r["measurement"]["cached_tokens"] or 0.0) for r in per_request
        )
        query_counter = session_delta.get("vllm:prefix_cache_queries")
        computed_counter = session_delta.get(
            "vllm:request_prefill_kv_computed_tokens_sum"
        )
        sampler_summary = sampler.summary()
        peak_usage = sampler_summary["peak"].get(
            "vllm:kv_cache_usage_perc",
            sampler_summary["peak"].get("vllm:gpu_cache_usage_perc"),
        )
        pressure_met = (
            (
                peak_usage is not None
                and peak_usage >= args.require_peak_kv_usage
            )
            if args.require_peak_kv_usage is not None
            else None
        )
        condition_results[name] = {
            "requests": len(sequences),
            "cache_reset_before": reset_ok,
            "same_task_multiset_as_empirical": (
                Counter(task.task_id for task in order) == empirical_multiset
            ),
            "total_canonical_tool_tokens": total_canonical_tool_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "measured_cached_tokens": total_cached_tokens,
            "measured_reuse_ratio": round(
                total_cached_tokens / total_prompt_tokens, 6
            )
            if total_prompt_tokens
            else 0.0,
            "predicted": predicted,
            "wall_seconds": round(time.perf_counter() - started, 6),
            "session_metric_delta": session_delta,
            "counter_validation": {
                "query_counter_matches_response_prompt_tokens": (
                    query_counter == total_prompt_tokens
                ),
                "cached_plus_computed_matches_queries": (
                    query_counter is not None
                    and computed_counter is not None
                    and total_cached_tokens + computed_counter == query_counter
                ),
            },
            "direct_measurements_by_reuse_bucket": summarize_request_measurements(
                [row["measurement"] for row in per_request]
            ),
            "kv_usage_sampling": sampler_summary,
            "memory_pressure": {
                "capacity_tokens": capacity_tokens,
                "prompt_token_volume": total_prompt_tokens,
                "prompt_volume_over_capacity": round(
                    total_prompt_tokens / capacity_tokens, 6
                ),
                "peak_kv_usage_fraction": peak_usage,
                "estimated_peak_resident_tokens": round(
                    peak_usage * capacity_tokens
                )
                if peak_usage is not None
                else None,
                "required_peak_fraction": args.require_peak_kv_usage,
                "requirement_met": pressure_met,
            },
            "per_request": per_request,
        }
        print(
            f"{name}: measured reuse "
            f"{condition_results[name]['measured_reuse_ratio']:.4f}, "
            f"predicted (bounded trie) "
            f"{predicted['estimated_block_reuse_ratio']:.4f}"
        )

    output = {
        "format_version": 2,
        "run_label": args.run_label,
        "server": base_url,
        "model": model,
        "partition": args.partition,
        "ordering": args.ordering,
        "menu_size": args.menu_size,
        "task_count": len(selected_tasks),
        "task_offset": args.offset,
        "ordering_support": {
            "mode": args.support_mode,
            "provenance": support_provenance,
            "tasks": len(support_tasks),
            "evaluation_overlap_tasks": len(
                selected_ids & {task.task_id for task in support_tasks}
            ),
        },
        "cache_capacity_tokens": capacity_tokens,
        "cache_config": cache_config,
        "conditions": condition_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    if args.require_peak_kv_usage is not None:
        failed = [
            name
            for name, result in condition_results.items()
            if not result["memory_pressure"]["requirement_met"]
        ]
        if failed:
            raise SystemExit(
                "KV-pressure requirement was not reached for: " + ", ".join(failed)
            )


if __name__ == "__main__":
    main()
