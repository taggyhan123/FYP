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
floor. At menu-size 64 the total token volume across ~100+ tasks (~650k-700k
tokens) already exceeds this server's real cache capacity
(block_size * num_gpu_blocks, ~189k tokens on the RTX 3090 used elsewhere in
this project), so eviction happens on real hardware without needing to
shrink --gpu-memory-utilization.
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
from tatm.prompting import build_menu, order_tool_ids, workload_record
from tatm.vllm_client import (
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
    parser.add_argument("--menu-size", type=int, default=64)
    parser.add_argument(
        "--ordering",
        default="alphabetical",
        help="Fixed intra-menu tool order, held constant across both replay conditions.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--replay-seed", type=int, default=2026)
    parser.add_argument("--max-schema-tokens", type=int, default=300)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model = served_model(base_url, args.model)
    cache_config = server_cache_config(base_url)
    block_size = int(cache_config["block_size"] or 16)
    num_gpu_blocks = int(cache_config["num_gpu_blocks"] or 0)
    capacity_tokens = block_size * num_gpu_blocks
    if not capacity_tokens:
        raise SystemExit(
            "Could not read num_gpu_blocks from vllm:cache_config_info; "
            "is the server running with the expected vLLM version?"
        )

    tools, tasks = load_processed(args.processed_dir)
    evidence = "gold_relevance" if args.partition == "toolret" else "exposed_menu"
    selected_tasks = [t for t in tasks if t.evidence_type == evidence][: args.limit]

    distractor_pool = sorted(
        tool_id
        for tool_id, tool in tools.items()
        if 0 < tool.schema_tokens <= args.max_schema_tokens
    )

    support: Counter[str] = Counter()
    for task in selected_tasks:
        support.update(deduplicated_existing_ids(task, tools))

    replays = replay_workloads(selected_tasks, tools, seed=args.replay_seed)
    conditions = {"empirical": replays["empirical"], "session_bursty": replays["session_bursty"]}

    def build_sequence(order: list) -> list[tuple[str, ...]]:
        sequences = []
        for task in order:
            tool_ids = deduplicated_existing_ids(task, tools)
            if args.menu_size:
                tool_ids = build_menu(
                    tool_ids, distractor_pool, args.menu_size, seed=args.random_seed
                )
            ordered = order_tool_ids(
                tool_ids, tools, support, args.ordering, random_seed=args.random_seed
            )
            sequences.append(ordered)
        return sequences

    condition_results = {}
    for name, order in conditions.items():
        sequences = build_sequence(order)
        predicted = bounded_trie_metrics(
            sequences, tools, block_size=block_size, capacity_tokens=capacity_tokens
        )

        reset_prefix_cache(base_url)
        session_before = parse_prometheus(fetch_text(f"{base_url}/metrics"))
        started = time.perf_counter()
        per_request = []
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
            request_json("POST", f"{base_url}/v1/chat/completions", payload)
            after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
            delta = metric_delta(before, after)
            per_request.append(
                {
                    "index": index,
                    "task_id": task.task_id,
                    "canonical_tool_tokens": record["canonical_tool_tokens"],
                    "prompt_tokens_cached": delta.get("vllm:prompt_tokens_cached", 0.0),
                }
            )
            if args.pause_seconds:
                time.sleep(args.pause_seconds)

        session_after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
        session_delta = metric_delta(session_before, session_after)

        total_prompt_tokens = sum(r["canonical_tool_tokens"] for r in per_request)
        total_cached_tokens = sum(r["prompt_tokens_cached"] for r in per_request)
        condition_results[name] = {
            "requests": len(sequences),
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
            "per_request": per_request,
        }
        print(
            f"{name}: measured reuse "
            f"{condition_results[name]['measured_reuse_ratio']:.4f}, "
            f"predicted (bounded trie) "
            f"{predicted['estimated_block_reuse_ratio']:.4f}"
        )

    output = {
        "format_version": 1,
        "run_label": args.run_label,
        "server": base_url,
        "model": model,
        "partition": args.partition,
        "ordering": args.ordering,
        "menu_size": args.menu_size,
        "task_count": len(selected_tasks),
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


if __name__ == "__main__":
    main()
