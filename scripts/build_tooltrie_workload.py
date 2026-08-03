#!/usr/bin/env python3
"""Causally reorder an existing workload with the ToolTrie-v0 planner.

The input must be produced by ``build_cluster_workload.py`` or
``build_bfcl_quality_workload.py`` so each ``tool_id`` is aligned with one
OpenAI-compatible tool object. Planning happens before observation for every
record, preventing the current or future requests from influencing their own
ordering.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.analysis import load_processed
from tatm.io import read_jsonl, write_jsonl
from tatm.baselines import CacheWeaverPlanner
from tatm.tooltrie import ToolTrie


def support_from_workload(path: Path) -> Counter[str]:
    support: Counter[str] = Counter()
    for record in read_jsonl(path):
        support.update(dict.fromkeys(record["tool_ids"]))
    return support


def aligned_tools(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tool_ids = record.get("tool_ids")
    tool_payloads = record.get("tools")
    if not isinstance(tool_ids, list) or not isinstance(tool_payloads, list):
        raise ValueError(
            f"{record.get('task_id', '<unknown>')}: tool_ids/tools must be lists"
        )
    if len(tool_ids) != len(tool_payloads) or len(tool_ids) != len(set(tool_ids)):
        raise ValueError(
            f"{record.get('task_id', '<unknown>')}: tool_ids and tools are not "
            "a one-to-one aligned menu"
        )
    return dict(zip(tool_ids, tool_payloads, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the causal recent-path ToolTrie-v0 ordering."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("tooltrie_v0", "cacheweaver"),
        default="tooltrie_v0",
        help=(
            "Planner to apply. cacheweaver is a faithful transcription of "
            "Algorithm 1 and always falls back to original input order."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--fallback",
        choices=("alphabetical", "frequency"),
        default="alphabetical",
    )
    parser.add_argument(
        "--training-input",
        type=Path,
        help=(
            "Separate workload used only to fit frozen frequency support. "
            "Required for --fallback frequency and must not contain evaluation "
            "requests."
        ),
    )
    parser.add_argument("--recency-window", type=int, default=128)
    parser.add_argument(
        "--capacity-tokens",
        type=int,
        help=(
            "Approximate planner metadata budget in schema tokens. For the "
            "current RTX 3090 setup, block_size * num_gpu_blocks is 188912."
        ),
    )
    parser.add_argument("--max-nodes", type=int, default=100_000)
    args = parser.parse_args()

    if (
        args.policy == "tooltrie_v0"
        and args.fallback == "frequency"
        and args.training_input is None
    ):
        parser.error("--fallback frequency requires --training-input")
    if (
        args.training_input is not None
        and args.training_input.resolve() == args.input.resolve()
    ):
        parser.error("--training-input must be separate from --input")

    tools, _ = load_processed(args.processed_dir)
    support = (
        support_from_workload(args.training_input)
        if args.training_input is not None
        else None
    )
    if args.policy == "cacheweaver":
        if args.training_input is not None:
            parser.error("CacheWeaver does not fit support from --training-input")
        planner = CacheWeaverPlanner(tools, history_window=args.recency_window)
    else:
        planner = ToolTrie(
            tools,
            fallback=args.fallback,
            support=support,
            recency_window=args.recency_window,
            capacity_tokens=args.capacity_tokens,
            max_nodes=args.max_nodes,
        )

    source_records = list(read_jsonl(args.input))
    if args.limit is not None:
        source_records = source_records[: args.limit]

    output_records = []
    total_hinted_tokens = 0
    matched_requests = 0
    for record in source_records:
        payload_by_id = aligned_tools(record)
        selected_ids = tuple(record["tool_ids"])
        plan = planner.plan(selected_ids)
        if plan.matched_prefix_ids:
            matched_requests += 1
        total_hinted_tokens += plan.hinted_schema_tokens

        reordered = dict(record)
        reordered["base_ordering"] = record.get("ordering")
        reordered["ordering"] = args.policy
        reordered["tool_ids"] = list(plan.ordered_ids)
        reordered["tools"] = [payload_by_id[item] for item in plan.ordered_ids]
        plan_field = (
            "cacheweaver_plan"
            if args.policy == "cacheweaver"
            else "tooltrie_plan"
        )
        state_field = (
            "cacheweaver_state_after"
            if args.policy == "cacheweaver"
            else "tooltrie_state_after"
        )
        reordered[plan_field] = plan.to_record()
        # Observe only after the decision has been recorded.
        reordered[state_field] = planner.observe(plan.ordered_ids)
        output_records.append(reordered)

    count = write_jsonl(args.output, output_records)
    summary = {
        "requests": count,
        "policy": args.policy,
        "requests_with_hinted_prefix": matched_requests,
        "hinted_schema_tokens": total_hinted_tokens,
        "fallback": args.fallback if args.policy == "tooltrie_v0" else "original",
        "training_input": (
            args.training_input.as_posix() if args.training_input else None
        ),
        "final_state": planner.snapshot(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {count} causally planned requests to {args.output}")


if __name__ == "__main__":
    main()
