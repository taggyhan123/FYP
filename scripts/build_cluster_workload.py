#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.analysis import deduplicated_existing_ids, load_processed
from tatm.io import write_jsonl
from tatm.prompting import build_menu, order_tool_ids, workload_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic OpenAI-compatible tool workloads."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "cluster-workload.jsonl",
    )
    parser.add_argument(
        "--partition",
        choices=("toolret", "bfcl"),
        default="bfcl",
    )
    parser.add_argument(
        "--ordering",
        choices=(
            "original",
            "alphabetical",
            "random",
            "frequency",
            "schema_cost_weighted",
            "fp_tree_global",
        ),
        default="frequency",
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--menu-size",
        type=int,
        default=0,
        help=(
            "Pad each task's gold tools with distractors up to this many tools. "
            "0 keeps the benchmark's own menu, which has a median of one tool "
            "and is too small to measure prefill effects."
        ),
    )
    parser.add_argument(
        "--max-schema-tokens",
        type=int,
        default=300,
        help="Exclude outlier schemas from the distractor pool.",
    )
    args = parser.parse_args()

    tools, tasks = load_processed(args.processed_dir)
    evidence = "gold_relevance" if args.partition == "toolret" else "exposed_menu"
    selected_tasks = [task for task in tasks if task.evidence_type == evidence]
    support: Counter[str] = Counter()
    for task in selected_tasks:
        support.update(deduplicated_existing_ids(task, tools))

    distractor_pool = sorted(
        tool_id
        for tool_id, tool in tools.items()
        if 0 < tool.schema_tokens <= args.max_schema_tokens
    )

    records = []
    for task in selected_tasks[: args.limit]:
        tool_ids = deduplicated_existing_ids(task, tools)
        if args.menu_size:
            tool_ids = build_menu(
                tool_ids,
                distractor_pool,
                args.menu_size,
                seed=args.random_seed,
            )
        ordered = order_tool_ids(
            tool_ids,
            tools,
            support,
            args.ordering,
            random_seed=args.random_seed,
        )
        records.append(workload_record(task, ordered, tools, args.ordering))
    count = write_jsonl(args.output, records)
    print(f"Wrote {count} requests to {args.output}")


if __name__ == "__main__":
    main()
