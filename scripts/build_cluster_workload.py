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
from tatm.prompting import order_tool_ids, workload_record


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
    args = parser.parse_args()

    tools, tasks = load_processed(args.processed_dir)
    evidence = "gold_relevance" if args.partition == "toolret" else "exposed_menu"
    selected_tasks = [task for task in tasks if task.evidence_type == evidence]
    support: Counter[str] = Counter()
    for task in selected_tasks:
        support.update(deduplicated_existing_ids(task, tools))

    records = []
    for task in selected_tasks[: args.limit]:
        ordered = order_tool_ids(
            task.tool_ids,
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
