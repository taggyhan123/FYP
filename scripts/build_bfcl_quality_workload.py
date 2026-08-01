#!/usr/bin/env python3
"""Build a BFCL workload stratified across all five task categories.

`build_cluster_workload.py --partition bfcl` takes the first N tasks in
dataset order, which is entirely `irrelevance` tasks (see
`src/tatm/datasets.py::process_bfcl`, which globs configs alphabetically and
`irrelevance` sorts first). That is fine for latency experiments but useless
for a quality check, which needs simple/multiple/parallel/parallel_multiple
tasks to score function-name and argument accuracy, not just the no-tool
decision. This samples a fixed count per category instead.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.analysis import deduplicated_existing_ids, load_processed
from tatm.io import write_jsonl
from tatm.prompting import build_menu, order_tool_ids, workload_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a category-stratified BFCL quality workload."
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument("--output", type=Path, required=True)
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
        required=True,
    )
    parser.add_argument("--per-domain", type=int, default=20)
    parser.add_argument("--menu-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-schema-tokens", type=int, default=300)
    args = parser.parse_args()

    tools, tasks = load_processed(args.processed_dir)
    bfcl_tasks = [task for task in tasks if task.evidence_type == "exposed_menu"]

    support: Counter[str] = Counter()
    for task in bfcl_tasks:
        support.update(deduplicated_existing_ids(task, tools))

    distractor_pool = sorted(
        tool_id
        for tool_id, tool in tools.items()
        if 0 < tool.schema_tokens <= args.max_schema_tokens
    )

    by_domain: dict[str, list] = defaultdict(list)
    for task in bfcl_tasks:
        by_domain[task.domain].append(task)

    selected = []
    for domain in sorted(by_domain):
        selected.extend(by_domain[domain][: args.per_domain])

    records = []
    for task in selected:
        tool_ids = deduplicated_existing_ids(task, tools)
        if args.menu_size:
            tool_ids = build_menu(
                tool_ids, distractor_pool, args.menu_size, seed=args.random_seed
            )
        ordered = order_tool_ids(
            tool_ids, tools, support, args.ordering, random_seed=args.random_seed
        )
        records.append(workload_record(task, ordered, tools, args.ordering))

    count = write_jsonl(args.output, records)
    domain_counts = Counter(task.domain for task in selected)
    print(f"Wrote {count} requests to {args.output}: {dict(sorted(domain_counts.items()))}")


if __name__ == "__main__":
    main()
