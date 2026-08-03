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
    parser.add_argument(
        "--domain",
        dest="domains",
        action="append",
        choices=(
            "irrelevance",
            "multiple",
            "parallel",
            "parallel_multiple",
            "simple_python",
        ),
        help=(
            "Restrict the workload to one or more BFCL domains. Repeat the "
            "flag to select multiple domains. By default all domains are used."
        ),
    )
    parser.add_argument(
        "--offset-per-domain",
        type=int,
        default=0,
        help="Skip this many records at the start of every selected domain.",
    )
    parser.add_argument("--menu-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-schema-tokens", type=int, default=300)
    args = parser.parse_args()

    if args.per_domain < 1:
        parser.error("--per-domain must be >= 1")
    if args.offset_per_domain < 0:
        parser.error("--offset-per-domain must be >= 0")

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

    selected_domains = args.domains or sorted(by_domain)
    missing_domains = sorted(set(selected_domains) - set(by_domain))
    if missing_domains:
        parser.error(f"No processed BFCL tasks for: {', '.join(missing_domains)}")

    selected = []
    for domain in selected_domains:
        start = args.offset_per_domain
        stop = start + args.per_domain
        selected.extend(by_domain[domain][start:stop])

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
        record = workload_record(task, ordered, tools, args.ordering)
        # ``case_id`` distinguishes the same BFCL task evaluated against
        # several fixed distractor catalogs.  The task ID remains unchanged so
        # official/reduced BFCL ground truth lookup still works.
        record["menu_seed"] = args.random_seed
        record["case_id"] = f"{task.task_id}:menu_seed={args.random_seed}"
        records.append(record)

    count = write_jsonl(args.output, records)
    domain_counts = Counter(task.domain for task in selected)
    print(f"Wrote {count} requests to {args.output}: {dict(sorted(domain_counts.items()))}")


if __name__ == "__main__":
    main()
