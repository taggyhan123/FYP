#!/usr/bin/env python3
"""Apply frozen training-only static and co-occurrence ordering baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.analysis import load_processed
from tatm.baselines import FittedOrderingPlanner
from tatm.io import read_jsonl, write_jsonl


def aligned_tools(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tool_ids = record.get("tool_ids")
    payloads = record.get("tools")
    if not isinstance(tool_ids, list) or not isinstance(payloads, list):
        raise ValueError("tool_ids/tools must be aligned lists")
    if len(tool_ids) != len(payloads) or len(tool_ids) != len(set(tool_ids)):
        raise ValueError("tool_ids/tools are not a one-to-one aligned menu")
    return dict(zip(tool_ids, payloads, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a frozen fitted ordering to an existing workload."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--training-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=tuple(sorted(FittedOrderingPlanner.POLICIES)),
        required=True,
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-cooccurrence-transaction-size", type=int, default=25)
    parser.add_argument(
        "--exclude-evaluation-task-ids",
        action="store_true",
        help=(
            "Remove any training record whose task_id occurs in --input. "
            "Without this flag, any overlap is rejected."
        ),
    )
    args = parser.parse_args()

    if args.input.resolve() == args.training_input.resolve():
        parser.error("--training-input must be separate from --input")

    tools, _ = load_processed(args.processed_dir)
    source_records = list(read_jsonl(args.input))
    if args.limit is not None:
        source_records = source_records[: args.limit]
    if not source_records:
        parser.error("Input workload is empty")

    raw_training_records = list(read_jsonl(args.training_input))
    evaluation_task_ids = {record["task_id"] for record in source_records}
    overlapping_task_ids = {
        record["task_id"]
        for record in raw_training_records
        if record["task_id"] in evaluation_task_ids
    }
    if overlapping_task_ids and not args.exclude_evaluation_task_ids:
        sample = ", ".join(sorted(overlapping_task_ids)[:3])
        parser.error(
            "Training/evaluation task IDs overlap; pass "
            f"--exclude-evaluation-task-ids to remove them ({sample})"
        )
    training_records = [
        record
        for record in raw_training_records
        if record["task_id"] not in evaluation_task_ids
    ]
    if not training_records:
        parser.error("No disjoint training records remain")
    planner = FittedOrderingPlanner(
        tools,
        [record["tool_ids"] for record in training_records],
        policy=args.policy,
        max_cooccurrence_transaction_size=args.max_cooccurrence_transaction_size,
    )

    output_records = []
    for record in source_records:
        payload_by_id = aligned_tools(record)
        plan = planner.plan(record["tool_ids"])
        reordered = dict(record)
        reordered["base_ordering"] = record.get("ordering")
        reordered["ordering"] = args.policy
        reordered["tool_ids"] = list(plan.ordered_ids)
        reordered["tools"] = [payload_by_id[item] for item in plan.ordered_ids]
        reordered["fitted_ordering_plan"] = plan.to_record()
        output_records.append(reordered)

    count = write_jsonl(args.output, output_records)
    summary = {
        "requests": count,
        "policy": args.policy,
        "training_requests": len(training_records),
        "raw_training_requests": len(raw_training_records),
        "excluded_overlapping_requests": (
            len(raw_training_records) - len(training_records)
        ),
        "training_input": args.training_input.as_posix(),
        "state": planner.snapshot(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {count} fitted-order requests to {args.output}")


if __name__ == "__main__":
    main()
