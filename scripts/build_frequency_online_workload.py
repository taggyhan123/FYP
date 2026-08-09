#!/usr/bin/env python3
"""Apply the causal online presence-frequency ordering.

This is the adaptive counterpart to ``build_fitted_ordering_workload.py``.
That script freezes support on a task-disjoint training corpus; this one
estimates presence frequency from the served stream itself, so it needs no
training input. ``plan`` is called before ``observe`` for every request, so no
request can influence its own ordering.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.analysis import load_processed
from tatm.baselines import OnlineFrequencyPlanner
from tatm.io import read_jsonl, write_jsonl


def aligned_tools(record: dict[str, Any]) -> dict[str, Any]:
    tool_ids = record["tool_ids"]
    payloads = record["tools"]
    if len(tool_ids) != len(payloads):
        raise ValueError("Record has mismatched tool_ids and tools lengths")
    return dict(zip(tool_ids, payloads, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply causal online presence-frequency tool ordering."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    tools, _ = load_processed(args.processed_dir)
    planner = OnlineFrequencyPlanner(tools)

    source_records = list(read_jsonl(args.input))
    if args.limit is not None:
        source_records = source_records[: args.limit]

    output_records = []
    for record in source_records:
        payload_by_id = aligned_tools(record)
        selected_ids = tuple(record["tool_ids"])
        plan = planner.plan(selected_ids)

        reordered = dict(record)
        reordered["base_ordering"] = record.get("ordering")
        reordered["ordering"] = "frequency_online"
        reordered["tool_ids"] = list(plan.ordered_ids)
        reordered["tools"] = [payload_by_id[item] for item in plan.ordered_ids]
        reordered["frequency_online_plan"] = plan.to_record()
        # Observe only after the decision has been recorded.
        reordered["frequency_online_state_after"] = planner.observe(plan.ordered_ids)
        output_records.append(reordered)

    count = write_jsonl(args.output, output_records)
    summary = {
        "format_version": 1,
        "requests": count,
        "policy": "frequency_online",
        "information_regime": "causal",
        "training_input": None,
        "adapts_online": True,
        "input": args.input.as_posix(),
        "output": args.output.as_posix(),
        "final_state": planner.snapshot(),
    }
    if args.summary_output is not None:
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
