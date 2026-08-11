#!/usr/bin/env python3
"""Apply the causal online pair/triple co-occurrence ordering.

The adaptive counterpart to ``build_fitted_ordering_workload.py``'s
``conditional_pair`` and ``conditional_pair_triple`` policies, which freeze
co-occurrence statistics on a task-disjoint training corpus. Those policies
emit orderings byte-identical to plain ``frequency_fitted`` on every workload
measured so far, so brief section 7 question 5 has never actually been posed.

Two separate reasons produce that collapse, and only one is fixed here. The
fitted policies fail because a disjoint corpus never observed the evaluation
menus' pairs, so every lookup returns zero; estimating online fixes that.
Padded menus fail for a structural reason no planner can fix: when a menu is a
fixed core plus one varying tool, pair support is a deterministic function of
presence counts, so a pair key carries no signal a frequency key does not.
``scripts/audit_pair_triple_information.py`` measures this at 100.00% of
``bfcl-padded64`` pairs with zero violations, against 67.34% on
``toolret-bm25-k128``.

Run this on retrieved menus. On padded menus the result is a foregone
conclusion.

``plan`` is called before ``observe`` for every request, so no request can
influence its own ordering.
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
from tatm.baselines import OnlinePairTriplePlanner
from tatm.io import read_jsonl, write_jsonl


def aligned_tools(record: dict[str, Any]) -> dict[str, Any]:
    tool_ids = record["tool_ids"]
    payloads = record["tools"]
    if len(tool_ids) != len(payloads):
        raise ValueError("Record has mismatched tool_ids and tools lengths")
    return dict(zip(tool_ids, payloads, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply causal online pair/triple co-occurrence tool ordering."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    tools, _ = load_processed(args.processed_dir)
    planner = OnlinePairTriplePlanner(tools)

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
        reordered["ordering"] = "pair_triple_online"
        reordered["tool_ids"] = list(plan.ordered_ids)
        reordered["tools"] = [payload_by_id[item] for item in plan.ordered_ids]
        reordered["pair_triple_online_plan"] = plan.to_record()
        # Observe only after the decision has been recorded.
        reordered["pair_triple_online_state_after"] = planner.observe(plan.ordered_ids)
        output_records.append(reordered)

    count = write_jsonl(args.output, output_records)
    summary = {
        "format_version": 1,
        "requests": count,
        "policy": "pair_triple_online",
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
