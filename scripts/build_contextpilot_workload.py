#!/usr/bin/env python3
"""Build a tool-ordering workload with the actual ContextPilot codebase.

This is deliberately an offline/transductive comparison: ContextPilot sees the
whole evaluation batch to construct its context index. ``intra`` keeps the
request sequence fixed; ``intra_schedule`` additionally uses ContextPilot's
inter-context scheduler (whose mapping can be identity) and must be reported
separately from fixed-order runs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.contextpilot_adapter import materialize_contextpilot_workload
from tatm.io import read_jsonl, write_json, write_jsonl


EXPECTED_CONTEXTPILOT_COMMIT = "1fa0a143fdeda344585666648ab2b30cb7fea77f"


def git_output(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply pinned ContextPilot context ordering to tool IDs."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Default: <output>.summary.json",
    )
    parser.add_argument(
        "--contextpilot-repo",
        type=Path,
        required=True,
        help="Clean checkout of the pinned upstream ContextPilot repository.",
    )
    parser.add_argument(
        "--expected-commit",
        default=EXPECTED_CONTEXTPILOT_COMMIT,
        help="Fail if the external checkout is not this exact commit.",
    )
    parser.add_argument(
        "--mode",
        choices=("intra", "intra_schedule"),
        default="intra",
    )
    parser.add_argument(
        "--linkage-method",
        choices=("average", "complete", "single"),
        default="average",
    )
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.alpha < 0:
        parser.error("--alpha must be >= 0")
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    reference_repo = args.contextpilot_repo.resolve()
    commit = git_output(reference_repo, "rev-parse", "HEAD")
    if commit != args.expected_commit:
        parser.error(
            f"ContextPilot checkout is {commit}, expected {args.expected_commit}"
        )
    dirty = git_output(reference_repo, "status", "--porcelain")
    if dirty:
        parser.error("ContextPilot checkout has local modifications")

    try:
        context_index_module = importlib.import_module("contextpilot.context_index")
        ordering_module = importlib.import_module("contextpilot.context_ordering")
    except ImportError as error:
        parser.error(
            "ContextPilot is unavailable in this Python environment; install "
            "the pinned checkout as described in NUS_GPU_PHASE2_INSTRUCTIONS.md "
            f"({error})"
        )
    imported_path = Path(context_index_module.__file__).resolve()
    if reference_repo not in imported_path.parents:
        parser.error(
            f"Imported ContextPilot from {imported_path}, not {reference_repo}"
        )

    records = list(read_jsonl(args.input))
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        parser.error("Input workload is empty")

    tool_ids = sorted(
        {
            str(tool_id)
            for record in records
            for tool_id in record.get("tool_ids", [])
        }
    )
    id_to_int = {tool_id: index for index, tool_id in enumerate(tool_ids)}
    int_to_id = {index: tool_id for tool_id, index in id_to_int.items()}
    contexts = [
        [id_to_int[str(tool_id)] for tool_id in record["tool_ids"]]
        for record in records
    ]

    indexer = context_index_module.ContextIndex(
        linkage_method=args.linkage_method,
        use_gpu=False,
        alpha=args.alpha,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    result = indexer.fit_transform(contexts)
    if args.mode == "intra":
        reordered_integer_contexts = result.reordered_contexts
        original_indices = list(range(len(records)))
        group_count = None
    else:
        scheduler = ordering_module.InterContextScheduler()
        (
            reordered_integer_contexts,
            _scheduled_originals,
            original_indices,
            groups,
        ) = scheduler.schedule_contexts(result)
        group_count = len(groups)

    reordered_contexts = [
        [int_to_id[int(item)] for item in context]
        for context in reordered_integer_contexts
    ]
    provenance = {
        "repository": "https://github.com/EfficientContext/ContextPilot",
        "commit": commit,
        "alpha": args.alpha,
        "linkage_method": args.linkage_method,
        "use_gpu_for_planning": False,
    }
    output_records = materialize_contextpilot_workload(
        records,
        reordered_contexts,
        list(original_indices),
        result.search_paths,
        mode=args.mode,
        provenance=provenance,
    )
    count = write_jsonl(args.output, output_records)
    summary = {
        "format_version": 1,
        "requests": count,
        "input": args.input.as_posix(),
        "output": args.output.as_posix(),
        "ordering": f"contextpilot_{args.mode}",
        "offline_transductive": True,
        "scheduler_enabled": args.mode == "intra_schedule",
        "request_order_changed": list(original_indices) != list(range(len(records))),
        "unique_tools": len(tool_ids),
        "group_count": group_count,
        "index_stats": _json_scalar(result.stats),
        "reference": provenance,
    }
    summary_output = args.summary_output or args.output.with_suffix(
        args.output.suffix + ".summary.json"
    )
    write_json(summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {count} ContextPilot requests to {args.output}")


if __name__ == "__main__":
    main()
