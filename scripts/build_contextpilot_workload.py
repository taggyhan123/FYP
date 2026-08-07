#!/usr/bin/env python3
"""Build explicitly labelled ContextPilot-derived tool-ordering workloads.

Offline/transductive, causal static-refit, and persistent incremental modes are
different algorithms and are emitted under different ordering names.  This
script also records the parameter, dependency, and API provenance needed to
avoid presenting a tool-order adaptation as the full ContextPilot system.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.contextpilot_adapter import (
    build_online_incremental_orderings,
    build_static_refit_causal_orderings,
    materialize_contextpilot_causal_workload,
    materialize_contextpilot_workload,
)
from tatm.io import read_jsonl, write_json, write_jsonl


EXPECTED_CONTEXTPILOT_COMMIT = "1fa0a143fdeda344585666648ab2b30cb7fea77f"
CAUSAL_MODES = ("static_refit_causal", "online_incremental")
OFFLINE_MODES = ("intra", "intra_schedule")


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def planning_summary(
    plans: list[dict[str, Any]], request_count: int
) -> dict[str, Any]:
    values = [float(plan.get("planning_seconds", 0.0)) for plan in plans]
    total = sum(values)
    return {
        "requests": request_count,
        "planning_operations": len(values),
        "total_seconds": round(total, 9),
        "mean_seconds_per_request": (
            round(total / request_count, 9) if request_count else None
        ),
        "max_operation_seconds": round(max(values), 9) if values else None,
    }


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
        choices=(*OFFLINE_MODES, *CAUSAL_MODES),
        default="intra",
        help=(
            "intra/intra_schedule are offline batch modes; "
            "static_refit_causal refits each observed prefix; "
            "online_incremental uses ContextPilot.reorder with a persistent index"
        ),
    )
    parser.add_argument(
        "--linkage-method",
        choices=("average", "complete", "single"),
        default="average",
    )
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument(
        "--allow-nonstandard-alpha",
        action="store_true",
        help=(
            "Allow alpha outside the paper's declared [0.001, 0.01] range. "
            "The nonstandard value remains recorded in provenance."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument(
        "--conversation-prefix",
        default="tatm-evaluation",
        help=(
            "Prefix for per-request conversation IDs in online_incremental mode. "
            "Independent benchmark cases never share de-duplication state."
        ),
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.alpha < 0:
        parser.error("--alpha must be >= 0")
    if not 0.001 <= args.alpha <= 0.01 and not args.allow_nonstandard_alpha:
        parser.error(
            "--alpha must be within the paper's [0.001, 0.01] range; pass "
            "--allow-nonstandard-alpha only for an explicitly labelled "
            "sensitivity or historical reproduction"
        )
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
        live_index_module = (
            importlib.import_module("contextpilot.server.live_index")
            if args.mode == "online_incremental"
            else None
        )
    except ImportError as error:
        parser.error(
            "ContextPilot is unavailable in this Python environment; install "
            "the pinned checkout as described in "
            "NUS_GPU_CONTEXTPILOT_CONFIRMATION_INSTRUCTIONS.md "
            f"({error})"
        )
    imported_path = Path(context_index_module.__file__).resolve()
    if reference_repo not in imported_path.parents:
        parser.error(
            f"Imported ContextPilot from {imported_path}, not {reference_repo}"
        )
    if live_index_module is not None:
        live_imported_path = Path(live_index_module.__file__).resolve()
        if reference_repo not in live_imported_path.parents:
            parser.error(
                f"Imported ContextPilot live API from {live_imported_path}, "
                f"not {reference_repo}"
            )

    records = list(read_jsonl(args.input))
    input_record_count = len(records)
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
    package_versions = {
        name: installed_version(name) for name in ("contextpilot", "numpy", "scipy")
    }
    provenance = {
        "repository": "https://github.com/EfficientContext/ContextPilot",
        "commit": commit,
        "package_versions": package_versions,
        "alpha": args.alpha,
        "alpha_within_paper_range": 0.001 <= args.alpha <= 0.01,
        "linkage_method": args.linkage_method,
        "use_gpu_for_planning": False,
        "context_unit": "tool_schema",
        "annotations_enabled": False,
        "deduplication_enabled": False,
        "eviction_feedback_enabled": False,
    }
    plans: list[dict[str, Any]] = []
    index_stats: Any = None
    group_count = None
    request_order_changed = False
    if args.mode in OFFLINE_MODES:
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
        planning_started = time.perf_counter()
        result = indexer.fit_transform(contexts)
        if args.mode == "intra":
            reordered_integer_contexts = result.reordered_contexts
            original_indices = list(range(len(records)))
        else:
            scheduler = ordering_module.InterContextScheduler()
            (
                reordered_integer_contexts,
                _scheduled_originals,
                original_indices,
                groups,
            ) = scheduler.schedule_contexts(result)
            group_count = len(groups)
        planning_seconds = time.perf_counter() - planning_started
        plans = [{"planning_seconds": planning_seconds}]
        reordered_contexts = [
            [int_to_id[int(item)] for item in context]
            for context in reordered_integer_contexts
        ]
        output_records = materialize_contextpilot_workload(
            records,
            reordered_contexts,
            list(original_indices),
            result.search_paths,
            mode=args.mode,
            provenance=provenance,
        )
        request_order_changed = list(original_indices) != list(range(len(records)))
        index_stats = result.stats
        ordering = f"contextpilot_{args.mode}"
    elif args.mode == "static_refit_causal":
        provenance.update(
            {
                "api": "ContextIndex.fit_transform over each observed prefix",
                "persistent_index": False,
                "official_online_api_used": False,
            }
        )

        def index_factory() -> Any:
            return context_index_module.ContextIndex(
                linkage_method=args.linkage_method,
                use_gpu=False,
                alpha=args.alpha,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
            )

        reordered_contexts, plans = build_static_refit_causal_orderings(
            records, index_factory
        )
        output_records = materialize_contextpilot_causal_workload(
            records,
            reordered_contexts,
            plans,
            mode=args.mode,
            provenance=provenance,
        )
        ordering = "contextpilot_static_refit_causal"
    else:
        assert live_index_module is not None
        provenance.update(
            {
                "api": "contextpilot.server.live_index.ContextPilot.reorder",
                "persistent_index": True,
                "official_online_api_used": True,
            }
        )
        online_planner = live_index_module.ContextPilot(
            alpha=args.alpha,
            use_gpu=False,
            linkage_method=args.linkage_method,
            batch_size=args.batch_size,
        )
        reordered_contexts, plans = build_online_incremental_orderings(
            records,
            online_planner,
            conversation_prefix=args.conversation_prefix,
        )
        output_records = materialize_contextpilot_causal_workload(
            records,
            reordered_contexts,
            plans,
            mode=args.mode,
            provenance=provenance,
        )
        ordering = "contextpilot_online_incremental"
        index_stats = getattr(online_planner, "live_stats", None)

    count = write_jsonl(args.output, output_records)
    summary = {
        "format_version": 2,
        "requests": count,
        "input": args.input.as_posix(),
        "input_records": input_record_count,
        "limit": args.limit,
        "output": args.output.as_posix(),
        "input_sha256": file_sha256(args.input),
        "output_sha256": file_sha256(args.output),
        "ordering": ordering,
        "mode": args.mode,
        "information_regime": (
            "offline_transductive" if args.mode in OFFLINE_MODES else "causal"
        ),
        "offline_transductive": args.mode in OFFLINE_MODES,
        "scheduler_enabled": args.mode == "intra_schedule",
        "request_order_changed": request_order_changed,
        "official_online_api_used": args.mode == "online_incremental",
        "full_contextpilot_system": False,
        "annotations_enabled": False,
        "eviction_feedback_enabled": False,
        "unique_tools": len(tool_ids),
        "group_count": group_count,
        "planning": planning_summary(plans, count),
        "index_stats": _json_scalar(index_stats),
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
