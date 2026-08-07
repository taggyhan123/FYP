#!/usr/bin/env python3
"""Reconstruct and hash the controlled-pressure ordering sequences.

The GPU pressure outputs intentionally omit tool IDs from per-request records.
This audit reruns only the deterministic workload construction and ordering
code—no model or GPU—to establish whether nominal policies emitted the same
tool sequences under the pinned manifest parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.analysis import (
    deduplicated_existing_ids,
    load_processed,
    replay_workloads,
)
from tatm.io import write_json
from tatm.pressure_summary import PRESSURE_ORDERINGS, PRESSURE_REGIMES
from tatm.prompting import build_menu, order_tool_ids


def sequence_digest(sequences: list[tuple[str, ...]]) -> str:
    payload = json.dumps(
        sequences,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def equivalence_classes(
    sequences: dict[str, list[tuple[str, ...]]],
) -> list[list[str]]:
    remaining = list(sequences)
    groups: list[list[str]] = []
    while remaining:
        first = remaining.pop(0)
        group = [first]
        for candidate in list(remaining):
            if sequences[candidate] == sequences[first]:
                group.append(candidate)
                remaining.remove(candidate)
        groups.append(group)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact ordering equivalence for the pressure manifest."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-commit",
        default="caf8ab576eb15d95265a53ec76c772c8af6c7929",
    )
    args = parser.parse_args()

    tools, tasks = load_processed(args.processed_dir)
    partition_tasks = [task for task in tasks if task.evidence_type == "exposed_menu"]
    selected_tasks = partition_tasks[:200]
    if len(selected_tasks) != 200:
        raise SystemExit("The declared first-200 BFCL slice is unavailable")
    selected_ids = {task.task_id for task in selected_tasks}
    support_tasks = [
        task for task in partition_tasks if task.task_id not in selected_ids
    ]
    support: Counter[str] = Counter()
    for task in support_tasks:
        support.update(deduplicated_existing_ids(task, tools))
    distractor_pool = sorted(
        tool_id
        for tool_id, tool in tools.items()
        if 0 < tool.schema_tokens <= 300
    )
    replays = replay_workloads(selected_tasks, tools, seed=2026)

    regime_results = {}
    class_patterns = []
    for regime in PRESSURE_REGIMES:
        replay_tasks = replays[regime]
        menus = [
            build_menu(
                deduplicated_existing_ids(task, tools),
                distractor_pool,
                64,
                seed=42,
            )
            for task in replay_tasks
        ]
        sequences = {
            ordering: [
                order_tool_ids(
                    menu,
                    tools,
                    support,
                    ordering,
                    random_seed=42,
                )
                for menu in menus
            ]
            for ordering in PRESSURE_ORDERINGS
        }
        groups = equivalence_classes(sequences)
        class_patterns.append(groups)
        reference = sequences["frequency"]
        regime_results[regime] = {
            "requests": len(replay_tasks),
            "task_sequence_sha256": sequence_digest(
                [(task.task_id,) for task in replay_tasks]
            ),
            "equivalence_classes": groups,
            "orderings": {
                ordering: {
                    "sequence_sha256": sequence_digest(rows),
                    "requests_different_from_frequency": sum(
                        left != right for left, right in zip(rows, reference)
                    ),
                }
                for ordering, rows in sequences.items()
            },
        }

    output = {
        "format_version": 1,
        "experiment": "initial-brief-controlled-cache-pressure-rerun",
        "audit": "deterministic-ordering-sequence-equivalence",
        "generated_by": "scripts/audit_pressure_ordering_equivalence.py",
        "source_commit": args.source_commit,
        "inputs": {
            "tools_jsonl_sha256": file_digest(args.processed_dir / "tools.jsonl"),
            "tasks_jsonl_sha256": file_digest(args.processed_dir / "tasks.jsonl"),
            "canonical_tools": len(tools),
            "tasks": len(tasks),
        },
        "parameters": {
            "partition": "bfcl",
            "offset": 0,
            "requests": 200,
            "menu_size": 64,
            "max_schema_tokens": 300,
            "support_mode": "disjoint",
            "support_tasks": len(support_tasks),
            "evaluation_overlap_tasks": 0,
            "random_seed": 42,
            "replay_seed": 2026,
        },
        "sequence_hash": "sha256 over compact JSON array of ordered tool-ID arrays",
        "same_equivalence_classes_in_every_regime": all(
            pattern == class_patterns[0] for pattern in class_patterns
        ),
        "distinct_orderings_per_regime": len(class_patterns[0]),
        "regimes": regime_results,
    }
    write_json(args.output, output)
    print(f"Distinct orderings per regime: {output['distinct_orderings_per_regime']}")
    print(f"Equivalence classes: {class_patterns[0]}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
