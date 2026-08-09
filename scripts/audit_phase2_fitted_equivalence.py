#!/usr/bin/env python3
"""Reconstruct exact Phase-2 fitted-policy ordering equivalence.

The historical systems summaries showed nearly identical cache ratios but did
not prove identical behavior.  This CPU-only audit repeats the declared
workload construction and hashes the emitted tool-ID sequences for all five
fitted policies on both 200-request padded systems workloads.
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

from tatm.analysis import deduplicated_existing_ids, load_processed
from tatm.baselines import FittedOrderingPlanner
from tatm.io import write_json
from tatm.prompting import build_menu


POLICIES = (
    "frequency_fitted",
    "schema_cost_fitted",
    "fp_tree_conditional",
    "conditional_pair",
    "conditional_pair_triple",
)
PARTITIONS = {
    "bfcl": "exposed_menu",
    "toolret": "gold_relevance",
}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_digest(sequences: list[tuple[str, ...]]) -> str:
    payload = json.dumps(
        sequences, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def equivalence_classes(
    sequences: dict[str, list[tuple[str, ...]]],
) -> list[list[str]]:
    remaining = list(sequences)
    classes: list[list[str]] = []
    while remaining:
        reference = remaining.pop(0)
        group = [reference]
        for candidate in list(remaining):
            if sequences[candidate] == sequences[reference]:
                group.append(candidate)
                remaining.remove(candidate)
        classes.append(group)
    return classes


def audit(processed_dir: Path) -> dict:
    tools, tasks = load_processed(processed_dir)
    partition_results = {}
    for partition, evidence_type in PARTITIONS.items():
        partition_tasks = [
            task for task in tasks if task.evidence_type == evidence_type
        ]
        evaluation_tasks = partition_tasks[:200]
        training_tasks = partition_tasks[200:]
        if len(evaluation_tasks) != 200 or not training_tasks:
            raise ValueError(f"{partition}: declared slices are unavailable")

        distractor_pool = sorted(
            tool_id
            for tool_id, tool in tools.items()
            if 0 < tool.schema_tokens <= 300
        )
        evaluation_menus = [
            build_menu(
                deduplicated_existing_ids(task, tools),
                distractor_pool,
                64,
                seed=42,
            )
            for task in evaluation_tasks
        ]
        training_transactions = [
            deduplicated_existing_ids(task, tools) for task in training_tasks
        ]
        sequences = {}
        states = {}
        for policy in POLICIES:
            planner = FittedOrderingPlanner(
                tools, training_transactions, policy=policy
            )
            sequences[policy] = [
                planner.plan(menu).ordered_ids for menu in evaluation_menus
            ]
            states[policy] = planner.snapshot()

        classes = equivalence_classes(sequences)
        reference = sequences["frequency_fitted"]
        partition_results[partition] = {
            "evaluation_requests": len(evaluation_tasks),
            "training_requests": len(training_tasks),
            "menu_size": 64,
            "random_seed": 42,
            "equivalence_classes": classes,
            "distinct_emitted_orderings": len(classes),
            "policies": {
                policy: {
                    "sequence_sha256": sequence_digest(rows),
                    "requests_different_from_frequency_fitted": sum(
                        left != right
                        for left, right in zip(rows, reference, strict=True)
                    ),
                    "state": states[policy],
                }
                for policy, rows in sequences.items()
            },
        }

    bfcl_expected = [list(POLICIES)]
    toolret_expected = [
        ["frequency_fitted", "conditional_pair", "conditional_pair_triple"],
        ["schema_cost_fitted"],
        ["fp_tree_conditional"],
    ]
    checks = {
        "bfcl_five_labels_emit_one_ordering": partition_results["bfcl"][
            "equivalence_classes"
        ]
        == bfcl_expected,
        "toolret_has_three_distinct_fitted_orderings": partition_results[
            "toolret"
        ]["equivalence_classes"]
        == toolret_expected,
    }
    all_checks_passed = all(checks.values())
    return {
        "format_version": 1,
        "experiment": "tooltrie-phase2-fitted-policy-equivalence",
        "generated_by": "scripts/audit_phase2_fitted_equivalence.py",
        "status": "accepted" if all_checks_passed else "failed",
        "all_checks_passed": all_checks_passed,
        "checks": checks,
        "inputs": {
            "tools_jsonl_sha256": file_digest(processed_dir / "tools.jsonl"),
            "tasks_jsonl_sha256": file_digest(processed_dir / "tasks.jsonl"),
            "canonical_tools": len(tools),
            "tasks": len(tasks),
        },
        "partitions": partition_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        result = audit(args.processed_dir.resolve())
    except (OSError, ValueError) as error:
        parser.error(str(error))
    write_json(args.output, result)
    print(json.dumps(result["checks"], indent=2, sort_keys=True))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
