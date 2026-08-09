#!/usr/bin/env python3
"""Audit the git-tracked ``frequency_online`` proposal and its comparisons.

The GPU executor retained raw replay files in a server-only archive.  This
audit therefore checks only what the compact repository package can prove:
policy metadata, aggregate result shape, cross-file arithmetic, the comparison
against the accepted dual-model summaries, and the disclosed cache-capacity
difference.  It deliberately does not certify raw counter cleanliness.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREQUENCY_ROOT = (
    PROJECT_ROOT / "reports" / "frequency-online" / "20260809-200701"
)
DEFAULT_DUAL_ROOT = (
    PROJECT_ROOT / "reports" / "contextpilot-dual-model" / "20260809-004603"
)

MODELS = {
    "Qwen/Qwen3-4B": "qwen3-4b",
    "Qwen/Qwen3-0.6B": "qwen3-0.6b",
}
WORKLOADS = (
    "bfcl-padded64",
    "toolret-padded64",
    "toolret-bm25-k4",
    "toolret-bm25-k16",
    "toolret-bm25-k64",
    "toolret-bm25-k128",
)
EXPECTED_TOOLTRIE_LOSSES = {
    ("Qwen/Qwen3-4B", "toolret-bm25-k4"),
    ("Qwen/Qwen3-0.6B", "toolret-bm25-k4"),
}
EXPECTED_FREQUENCY_CAPACITY = {
    "Qwen/Qwen3-4B": 101_120,
    "Qwen/Qwen3-0.6B": 190_896,
}
EXPECTED_DUAL_CAPACITY = {
    "Qwen/Qwen3-4B": 96_832,
    "Qwen/Qwen3-0.6B": 188_912,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cached_ratio(summary: dict[str, Any], condition: str) -> float:
    return float(
        summary["conditions"][condition]["measurements"]["cached_ratio"]["mean"]
    )


def audit(
    frequency_root: Path = DEFAULT_FREQUENCY_ROOT,
    dual_root: Path = DEFAULT_DUAL_ROOT,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    reuse = load_json(frequency_root / "frequency-online-reuse.json")
    add(
        "policy_metadata",
        reuse.get("policy") == "frequency_online"
        and reuse.get("information_regime") == "causal"
        and reuse.get("adapts_online") is True
        and reuse.get("training_input") is None
        and set(reuse.get("models") or {}) == set(MODELS),
    )

    base_commit = (frequency_root / "fyp-base-commit.txt").read_text(
        encoding="utf-8"
    ).strip()
    add(
        "declared_base_commit",
        base_commit == "15285704e73af680c0125ea4bfeb0b54a14f278e",
        base_commit,
    )

    workload_summary_ok = True
    workload_states: dict[str, Any] = {}
    for workload in WORKLOADS:
        summary = load_json(
            frequency_root / f"{workload}-frequency_online-summary.json"
        )
        state = summary.get("final_state") or {}
        workload_summary_ok = workload_summary_ok and (
            summary.get("policy") == "frequency_online"
            and summary.get("information_regime") == "causal"
            and summary.get("adapts_online") is True
            and summary.get("training_input") is None
            and summary.get("requests") == 200
            and state.get("requests_observed") == 200
            and isinstance(state.get("tools_seen"), int)
            and state["tools_seen"] > 0
        )
        workload_states[workload] = state
    add("workload_builder_summaries", workload_summary_ok, workload_states)

    comparisons: list[dict[str, Any]] = []
    control_differences: list[dict[str, Any]] = []
    result_shape_ok = True
    for model, slug in MODELS.items():
        model_payload = reuse["models"][model]
        frequency_cells = model_payload["conditions"].get("frequency_online") or {}
        result_shape_ok = result_shape_ok and set(frequency_cells) == set(WORKLOADS)

        for workload in WORKLOADS:
            cell = frequency_cells[workload]
            result_shape_ok = result_shape_ok and (
                cell.get("trials") == 3
                and cell.get("spread") == 0.0
                and isinstance(cell.get("cached_ratio_mean"), (int, float))
                and 0.0 <= cell["cached_ratio_mean"] <= 1.0
            )
            dual_summary = load_json(
                dual_root / slug / "summaries" / f"{workload}-summary.json"
            )
            frequency_value = float(cell["cached_ratio_mean"])
            tooltrie_value = cached_ratio(dual_summary, "tooltrie_v0")
            cp_online_value = cached_ratio(
                dual_summary, "contextpilot-online_incremental"
            )
            cp_static_value = cached_ratio(
                dual_summary, "contextpilot-static_refit_causal"
            )
            comparisons.append(
                {
                    "model": model,
                    "workload": workload,
                    "frequency_online": frequency_value,
                    "tooltrie_v0": tooltrie_value,
                    "frequency_minus_tooltrie_pp": round(
                        (frequency_value - tooltrie_value) * 100, 6
                    ),
                    "frequency_exceeds_tooltrie": frequency_value > tooltrie_value,
                    "contextpilot_online": cp_online_value,
                    "contextpilot_static_refit": cp_static_value,
                }
            )

        controls = model_payload["conditions"].get(
            "contextpilot-online_incremental"
        ) or {}
        result_shape_ok = result_shape_ok and set(controls) == {
            "toolret-bm25-k16",
            "toolret-bm25-k128",
        }
        for workload, cell in controls.items():
            dual_summary = load_json(
                dual_root / slug / "summaries" / f"{workload}-summary.json"
            )
            dual_value = cached_ratio(
                dual_summary, "contextpilot-online_incremental"
            )
            observed = float(cell["cached_ratio_mean"])
            result_shape_ok = result_shape_ok and (
                cell.get("trials") == 3 and cell.get("spread") == 0.0
            )
            control_differences.append(
                {
                    "model": model,
                    "workload": workload,
                    "same_session": observed,
                    "accepted_dual_model": dual_value,
                    "absolute_difference": abs(observed - dual_value),
                }
            )
    add(
        "aggregate_result_shape",
        result_shape_ok and len(comparisons) == 12 and len(control_differences) == 4,
        {"comparison_cells": len(comparisons), "control_cells": len(control_differences)},
    )

    losses = {
        (row["model"], row["workload"])
        for row in comparisons
        if not row["frequency_exceeds_tooltrie"]
    }
    wins = len(comparisons) - len(losses)
    add(
        "frequency_online_vs_tooltrie_count",
        wins == 10 and losses == EXPECTED_TOOLTRIE_LOSSES,
        {"wins": wins, "cells": len(comparisons), "losses": sorted(losses)},
    )
    add(
        "same_session_contextpilot_controls_reproduce",
        # The accepted dual-model summaries round ratios to six decimals.
        all(row["absolute_difference"] <= 1e-4 for row in control_differences),
        control_differences,
    )

    capacity_differences: list[dict[str, Any]] = []
    capacity_ok = True
    for model, slug in MODELS.items():
        frequency_config_name = "vllm-cache-config-4b.json"
        if model.endswith("0.6B"):
            frequency_config_name = "vllm-cache-config-0.6b.json"
        frequency_config = load_json(frequency_root / frequency_config_name)
        dual_config = load_json(dual_root / slug / "server-cache-config.json")
        frequency_capacity = int(frequency_config["capacity_tokens"])
        dual_capacity = int(dual_config["capacity_tokens"])
        capacity_ok = capacity_ok and (
            frequency_capacity == EXPECTED_FREQUENCY_CAPACITY[model]
            and dual_capacity == EXPECTED_DUAL_CAPACITY[model]
            and frequency_capacity != dual_capacity
        )
        capacity_differences.append(
            {
                "model": model,
                "frequency_online_run": frequency_capacity,
                "accepted_dual_model_run": dual_capacity,
                "difference_tokens": frequency_capacity - dual_capacity,
            }
        )
    add("cross_run_capacity_difference_recorded", capacity_ok, capacity_differences)

    hash_record = (frequency_root / "raw-archive.sha256").read_text(
        encoding="utf-8"
    ).strip()
    add(
        "raw_archive_hash_recorded",
        re.fullmatch(r"[0-9a-f]{64}  /home/taghan/[^\s]+\.tar\.gz", hash_record)
        is not None,
        hash_record,
    )

    all_checks_passed = all(check["passed"] for check in checks)
    return {
        "format_version": 1,
        "status": (
            "compact_aggregates_consistent_raw_replays_not_locally_verified"
            if all_checks_passed
            else "failed"
        ),
        "all_checks_passed": all_checks_passed,
        "checks_passed": sum(int(check["passed"]) for check in checks),
        "checks_total": len(checks),
        "raw_replay_counter_cleanliness_verified_by_this_audit": False,
        "quality_measured": False,
        "comparisons": comparisons,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frequency-root", type=Path, default=DEFAULT_FREQUENCY_ROOT
    )
    parser.add_argument("--dual-root", type=Path, default=DEFAULT_DUAL_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit(args.frequency_root.resolve(), args.dual_root.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
