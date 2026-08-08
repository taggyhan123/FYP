#!/usr/bin/env python3
"""Fail-closed audit for the Qwen3-4B/0.6B ContextPilot replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_contextpilot_dual_model import (
    CONDITIONS,
    EXPECTED_CONTEXTPILOT_COMMIT,
    EXPECTED_ORDERING,
    MODEL_REVISIONS,
    PROJECT_ROOT,
    QUALITY_CONDITION_ORDER,
    QUALITY_METRICS,
    QUALITY_PAIRS,
    SYSTEM_WORKLOADS,
    load_json,
    systems_plan,
    validate_quality_score,
    validate_replay,
    validate_shared_workloads,
)


MODEL_DIRS = {
    "Qwen/Qwen3-4B": "qwen3-4b",
    "Qwen/Qwen3-0.6B": "qwen3-0.6b",
}
CAPACITY_INDEPENDENT_CONDITIONS = (
    "original",
    "alphabetical",
    "contextpilot-static_refit_causal",
    "contextpilot-online_incremental",
)


def add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, details: Any = None
) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def request_projection(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    return [
        {
            "case_id": str(row.get("case_id") or row.get("task_id")),
            "tool_ids": list(row.get("tool_ids") or []),
            "prompt_tokens": (row.get("usage") or {}).get("prompt_tokens"),
        }
        for row in payload.get("results") or []
    ]


def cache_signature(config: dict[str, Any]) -> tuple[bool | None, int, int]:
    block_size = int(config.get("block_size") or 0)
    num_gpu_blocks = int(config.get("num_gpu_blocks") or 0)
    return (
        config.get("enable_prefix_caching"),
        block_size,
        block_size * num_gpu_blocks,
    )


def audit(root: Path, expected_fyp_commit: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    shared = root / "shared-workloads"
    declared_manifest = PROJECT_ROOT / "cluster" / "contextpilot-dual-model-manifest.json"
    copied_manifest = root / "protocol-manifest.json"
    add_check(
        checks,
        "protocol_manifest_preserved",
        copied_manifest.is_file()
        and copied_manifest.read_bytes() == declared_manifest.read_bytes(),
    )
    tokenizer_path = root / "tokenizer-compatibility.json"
    try:
        tokenizer = load_json(tokenizer_path)
        tokenizer_passed = (
            tokenizer.get("tokenizer_json_identical") is True
            and tokenizer.get("chat_template_identical") is True
            and tokenizer.get("compatible_for_shared_schema_token_accounting")
            is True
            and set(tokenizer.get("models") or {}) == set(MODEL_DIRS)
        )
    except (FileNotFoundError, json.JSONDecodeError):
        tokenizer_passed = False
    add_check(checks, "tokenizer_and_chat_template_compatibility", tokenizer_passed)
    try:
        validate_shared_workloads(shared)
        add_check(checks, "shared_workload_provenance", True)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        add_check(checks, "shared_workload_provenance", False, str(error))

    provenance_expectations = {
        "fyp-git-commit.txt": expected_fyp_commit,
        "contextpilot-git-commit.txt": EXPECTED_CONTEXTPILOT_COMMIT,
        "vllm-version.txt": "0.26.0",
    }
    for filename, expected in provenance_expectations.items():
        path = root / filename
        actual = path.read_text(encoding="utf-8").strip() if path.is_file() else None
        add_check(checks, f"provenance:{filename}", actual == expected, actual)
    add_check(
        checks,
        "provenance:gpu-environment.csv",
        (root / "gpu-environment.csv").is_file(),
    )

    model_payloads: dict[str, dict[str, Any]] = {}
    for model, directory_name in MODEL_DIRS.items():
        model_root = root / directory_name
        analysis_root = root / "analysis" / directory_name
        revision = MODEL_REVISIONS[model]
        model_checks_before = len(checks)
        provenance_path = model_root / "run-provenance.json"
        summary_path = model_root / "model-run-summary.json"
        config_path = model_root / "server-cache-config.json"
        try:
            provenance = load_json(provenance_path)
            summary = load_json(summary_path)
            config = load_json(config_path)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            add_check(checks, f"{directory_name}:core_files", False, str(error))
            continue

        add_check(
            checks,
            f"{directory_name}:provenance",
            provenance.get("model") == model
            and provenance.get("model_revision") == revision
            and summary.get("model") == model
            and summary.get("model_revision") == revision,
        )
        capacity = config.get("capacity_tokens")
        try:
            config_signature = cache_signature(config)
        except (TypeError, ValueError):
            config_signature = (None, 0, 0)
        add_check(
            checks,
            f"{directory_name}:cache_config",
            config_signature[0] is True
            and config_signature[1] == 16
            and isinstance(capacity, int)
            and capacity > 0,
            config,
        )

        warmup_path = model_root / "diagnostics" / "unmeasured-warmup.json"
        try:
            warmup = validate_replay(
                warmup_path,
                expected_model=model,
                expected_requests=1,
                expected_ordering="original",
            )
            warmup_passed = cache_signature(
                warmup.get("server_cache_config") or {}
            ) == config_signature
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            warmup_passed = False
        add_check(
            checks,
            f"{directory_name}:excluded_warmup",
            warmup_passed,
        )

        accepted_systems = 0
        for item in systems_plan():
            path = (
                model_root
                / "systems"
                / f"{item.stem}-{item.condition}-trial-{item.trial}.json"
            )
            try:
                replay = validate_replay(
                    path,
                    expected_model=model,
                    expected_requests=200,
                    expected_ordering=EXPECTED_ORDERING[item.condition],
                )
                if cache_signature(
                    replay.get("server_cache_config") or {}
                ) != config_signature:
                    raise ValueError("embedded server cache config changed")
                accepted_systems += 1
            except (
                FileNotFoundError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                add_check(checks, f"{directory_name}:{path.name}", False, str(error))
        add_check(
            checks,
            f"{directory_name}:systems_replay_count",
            accepted_systems == 90,
            accepted_systems,
        )

        summary_guards = 0
        for stem, _ in SYSTEM_WORKLOADS:
            path = model_root / "summaries" / f"{stem}-summary.json"
            try:
                payload = load_json(path)
                passed = (
                    set(payload.get("conditions") or {}) == set(CONDITIONS)
                    and payload.get("all_conditions_have_same_case_set") is True
                    and payload.get("all_conditions_have_same_request_sequence")
                    is True
                    and payload.get("all_conditions_have_same_selected_tool_sets")
                    is True
                    and all(
                        row.get("trials") == 3 and row.get("requests") == 200
                        for row in (payload.get("conditions") or {}).values()
                    )
                    and payload.get("conditions", {})
                    .get("original", {})
                    .get("execution_condition", {})
                    .get("role")
                    == "ordinary_text_prefill_fallback"
                    and all(
                        payload.get("conditions", {})
                        .get(condition, {})
                        .get("execution_condition", {})
                        .get("role")
                        == "ordering_candidate"
                        for condition in CONDITIONS
                        if condition != "original"
                    )
                )
            except (FileNotFoundError, json.JSONDecodeError):
                passed = False
            summary_guards += int(passed)
        add_check(
            checks,
            f"{directory_name}:systems_summary_guards",
            summary_guards == 6,
            summary_guards,
        )

        accepted_quality = 0
        accepted_scores = 0
        for condition in QUALITY_CONDITION_ORDER:
            replay = model_root / "quality" / f"quality-{condition}-replay.json"
            score = model_root / "quality" / f"quality-{condition}-score.json"
            try:
                quality_replay = validate_replay(
                    replay,
                    expected_model=model,
                    expected_requests=800,
                    expected_ordering=EXPECTED_ORDERING[condition],
                )
                if cache_signature(
                    quality_replay.get("server_cache_config") or {}
                ) != config_signature:
                    raise ValueError("embedded server cache config changed")
                accepted_quality += 1
            except (
                FileNotFoundError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                add_check(checks, f"{directory_name}:{replay.name}", False, str(error))
            try:
                validate_quality_score(score)
                accepted_scores += 1
            except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
                add_check(checks, f"{directory_name}:{score.name}", False, str(error))
        add_check(
            checks,
            f"{directory_name}:quality_replay_count",
            accepted_quality == 5,
            accepted_quality,
        )
        add_check(
            checks,
            f"{directory_name}:quality_score_count",
            accepted_scores == 5,
            accepted_scores,
        )
        compact_path = model_root / "summaries" / "quality-scores-compact.json"
        try:
            compact = load_json(compact_path)
            compact_passed = (
                compact.get("model") == model
                and compact.get("n_per_condition") == 800
                and set(compact.get("conditions") or {}) == set(CONDITIONS)
                and all(
                    row.get("overall", {}).get("tasks_scored") == 800
                    and row.get("overall", {}).get("relevance_tasks") == 640
                    and row.get("overall", {}).get("irrelevance_tasks") == 160
                    for row in (compact.get("conditions") or {}).values()
                )
            )
        except (FileNotFoundError, json.JSONDecodeError):
            compact_passed = False
        add_check(
            checks,
            f"{directory_name}:quality_compact_summary",
            compact_passed,
        )

        accepted_comparisons = 0
        for baseline, candidate in QUALITY_PAIRS:
            for metric in QUALITY_METRICS:
                path = (
                    analysis_root
                    / "comparisons"
                    / f"quality-{candidate}-vs-{baseline}-{metric}.json"
                )
                try:
                    payload = load_json(path)
                    paired_cases = 160 if metric == "no_tool_correct" else 640
                    passed = (
                        payload.get("sequence_state_dependent") is True
                        and payload.get(
                            "cluster_bootstrap_generalizes_across_request_sequences"
                        )
                        is False
                        and payload.get("mcnemar_independence_assumption_met") is False
                        and payload.get("paired_cases") == paired_cases
                    )
                except (FileNotFoundError, json.JSONDecodeError):
                    passed = False
                accepted_comparisons += int(passed)
        add_check(
            checks,
            f"{directory_name}:quality_comparison_count",
            accepted_comparisons == 27,
            accepted_comparisons,
        )
        analysis_summary_path = analysis_root / "analysis-summary.json"
        try:
            analysis_summary = load_json(analysis_summary_path)
            analysis_summary_passed = (
                analysis_summary.get("status") == "accepted"
                and analysis_summary.get("model") == model
                and analysis_summary.get("model_revision") == revision
                and analysis_summary.get("quality_comparison_files") == 27
                and analysis_summary.get("bootstrap_samples") == 50000
                and analysis_summary.get("bootstrap_seed") == 42
                and analysis_summary.get("sequence_state_dependent") is True
            )
        except (FileNotFoundError, json.JSONDecodeError):
            analysis_summary_passed = False
        add_check(
            checks,
            f"{directory_name}:analysis_summary",
            analysis_summary_passed,
        )

        expected_summary = {
            "status": "gpu_complete_pending_cpu_analysis",
            "accepted_systems_replays": 90,
            "accepted_quality_replays": 5,
            "quality_score_files": 5,
            "diagnostic_warmups_excluded": 1,
            "all_replays_reset_and_counter_clean": True,
        }
        add_check(
            checks,
            f"{directory_name}:model_run_summary",
            all(summary.get(key) == value for key, value in expected_summary.items()),
            summary,
        )
        add_check(
            checks,
            f"{directory_name}:all_model_checks_reached",
            len(checks) > model_checks_before,
        )
        model_payloads[model] = {
            "root": model_root,
            "capacity_tokens": capacity,
        }

    if set(model_payloads) == set(MODEL_DIRS):
        mismatches: list[str] = []
        primary_root = model_payloads["Qwen/Qwen3-4B"]["root"]
        replica_root = model_payloads["Qwen/Qwen3-0.6B"]["root"]
        for stem, _ in SYSTEM_WORKLOADS:
            for condition in CAPACITY_INDEPENDENT_CONDITIONS:
                relative = (
                    Path("systems") / f"{stem}-{condition}-trial-1.json"
                )
                try:
                    same = request_projection(
                        primary_root / relative
                    ) == request_projection(replica_root / relative)
                except (FileNotFoundError, json.JSONDecodeError):
                    same = False
                if not same:
                    mismatches.append(f"{stem}:{condition}")
        for condition in CAPACITY_INDEPENDENT_CONDITIONS:
            relative = Path("quality") / f"quality-{condition}-replay.json"
            try:
                same = request_projection(primary_root / relative) == request_projection(
                    replica_root / relative
                )
            except (FileNotFoundError, json.JSONDecodeError):
                same = False
            if not same:
                mismatches.append(f"quality:{condition}")
        add_check(
            checks,
            "cross_model_capacity_independent_workloads",
            not mismatches,
            mismatches,
        )

        capacities = {
            model: payload["capacity_tokens"] for model, payload in model_payloads.items()
        }
        add_check(
            checks,
            "native_capacities_recorded_separately",
            all(isinstance(value, int) and value > 0 for value in capacities.values()),
            capacities,
        )
    else:
        add_check(
            checks,
            "cross_model_capacity_independent_workloads",
            False,
            "one or both model arms are missing",
        )

    passed = all(item["passed"] for item in checks)
    return {
        "format_version": 1,
        "status": "accepted" if passed else "quarantined",
        "expected_fyp_commit": expected_fyp_commit,
        "accepted_gpu_replays": 190 if passed else None,
        "all_checks_passed": passed,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-fyp-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"Refusing to overwrite audit output: {args.output}")
    result = audit(args.root.resolve(), args.expected_fyp_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
