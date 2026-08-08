#!/usr/bin/env python3
"""Audit the tracked compact ContextPilot confirmation evidence.

This cannot replace validation of the raw GPU archive. It checks internal
consistency, provenance, declared feature scope, score/comparison agreement,
and whether known acceptance cells or inference metadata are still missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIRMATION = (
    PROJECT_ROOT / "reports" / "contextpilot-confirmation" / "20260807-222212"
)
DEFAULT_QUALITY_4B = (
    PROJECT_ROOT / "reports" / "contextpilot-quality-4b" / "20260808-105833"
)
EXPECTED_CONTEXT_COMMIT = "1fa0a143fdeda344585666648ab2b30cb7fea77f"
EXPECTED_CONDITIONS = {
    "original",
    "alphabetical",
    "tooltrie_v0",
    "contextpilot-online_incremental",
}
SYSTEM_STEMS = (
    "bfcl-padded64",
    "toolret-padded64",
    "toolret-bm25-k4",
    "toolret-bm25-k16",
    "toolret-bm25-k64",
    "toolret-bm25-k128",
)
METRICS = {
    "name_correct": "function_name_accuracy",
    "full_correct": "full_accuracy",
    "no_tool_correct": "no_tool_accuracy",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check(
    checks: list[dict[str, Any]], name: str, passed: bool, details: Any = None
) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def _quality_value(payload: dict[str, Any], condition: str, metric: str) -> float:
    return float(payload["conditions"][condition]["overall"][METRICS[metric]])


def _comparison_paths(root: Path) -> list[Path]:
    return sorted(root.glob("quality-*-vs-*.json"))


def _condition_from_path(path_text: str) -> str:
    name = Path(path_text).name
    marker = "quality-"
    suffixes = (
        "-score-relevance.json",
        "-score-irrelevance.json",
        "-score.json",
    )
    value = name[len(marker) :] if name.startswith(marker) else name
    for suffix in suffixes:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def audit(confirmation: Path, quality_4b: Path) -> dict[str, Any]:
    measurement_checks: list[dict[str, Any]] = []
    metadata_checks: list[dict[str, Any]] = []

    context_commit = (confirmation / "contextpilot-git-commit.txt").read_text(
        encoding="utf-8"
    ).strip()
    check(
        measurement_checks,
        "pinned_contextpilot_commit",
        context_commit == EXPECTED_CONTEXT_COMMIT,
        context_commit,
    )

    for config_name in ("vllm-cache-config.json", "vllm-cache-config-8b.json"):
        config = load(confirmation / config_name)
        check(
            measurement_checks,
            f"{config_name}:prefix_cache_enabled",
            config.get("enable_prefix_caching") is True,
        )
        check(
            measurement_checks,
            f"{config_name}:live_block_size",
            int(config.get("block_size", 0)) == 16,
            config.get("block_size"),
        )
        check(
            measurement_checks,
            f"{config_name}:positive_capacity",
            int(config.get("capacity_tokens", 0)) > 0,
            config.get("capacity_tokens"),
        )

    builder_paths = sorted(
        confirmation.glob("*-contextpilot-online_incremental-summary.json")
    )
    check(
        measurement_checks,
        "persistent_builder_summary_count",
        len(builder_paths) == 7,
        len(builder_paths),
    )
    for path in builder_paths:
        payload = load(path)
        reference = payload.get("reference") or {}
        check(
            measurement_checks,
            f"{path.name}:provenance",
            payload.get("mode") == "online_incremental"
            and payload.get("information_regime") == "causal"
            and payload.get("official_online_api_used") is True
            and payload.get("full_contextpilot_system") is False
            and payload.get("annotations_enabled") is False
            and payload.get("eviction_feedback_enabled") is False
            and payload.get("request_order_changed") is False
            and reference.get("commit") == EXPECTED_CONTEXT_COMMIT
            and reference.get("alpha") == 0.001
            and reference.get("persistent_index") is True,
        )
        hashes = (payload.get("input_sha256"), payload.get("output_sha256"))
        check(
            measurement_checks,
            f"{path.name}:sha256_fields",
            all(
                isinstance(value, str)
                and len(value) == 64
                and all(char in "0123456789abcdef" for char in value)
                for value in hashes
            ),
            hashes,
        )

    for stem in SYSTEM_STEMS:
        path = confirmation / f"{stem}-summary.json"
        payload = load(path)
        check(
            measurement_checks,
            f"{stem}:equivalence_guards",
            payload.get("all_conditions_have_same_case_set") is True
            and payload.get("all_conditions_have_same_request_sequence") is True
            and payload.get("all_conditions_have_same_selected_tool_sets") is True,
        )
        conditions = payload.get("conditions") or {}
        check(
            measurement_checks,
            f"{stem}:condition_set",
            set(conditions) == EXPECTED_CONDITIONS,
            sorted(conditions),
        )
        for condition, row in conditions.items():
            measurements = row.get("measurements") or {}
            cached = float((measurements.get("cached_prompt_tokens") or {}).get("mean", -1))
            computed = float(
                (measurements.get("computed_prompt_tokens") or {}).get("mean", -1)
            )
            ratio = float((measurements.get("cached_ratio") or {}).get("mean", -1))
            denominator = cached + computed
            check(
                measurement_checks,
                f"{stem}:{condition}:trial_shape",
                row.get("trials") == 3
                and row.get("requests") == 200
                and all(
                    (measurements.get(metric) or {}).get("n") == 3
                    for metric in (
                        "cached_prompt_tokens",
                        "computed_prompt_tokens",
                        "cached_ratio",
                        "ttft_seconds",
                    )
                ),
            )
            check(
                measurement_checks,
                f"{stem}:{condition}:counter_identity",
                denominator > 0
                and abs(denominator - float(row.get("prompt_tokens", -1))) < 0.5
                and abs(ratio - cached / denominator) < 1e-6,
                {
                    "cached": cached,
                    "computed": computed,
                    "prompt_tokens": row.get("prompt_tokens"),
                    "ratio": ratio,
                },
            )

    quality_payloads = {
        "8b": load(confirmation / "quality-scores-compact.json"),
        "4b": load(quality_4b / "quality-scores-compact.json"),
    }
    for label, payload in quality_payloads.items():
        check(
            measurement_checks,
            f"quality_{label}:shape",
            payload.get("n_per_condition") == 800
            and set(payload.get("conditions") or {})
            == {"alphabetical", "tooltrie_v0", "contextpilot-online_incremental"}
            and all(
                row.get("overall", {}).get("relevance_tasks") == 640
                and row.get("overall", {}).get("irrelevance_tasks") == 160
                for row in (payload.get("conditions") or {}).values()
            ),
        )

    comparison_groups = {
        "8b": (confirmation, quality_payloads["8b"]),
        "4b": (quality_4b, quality_payloads["4b"]),
    }
    files_needing_metadata: list[str] = []
    for label, (root, quality) in comparison_groups.items():
        paths = _comparison_paths(root)
        check(
            measurement_checks,
            f"quality_{label}:comparison_count",
            len(paths) == 9,
            len(paths),
        )
        for path in paths:
            comparison = load(path)
            metric = str(comparison["metric"])
            baseline = _condition_from_path(comparison["baseline_files"][0])
            candidate = _condition_from_path(comparison["candidate_files"][0])
            baseline_value = _quality_value(quality, baseline, metric)
            candidate_value = _quality_value(quality, candidate, metric)
            check(
                measurement_checks,
                f"quality_{label}:{path.name}:score_agreement",
                abs(float(comparison["baseline_accuracy"]) - baseline_value) < 1e-4
                and abs(float(comparison["candidate_accuracy"]) - candidate_value)
                < 1e-4
                and abs(
                    float(comparison["difference_percentage_points"])
                    - 100 * (candidate_value - baseline_value)
                )
                < 0.02
                and comparison.get("paired_cases")
                == (160 if metric == "no_tool_correct" else 640),
            )
            metadata_clean = (
                comparison.get("sequence_state_dependent") is True
                and comparison.get(
                    "cluster_bootstrap_generalizes_across_request_sequences"
                )
                is False
                and comparison.get("mcnemar_independence_assumption_met") is False
            )
            check(
                metadata_checks,
                f"quality_{label}:{path.name}:sequence_state_metadata",
                metadata_clean,
            )
            if not metadata_clean:
                try:
                    rendered_path = path.relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    rendered_path = path.as_posix()
                files_needing_metadata.append(rendered_path)

    raw_hashes = {
        "8b_confirmation": (confirmation / "raw-archive.sha256").read_text(
            encoding="utf-8"
        ).strip(),
        "4b_addendum": (quality_4b / "raw-archive.sha256").read_text(
            encoding="utf-8"
        ).strip(),
    }
    check(
        measurement_checks,
        "raw_archive_hashes_recorded",
        all(len(value.split()[0]) == 64 for value in raw_hashes.values()),
        raw_hashes,
    )

    measurement_passed = all(item["passed"] for item in measurement_checks)
    metadata_passed = all(item["passed"] for item in metadata_checks)
    return {
        "format_version": 1,
        "status": (
            "complete"
            if measurement_passed and metadata_passed
            else "partial_requires_cluster_followup"
        ),
        "tracked_measurement_artifact_checks_passed": measurement_passed,
        "tracked_analysis_metadata_checks_passed": metadata_passed,
        "static_refit_causal_gpu_arm_present": False,
        "accepted_persistent_gpu_replays_declared": 75,
        "missing_static_refit_gpu_replays": 19,
        "raw_archives_verified_from_this_machine": False,
        "raw_archive_hashes_recorded": raw_hashes,
        "files_needing_sequence_state_regeneration": files_needing_metadata,
        "measurement_checks": measurement_checks,
        "analysis_metadata_checks": metadata_checks,
        "audit_scope_note": (
            "Compact tracked artifacts only. Raw replay rows and archives remain "
            "on the NUS server and require the resume protocol."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-dir", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--quality-4b-dir", type=Path, default=DEFAULT_QUALITY_4B)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.confirmation_dir, args.quality_4b_dir)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["tracked_measurement_artifact_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
