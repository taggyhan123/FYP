#!/usr/bin/env python3
"""Audit tracked static-refit and SGLang-counter handover artifacts.

This audit validates compact summaries only.  It checks the 18 static-refit
systems replays, the 8B aggregate quality cell and paired comparisons, and the
72 historical SGLang aggregate-counter decisions.  Raw per-request files and
the two tar archives remain server-only and are outside this audit's scope.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATIC_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "contextpilot-static-refit-resume"
    / "20260808-234909"
)
DEFAULT_CONFIRMATION_ROOT = (
    PROJECT_ROOT / "reports" / "contextpilot-confirmation" / "20260807-222212"
)

STEMS = (
    "bfcl-padded64",
    "toolret-padded64",
    "toolret-bm25-k4",
    "toolret-bm25-k16",
    "toolret-bm25-k64",
    "toolret-bm25-k128",
)
CONDITIONS = {
    "original",
    "alphabetical",
    "tooltrie_v0",
    "contextpilot-online_incremental",
    "contextpilot-static_refit_causal",
}
BASELINES = ("alphabetical", "tooltrie_v0")
METRICS = {
    "name_correct": 640,
    "full_correct": 640,
    "no_tool_correct": 160,
}
NUMERIC_COMPARISON_FIELDS = (
    "metric",
    "paired_cases",
    "baseline_accuracy",
    "candidate_accuracy",
    "difference_percentage_points",
    "cluster_bootstrap_95_ci_percentage_points",
    "discordant_pairs",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(
    static_root: Path = DEFAULT_STATIC_ROOT,
    confirmation_root: Path = DEFAULT_CONFIRMATION_ROOT,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    json_files = sorted(static_root.glob("*.json"))
    parse_failures: list[str] = []
    for path in json_files:
        try:
            load_json(path)
        except (OSError, json.JSONDecodeError) as error:
            parse_failures.append(f"{path.name}: {error}")
    add(
        "all_tracked_json_parses",
        len(json_files) == 24 and not parse_failures,
        {"files": len(json_files), "failures": parse_failures},
    )

    static_system_replays = 0
    systems_ok = True
    systems_rows: list[dict[str, Any]] = []
    for stem in STEMS:
        builder = load_json(
            static_root / f"{stem}-contextpilot-static_refit_causal-summary.json"
        )
        reference = builder.get("reference") or {}
        systems_ok = systems_ok and (
            builder.get("mode") == "static_refit_causal"
            and builder.get("ordering") == "contextpilot_static_refit_causal"
            and builder.get("information_regime") == "causal"
            and builder.get("offline_transductive") is False
            and builder.get("official_online_api_used") is False
            and builder.get("full_contextpilot_system") is False
            and builder.get("annotations_enabled") is False
            and builder.get("eviction_feedback_enabled") is False
            and builder.get("request_order_changed") is False
            and builder.get("requests") == 200
            and reference.get("commit")
            == "1fa0a143fdeda344585666648ab2b30cb7fea77f"
            and reference.get("alpha") == 0.001
        )

        combined = load_json(static_root / f"{stem}-combined-summary.json")
        conditions = combined.get("conditions") or {}
        summary_ok = (
            combined.get("engine") == "vllm"
            and set(conditions) == CONDITIONS
            and combined.get("all_conditions_have_same_case_set") is True
            and combined.get("all_conditions_have_same_request_sequence") is True
            and combined.get("all_conditions_have_same_selected_tool_sets") is True
        )
        for condition, row in conditions.items():
            measurements = row.get("measurements") or {}
            cached = float(measurements["cached_prompt_tokens"]["mean"])
            computed = float(measurements["computed_prompt_tokens"]["mean"])
            ratio = float(measurements["cached_ratio"]["mean"])
            prompt_tokens = int(row["prompt_tokens"])
            summary_ok = summary_ok and (
                row.get("requests") == 200
                and row.get("trials") == 3
                and math.isclose(cached + computed, prompt_tokens, abs_tol=1e-6)
                and math.isclose(ratio, cached / prompt_tokens, abs_tol=1e-6)
            )
        static_row = conditions["contextpilot-static_refit_causal"]
        static_system_replays += int(static_row.get("trials") or 0)
        systems_rows.append(
            {
                "workload": stem,
                "cached_ratio": static_row["measurements"]["cached_ratio"]["mean"],
                "requests": static_row["requests"],
                "trials": static_row["trials"],
            }
        )
        systems_ok = systems_ok and summary_ok
    add(
        "static_refit_systems_compact",
        systems_ok and static_system_replays == 18,
        {"accepted_replays": static_system_replays, "workloads": systems_rows},
    )

    quality = load_json(static_root / "quality-8b-static-refit-compact.json")
    persistent_quality = load_json(confirmation_root / "quality-scores-compact.json")
    persistent_condition = persistent_quality["conditions"][
        "contextpilot-online_incremental"
    ]
    quality_ok = (
        quality.get("model") == "Qwen/Qwen3-8B"
        and quality.get("condition") == "contextpilot-static_refit_causal"
        and quality.get("n") == 800
        and quality.get("overall", {}).get("tasks_scored") == 800
        and quality.get("overall", {}).get("relevance_tasks") == 640
        and quality.get("overall", {}).get("irrelevance_tasks") == 160
        and quality.get("overall") == persistent_condition.get("overall")
        and quality.get("by_domain") == persistent_condition.get("by_domain")
    )
    add(
        "static_and_persistent_quality_aggregates_identical",
        quality_ok,
        quality.get("overall"),
    )

    comparison_ok = True
    comparisons_checked = 0
    for baseline in BASELINES:
        for metric, paired_cases in METRICS.items():
            static_comparison = load_json(
                static_root
                / (
                    "quality-contextpilot-static_refit_causal-vs-"
                    f"{baseline}-{metric}.json"
                )
            )
            persistent_comparison = load_json(
                confirmation_root
                / (
                    "quality-contextpilot-online_incremental-vs-"
                    f"{baseline}-{metric}.json"
                )
            )
            comparison_ok = comparison_ok and (
                static_comparison.get("paired_cases") == paired_cases
                and static_comparison.get("sequence_state_dependent") is True
                and static_comparison.get(
                    "cluster_bootstrap_generalizes_across_request_sequences"
                )
                is False
                and static_comparison.get("mcnemar_independence_assumption_met")
                is False
                and static_comparison.get("bootstrap_samples") == 50_000
                and static_comparison.get("bootstrap_seed") == 42
                and all(
                    static_comparison.get(field) == persistent_comparison.get(field)
                    for field in NUMERIC_COMPARISON_FIELDS
                )
            )
            comparisons_checked += 1
    add(
        "static_and_persistent_paired_statistics_identical",
        comparison_ok and comparisons_checked == 6,
        {"comparisons": comparisons_checked},
    )

    sglang = load_json(static_root / "aggregate-counter-audit-summary.json")
    runs = sglang.get("runs") or []
    run_files = [str(row.get("file")) for row in runs]
    condition_counts = Counter(
        name.rsplit("-sglang-trial-", 1)[0].split("-", 1)[1]
        for name in run_files
        if "-sglang-trial-" in name and "-" in name
    )
    run_checks_ok = all(
        row.get("aggregate_cached_tokens") == row.get("reported_cached_tokens")
        and row.get("checks")
        == {
            "cached_total_equals_sum": True,
            "no_failed_requests": True,
            "only_index_0_missing": True,
            "prompt_counter_matches": True,
            "request_counter_matches": True,
        }
        for row in runs
    )
    sglang_ok = (
        sglang.get("declared_runs") == 72
        and sglang.get("accepted_runs") == 72
        and sglang.get("refused_runs") == 0
        and sglang.get("all_clean") is True
        and len(runs) == 72
        and len(set(run_files)) == 72
        and len(sglang.get("conditions") or []) == 12
        and set(condition_counts.values()) == {6}
        and run_checks_ok
    )
    add(
        "sglang_independent_aggregate_counter_audit",
        sglang_ok,
        {
            "declared": sglang.get("declared_runs"),
            "accepted": sglang.get("accepted_runs"),
            "refused": sglang.get("refused_runs"),
            "unique_files": len(set(run_files)),
            "conditions": len(condition_counts),
        },
    )

    archive_lines = [
        line
        for line in (static_root / "raw-archives.sha256")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    add(
        "two_raw_archive_hashes_recorded",
        len(archive_lines) == 2
        and all(
            re.fullmatch(r"[0-9a-f]{64}  /home/taghan/[^\s]+\.tar\.gz", line)
            is not None
            for line in archive_lines
        ),
        archive_lines,
    )

    all_checks_passed = all(check["passed"] for check in checks)
    return {
        "format_version": 1,
        "status": (
            "compact_evidence_valid_raw_archives_not_locally_verified"
            if all_checks_passed
            else "failed"
        ),
        "all_checks_passed": all_checks_passed,
        "checks_passed": sum(int(check["passed"]) for check in checks),
        "checks_total": len(checks),
        "raw_archives_verified_by_this_audit": False,
        "per_case_static_vs_persistent_identity_verified_by_this_audit": False,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-root", type=Path, default=DEFAULT_STATIC_ROOT)
    parser.add_argument(
        "--confirmation-root", type=Path, default=DEFAULT_CONFIRMATION_ROOT
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit(args.static_root.resolve(), args.confirmation_root.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
