#!/usr/bin/env python3
"""Audit the compact, git-tracked ContextPilot dual-model evidence.

The GPU-side audit validates raw replay files. This complementary audit is
deliberately limited to artifacts committed to git: it checks provenance,
summary arithmetic, quality comparison metadata, and the claims made from the
compact tables. It cannot verify or replace the raw tar archive.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = (
    PROJECT_ROOT / "reports" / "contextpilot-dual-model" / "20260809-004603"
)
DECLARED_MANIFEST = PROJECT_ROOT / "cluster" / "contextpilot-dual-model-manifest.json"

MODELS = {
    "qwen3-4b": {
        "model": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "capacity_tokens": 96_832,
        "role": "primary",
    },
    "qwen3-0.6b": {
        "model": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "capacity_tokens": 188_912,
        "role": "replication",
    },
}
STEMS = (
    "bfcl-padded64",
    "toolret-padded64",
    "toolret-bm25-k4",
    "toolret-bm25-k16",
    "toolret-bm25-k64",
    "toolret-bm25-k128",
)
CONDITIONS = (
    "original",
    "alphabetical",
    "tooltrie_v0",
    "contextpilot-online_incremental",
    "contextpilot-static_refit_causal",
)
QUALITY_PAIRS = (
    ("original", "tooltrie_v0"),
    ("original", "contextpilot-static_refit_causal"),
    ("original", "contextpilot-online_incremental"),
    ("alphabetical", "tooltrie_v0"),
    ("alphabetical", "contextpilot-static_refit_causal"),
    ("alphabetical", "contextpilot-online_incremental"),
    ("tooltrie_v0", "contextpilot-static_refit_causal"),
    ("tooltrie_v0", "contextpilot-online_incremental"),
    ("contextpilot-static_refit_causal", "contextpilot-online_incremental"),
)
QUALITY_METRICS = {
    "name_correct": ("function_name_accuracy", 640),
    "full_correct": ("full_accuracy", 640),
    "no_tool_correct": ("no_tool_accuracy", 160),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(report_root: Path = DEFAULT_REPORT_ROOT) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    json_files = sorted(report_root.rglob("*.json"))
    json_parse_failures: list[str] = []
    for path in json_files:
        try:
            load_json(path)
        except (OSError, json.JSONDecodeError) as error:
            json_parse_failures.append(f"{path.relative_to(report_root)}: {error}")
    add(
        "all_tracked_json_parses",
        len(json_files) == 81 and not json_parse_failures,
        {"files": len(json_files), "failures": json_parse_failures},
    )

    copied_manifest = report_root / "protocol-manifest.json"
    add(
        "protocol_manifest_byte_identical",
        copied_manifest.is_file()
        and copied_manifest.read_bytes() == DECLARED_MANIFEST.read_bytes(),
    )

    gpu_audit = load_json(report_root / "dual-model-audit.json")
    gpu_checks = gpu_audit.get("checks") or []
    add(
        "gpu_audit_acceptance",
        gpu_audit.get("status") == "accepted"
        and gpu_audit.get("accepted_gpu_replays") == 190
        and gpu_audit.get("all_checks_passed") is True
        and len(gpu_checks) == 33
        and all(row.get("passed") is True for row in gpu_checks),
        {"checks": len(gpu_checks), "replays": gpu_audit.get("accepted_gpu_replays")},
    )

    tokenizer = load_json(report_root / "tokenizer-compatibility.json")
    add(
        "tokenizer_gate",
        tokenizer.get("tokenizer_json_identical") is True
        and tokenizer.get("chat_template_identical") is True
        and tokenizer.get("compatible_for_shared_schema_token_accounting") is True,
    )

    pins = {
        "fyp": (report_root / "fyp-git-commit.txt").read_text(encoding="utf-8").strip(),
        "contextpilot": (report_root / "contextpilot-git-commit.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "vllm": (report_root / "vllm-version.txt").read_text(encoding="utf-8").strip(),
    }
    add(
        "software_pins",
        pins
        == {
            "fyp": "15285704e73af680c0125ea4bfeb0b54a14f278e",
            "contextpilot": "1fa0a143fdeda344585666648ab2b30cb7fea77f",
            "vllm": "0.26.0",
        },
        pins,
    )

    systems_summaries = 0
    systems_cells = 0
    cp_wins_over_tooltrie = 0
    quality_comparisons = 0
    for slug, expected in MODELS.items():
        model_root = report_root / slug
        run_summary = load_json(model_root / "model-run-summary.json")
        cache_config = load_json(model_root / "server-cache-config.json")
        block_size = int(cache_config.get("block_size") or 0)
        num_blocks = int(cache_config.get("num_gpu_blocks") or 0)
        add(
            f"{slug}:model_and_capacity",
            run_summary.get("model") == expected["model"]
            and run_summary.get("model_revision") == expected["revision"]
            and run_summary.get("model_role") == expected["role"]
            and run_summary.get("native_capacity_tokens")
            == expected["capacity_tokens"]
            and cache_config.get("capacity_tokens") == expected["capacity_tokens"]
            and block_size * num_blocks == expected["capacity_tokens"]
            and cache_config.get("enable_prefix_caching") is True,
        )

        for stem in STEMS:
            summary = load_json(model_root / "summaries" / f"{stem}-summary.json")
            conditions = summary.get("conditions") or {}
            summary_ok = (
                summary.get("engine") == "vllm"
                and set(conditions) == set(CONDITIONS)
                and summary.get("all_conditions_have_same_case_set") is True
                and summary.get("all_conditions_have_same_request_sequence") is True
                and summary.get("all_conditions_have_same_selected_tool_sets") is True
            )
            for condition, row in conditions.items():
                measurements = row.get("measurements") or {}
                cached = (measurements.get("cached_prompt_tokens") or {}).get("mean")
                computed = (measurements.get("computed_prompt_tokens") or {}).get("mean")
                ratio = (measurements.get("cached_ratio") or {}).get("mean")
                prompt_tokens = row.get("prompt_tokens")
                role = (row.get("execution_condition") or {}).get("role")
                expected_role = (
                    "ordinary_text_prefill_fallback"
                    if condition == "original"
                    else "ordering_candidate"
                )
                row_ok = (
                    row.get("trials") == 3
                    and row.get("requests") == 200
                    and isinstance(prompt_tokens, int)
                    and prompt_tokens > 0
                    and isinstance(cached, (int, float))
                    and isinstance(computed, (int, float))
                    and isinstance(ratio, (int, float))
                    and math.isclose(cached + computed, prompt_tokens, abs_tol=1e-6)
                    and math.isclose(ratio, cached / prompt_tokens, abs_tol=1e-6)
                    and role == expected_role
                )
                summary_ok = summary_ok and row_ok
                systems_cells += 1
            systems_summaries += int(summary_ok)

            tooltrie_ratio = conditions["tooltrie_v0"]["measurements"][
                "cached_ratio"
            ]["mean"]
            if all(
                conditions[condition]["measurements"]["cached_ratio"]["mean"]
                > tooltrie_ratio
                for condition in (
                    "contextpilot-online_incremental",
                    "contextpilot-static_refit_causal",
                )
            ):
                cp_wins_over_tooltrie += 1

        quality = load_json(model_root / "quality-scores-compact.json")
        generated_quality = load_json(
            model_root / "summaries" / "quality-scores-compact.json"
        )
        quality_conditions = quality.get("conditions") or {}
        quality_ok = (
            quality.get("model") == slug
            and quality.get("n_per_condition") == 800
            and set(quality_conditions) == set(CONDITIONS)
            and generated_quality.get("model") == expected["model"]
            and generated_quality.get("n_per_condition") == 800
            and generated_quality.get("conditions") == quality_conditions
        )
        for condition in CONDITIONS:
            overall = quality_conditions.get(condition, {}).get("overall", {})
            quality_ok = quality_ok and (
                overall.get("tasks_scored") == 800
                and overall.get("relevance_tasks") == 640
                and overall.get("irrelevance_tasks") == 160
                and all(
                    isinstance(overall.get(metric), (int, float))
                    and 0 <= overall[metric] <= 1
                    for metric, _ in QUALITY_METRICS.values()
                )
            )
        add(f"{slug}:quality_compact", quality_ok)

        comparison_root = model_root / "comparisons"
        comparison_files = sorted(comparison_root.glob("*.json"))
        comparison_ok = len(comparison_files) == 27
        for baseline, candidate in QUALITY_PAIRS:
            for metric, (score_key, paired_cases) in QUALITY_METRICS.items():
                path = (
                    comparison_root
                    / f"quality-{candidate}-vs-{baseline}-{metric}.json"
                )
                payload = load_json(path)
                candidate_score = quality_conditions[candidate]["overall"][score_key]
                baseline_score = quality_conditions[baseline]["overall"][score_key]
                comparison_ok = comparison_ok and (
                    payload.get("sequence_state_dependent") is True
                    and payload.get(
                        "cluster_bootstrap_generalizes_across_request_sequences"
                    )
                    is False
                    and payload.get("mcnemar_independence_assumption_met") is False
                    and payload.get("paired_cases") == paired_cases
                    and payload.get("bootstrap_samples") == 50_000
                    and payload.get("bootstrap_seed") == 42
                    and math.isclose(
                        payload.get("candidate_accuracy"),
                        candidate_score,
                        abs_tol=0.0001,
                    )
                    and math.isclose(
                        payload.get("baseline_accuracy"),
                        baseline_score,
                        abs_tol=0.0001,
                    )
                )
                quality_comparisons += 1
        add(f"{slug}:quality_comparisons", comparison_ok)

    add(
        "systems_summary_arithmetic_and_guards",
        systems_summaries == 12 and systems_cells == 60,
        {"summaries": systems_summaries, "condition_cells": systems_cells},
    )
    add(
        "both_contextpilot_arms_exceed_tooltrie_in_all_systems_cells",
        cp_wins_over_tooltrie == 12,
        {"model_workload_cells": cp_wins_over_tooltrie},
    )
    add(
        "quality_comparison_metadata",
        quality_comparisons == 54,
        {"comparisons": quality_comparisons},
    )

    hash_record = (report_root / "raw-archive.sha256").read_text(
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
            "compact_evidence_valid_raw_archive_not_locally_verified"
            if all_checks_passed
            else "failed"
        ),
        "all_checks_passed": all_checks_passed,
        "checks_passed": sum(int(check["passed"]) for check in checks),
        "checks_total": len(checks),
        "raw_archive_verified_by_this_audit": False,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit(args.report_root.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
