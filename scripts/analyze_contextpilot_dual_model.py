#!/usr/bin/env python3
"""Generate CPU-only quality comparisons after both GPU servers are stopped."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from run_contextpilot_dual_model import (
    MODEL_REVISIONS,
    PROJECT_ROOT,
    QUALITY_CONDITION_ORDER,
    QUALITY_METRICS,
    QUALITY_PAIRS,
    load_json,
    run_command,
    validate_quality_score,
    write_new_json,
)


def comparison_specs(
    model_results: Path, output: Path
) -> list[tuple[str, str, str, Path, list[str]]]:
    specs: list[tuple[str, str, str, Path, list[str]]] = []
    for baseline, candidate in QUALITY_PAIRS:
        for metric in QUALITY_METRICS:
            target = (
                output
                / "comparisons"
                / f"quality-{candidate}-vs-{baseline}-{metric}.json"
            )
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "compare_bfcl_quality.py"),
                "--baseline",
                str(model_results / "quality" / f"quality-{baseline}-score.json"),
                "--candidate",
                str(model_results / "quality" / f"quality-{candidate}-score.json"),
                "--metric",
                metric,
                "--bootstrap-samples",
                "50000",
                "--bootstrap-seed",
                "42",
                "--sequence-state-dependent",
                "--output",
                str(target),
            ]
            specs.append((baseline, candidate, metric, target, command))
    return specs


def comparison_errors(payload: dict[str, Any], metric: str) -> list[str]:
    errors: list[str] = []
    paired_cases = 160 if metric == "no_tool_correct" else 640
    if payload.get("sequence_state_dependent") is not True:
        errors.append("sequence_state_dependent is not true")
    if payload.get("cluster_bootstrap_generalizes_across_request_sequences") is not False:
        errors.append("request-sequence generalization is not false")
    if payload.get("mcnemar_independence_assumption_met") is not False:
        errors.append("McNemar independence is not false")
    if payload.get("paired_cases") != paired_cases:
        errors.append(f"paired_cases is not {paired_cases}")
    if payload.get("bootstrap_samples") != 50000:
        errors.append("bootstrap_samples is not 50000")
    if payload.get("bootstrap_seed") != 42:
        errors.append("bootstrap_seed is not 42")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_REVISIONS), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    model_results = args.model_results.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite analysis directory: {output}")
    if not model_results.is_dir():
        parser.error(f"Model result directory does not exist: {model_results}")
    run_summary = load_json(model_results / "model-run-summary.json")
    if not (
        run_summary.get("status") == "gpu_complete_pending_cpu_analysis"
        and run_summary.get("model") == args.model
        and run_summary.get("model_revision") == MODEL_REVISIONS[args.model]
        and run_summary.get("accepted_systems_replays") == 90
        and run_summary.get("accepted_quality_replays") == 5
        and run_summary.get("quality_score_files") == 5
        and run_summary.get("all_replays_reset_and_counter_clean") is True
    ):
        parser.error("GPU model-run summary is incomplete or does not match")

    for condition in QUALITY_CONDITION_ORDER:
        validate_quality_score(
            model_results / "quality" / f"quality-{condition}-score.json"
        )

    output.mkdir(parents=True)
    specs = comparison_specs(model_results, output)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_command,
                command,
                output / "logs" / f"compare-{target.stem}.log",
            ): (metric, target)
            for _, _, metric, target, command in specs
        }
        for future in as_completed(futures):
            metric, target = futures[future]
            try:
                future.result()
                errors = comparison_errors(load_json(target), metric)
                if errors:
                    failures.append(f"{target}: {'; '.join(errors)}")
            except Exception as error:  # Preserve every independent failure.
                failures.append(f"{target}: {error}")

    if failures:
        write_new_json(
            output / "analysis-summary.json",
            {
                "format_version": 1,
                "status": "quarantined",
                "model": args.model,
                "failures": failures,
            },
        )
        raise SystemExit("CPU analysis failed; preserve this directory as quarantine")

    write_new_json(
        output / "analysis-summary.json",
        {
            "format_version": 1,
            "status": "accepted",
            "model": args.model,
            "model_revision": MODEL_REVISIONS[args.model],
            "quality_comparison_files": len(specs),
            "bootstrap_samples": 50000,
            "bootstrap_seed": 42,
            "sequence_state_dependent": True,
            "workers": args.workers,
        },
    )
    print(json.dumps(load_json(output / "analysis-summary.json"), indent=2))


if __name__ == "__main__":
    main()
