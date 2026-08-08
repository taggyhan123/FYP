#!/usr/bin/env python3
"""Compare BFCL score files with task-paired uncertainty estimates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.io import write_json
from tatm.paired_quality import compare_paired_binary, metric_scoped_scores


def load_scores(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores = payload.get("scores")
        if not isinstance(scores, list):
            raise ValueError(
                f"{path} has no task-level scores; regenerate it with the "
                "current score_bfcl_quality.py"
            )
        rows.extend(scores)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired BFCL comparison with a task-clustered bootstrap."
    )
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--metric",
        choices=("no_tool_correct", "name_correct", "full_correct"),
        required=True,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--sequence-state-dependent",
        action="store_true",
        help=(
            "Mark comparisons whose emitted ordering depends on earlier "
            "requests. Their interval is descriptive for the fixed sequence, "
            "not population inference over alternative request orders."
        ),
    )
    parser.add_argument(
        "--equivalence-margin-pp",
        type=float,
        help=(
            "Optional predeclared symmetric equivalence margin in percentage "
            "points. Omit it for estimation without a pass/fail claim."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_rows = load_scores(args.baseline)
    candidate_rows = load_scores(args.candidate)
    baseline_scoped = metric_scoped_scores(baseline_rows, args.metric)
    candidate_scoped = metric_scoped_scores(candidate_rows, args.metric)
    result = compare_paired_binary(
        baseline_scoped,
        candidate_scoped,
        metric=args.metric,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        equivalence_margin_pp=args.equivalence_margin_pp,
        sequence_state_dependent=args.sequence_state_dependent,
    )
    result["metric_scope"] = {
        "baseline_rows_loaded": len(baseline_rows),
        "baseline_rows_scoped": len(baseline_scoped),
        "candidate_rows_loaded": len(candidate_rows),
        "candidate_rows_scoped": len(candidate_scoped),
    }
    result["baseline_files"] = [path.as_posix() for path in args.baseline]
    result["candidate_files"] = [path.as_posix() for path in args.candidate]
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
