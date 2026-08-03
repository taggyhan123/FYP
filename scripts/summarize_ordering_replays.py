#!/usr/bin/env python3
"""Summarize repeated vLLM or SGLang ordering replays with 95% intervals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.io import write_json
from tatm.replay_summary import summarize_labeled_replays


def parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize clean repeated ordering replays."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        metavar="LABEL=PATH",
        help="Repeat the same label for repeated trials.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [
        (label, json.loads(path.read_text(encoding="utf-8")))
        for label, path in args.run
    ]
    summary = summarize_labeled_replays(payloads)
    if not summary["all_conditions_have_same_case_set"]:
        raise SystemExit("Conditions do not contain the same paired case set")
    if not summary["all_conditions_have_same_selected_tool_sets"]:
        raise SystemExit("Conditions changed one or more selected tool sets")
    write_json(args.output, summary)

    print(f"Engine: {summary['engine']}")
    print(
        "Same request sequence: "
        f"{summary['all_conditions_have_same_request_sequence']}"
    )
    print(f"{'condition':<28} {'reuse':>10} {'TTFT seconds':>14} {'wall seconds':>14}")
    for label, condition in summary["conditions"].items():
        measurements = condition["measurements"]
        print(
            f"{label:<28} "
            f"{measurements['cached_ratio']['mean']:>10.4f} "
            f"{measurements['ttft_seconds']['mean']:>14.3f} "
            f"{measurements['wall_seconds']['mean']:>14.3f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
