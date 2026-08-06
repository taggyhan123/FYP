#!/usr/bin/env python3
"""Validate and compact the initial-brief controlled-cache pressure matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.io import write_json
from tatm.pressure_summary import summarize_pressure_replays


def parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the predeclared 6 x 4 controlled-cache matrix."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--expected-capacity-tokens", type=int, required=True)
    parser.add_argument("--required-peak-fraction", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [
        (label, json.loads(path.read_text(encoding="utf-8")))
        for label, path in args.run
    ]
    summary = summarize_pressure_replays(
        payloads,
        expected_capacity_tokens=args.expected_capacity_tokens,
        required_peak_fraction=args.required_peak_fraction,
    )
    write_json(args.output, summary)

    print(
        f"Accepted pressure regimes: {summary['accepted_regime_runs']}/"
        f"{summary['all_regime_runs']}"
    )
    print(f"Validation checks: {summary['checks_passed']}/{summary['checks_total']}")
    print(f"Wrote {args.output}")
    if not summary["all_regime_runs_accepted"]:
        raise SystemExit("The controlled-cache pressure matrix is not accepted")


if __name__ == "__main__":
    main()
