#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.analysis import analyze_all, load_processed
from tatm.datasets import process_all
from tatm.reporting import write_reports
from tatm.serialization import TokenCounter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize TATM datasets and generate local analysis reports."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-0.6B",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "cluster" / "results",
        help="GPU probe results folded into the findings report when present.",
    )
    args = parser.parse_args()

    expected = (
        args.raw_dir / "toolret" / "tools",
        args.raw_dir / "toolret" / "queries",
        args.raw_dir / "bfcl",
    )
    if not all(path.exists() for path in expected):
        raise SystemExit(
            "Raw data is incomplete. Run `uv run python "
            "scripts/download_datasets.py` first."
        )

    print(f"Loading tokenizer: {args.tokenizer}")
    counter = TokenCounter(
        args.tokenizer,
        cache_dir=PROJECT_ROOT / "data" / "tokenizers",
    )
    print("Normalizing ToolRet and BFCL...")
    metadata = process_all(args.raw_dir, args.processed_dir, counter)
    tools, tasks = load_processed(args.processed_dir)
    print(f"Analyzing {len(tools):,} tools and {len(tasks):,} tasks...")
    summary = analyze_all(tools, tasks, tokenizer=args.tokenizer)
    write_reports(args.report_dir, metadata, summary, args.results_dir)
    print(json.dumps(metadata["combined"], indent=2))
    print(f"Reports: {args.report_dir}")


if __name__ == "__main__":
    main()
