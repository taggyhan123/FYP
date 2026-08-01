#!/usr/bin/env python3
"""Score a replayed BFCL workload against ground truth.

Consumes the JSON written by `replay_vllm_workload.py` (which already
captures `tool_calls` per request via `response_projection`) and scores
each result with `tatm.bfcl_score`. Requires
`data/raw/bfcl/possible_answer/*.json`, fetched by
`scripts/download_datasets.py` after the ground-truth files were added to
`src/tatm/download.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.bfcl_score import aggregate, load_ground_truth, score_task


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a replayed BFCL workload's tool calls against ground truth."
    )
    parser.add_argument("--replay-result", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    replay = json.loads(args.replay_result.read_text(encoding="utf-8"))
    ground_truth = load_ground_truth(args.raw_dir)

    scores = []
    skipped = 0
    for result in replay["results"]:
        score = score_task(
            result["domain"], result["task_id"], result.get("tool_calls"), ground_truth
        )
        if score is None:
            skipped += 1
            continue
        score["ordering"] = result["ordering"]
        scores.append(score)

    by_domain: dict[str, list] = {}
    for score in scores:
        by_domain.setdefault(score["domain"], []).append(score)

    output = {
        "format_version": 1,
        "replay_result": args.replay_result.as_posix(),
        "ordering": replay.get("results", [{}])[0].get("ordering"),
        "run_label": replay.get("run_label"),
        "skipped_no_ground_truth": skipped,
        "overall": aggregate(scores),
        "by_domain": {domain: aggregate(items) for domain, items in sorted(by_domain.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["overall"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
