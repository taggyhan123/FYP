#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.io import read_jsonl
from tatm.vllm_client import (
    fetch_text,
    metric_delta,
    parse_prometheus,
    request_json,
    response_projection,
    served_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a generated TATM workload against vLLM."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument(
        "--tool-choice",
        choices=("auto", "required", "none"),
        default="auto",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model = served_model(base_url, args.model)
    workload = list(read_jsonl(args.input))
    if args.limit is not None:
        workload = workload[: args.limit]

    results = []
    aggregate_before = parse_prometheus(fetch_text(f"{base_url}/metrics"))
    started = time.perf_counter()
    for index, record in enumerate(workload):
        payload = {
            "model": model,
            "messages": record["messages"],
            "tools": record["tools"],
            "tool_choice": args.tool_choice,
            "temperature": 0,
            "seed": 0,
            "max_tokens": args.max_tokens,
        }
        before = parse_prometheus(fetch_text(f"{base_url}/metrics"))
        request_started = time.perf_counter()
        response = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            payload,
        )
        wall_seconds = time.perf_counter() - request_started
        after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
        results.append(
            {
                "index": index,
                "task_id": record["task_id"],
                "source": record["source"],
                "domain": record["domain"],
                "evidence_type": record["evidence_type"],
                "ordering": record["ordering"],
                "tool_ids": record["tool_ids"],
                "canonical_tool_tokens": record["canonical_tool_tokens"],
                "wall_seconds": round(wall_seconds, 6),
                **response_projection(response),
                "metric_delta": metric_delta(before, after),
            }
        )
        if args.pause_seconds:
            time.sleep(args.pause_seconds)

    aggregate_after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
    output = {
        "format_version": 1,
        "run_label": args.run_label,
        "server": base_url,
        "model": model,
        "input": args.input.as_posix(),
        "request_count": len(results),
        "wall_seconds": round(time.perf_counter() - started, 6),
        "aggregate_metric_delta": metric_delta(
            aggregate_before,
            aggregate_after,
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
