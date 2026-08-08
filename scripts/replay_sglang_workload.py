#!/usr/bin/env python3
"""Replay a TATM workload against stock SGLang/RadixAttention."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.io import read_jsonl
from tatm.sglang_client import (
    cached_token_projection,
    flush_cache,
    initial_missing_cached_reconciliation,
    metric_delta,
    parse_prometheus,
)
from tatm.vllm_client import (
    fetch_text,
    request_json,
    response_projection,
    served_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a generated TATM workload against SGLang."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument(
        "--tool-choice",
        choices=("auto", "required", "none"),
        default="auto",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Set chat_template_kwargs.enable_thinking=false for Qwen3.",
    )
    parser.add_argument(
        "--flush-before",
        action="store_true",
        help="Flush RadixAttention once immediately before this replay.",
    )
    parser.add_argument(
        "--allow-counter-mismatch",
        action="store_true",
        help=(
            "Write a contaminated/mismatched run without exiting non-zero. "
            "Never include such a run in the comparison table."
        ),
    )
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    if args.pause_seconds < 0:
        parser.error("--pause-seconds must be >= 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    base_url = args.base_url.rstrip("/")
    model = served_model(base_url, args.model)
    workload = list(read_jsonl(args.input))
    if args.limit is not None:
        workload = workload[: args.limit]
    if not workload:
        parser.error("Input workload is empty")

    flush_message = flush_cache(base_url) if args.flush_before else None
    aggregate_before = parse_prometheus(fetch_text(f"{base_url}/metrics"))
    results = []
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
            "return_cached_tokens_details": True,
        }
        if args.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
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
                "case_id": record.get("case_id", record["task_id"]),
                "task_id": record["task_id"],
                "source": record["source"],
                "domain": record["domain"],
                "evidence_type": record["evidence_type"],
                "ordering": record["ordering"],
                "base_ordering": record.get("base_ordering"),
                "menu_seed": record.get("menu_seed"),
                "tool_ids": record["tool_ids"],
                "canonical_tool_tokens": record["canonical_tool_tokens"],
                "wall_seconds": round(wall_seconds, 6),
                **response_projection(response),
                **cached_token_projection(response),
                "metric_delta": metric_delta(before, after),
            }
        )
        if args.pause_seconds:
            time.sleep(args.pause_seconds)

    aggregate_after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
    aggregate_delta = metric_delta(aggregate_before, aggregate_after)
    prompt_tokens_from_responses = sum(
        int((result.get("usage") or {}).get("prompt_tokens", 0))
        for result in results
    )
    response_cached_values = [
        result["cached_prompt_tokens"]
        for result in results
        if result.get("cached_prompt_tokens") is not None
    ]
    cached_tokens_from_responses = sum(response_cached_values)
    strict_cached_counter_matches = (
        len(response_cached_values) == len(results)
        and aggregate_delta.get("sglang:cached_tokens_total")
        == cached_tokens_from_responses
    )
    failed_request_indices = [
        result["index"]
        for result in results
        if result.get("finish_reason") is None
    ]
    validations = {
        "request_counter_matches": (
            aggregate_delta.get("sglang:num_requests_total") == len(results)
        ),
        "prompt_counter_matches": (
            aggregate_delta.get("sglang:prompt_tokens_total")
            == prompt_tokens_from_responses
        ),
        "cached_counter_matches": strict_cached_counter_matches,
        "no_failed_requests": not failed_request_indices,
    }
    output = {
        "format_version": 1,
        "engine": "sglang",
        "run_label": args.run_label,
        "server": base_url,
        "model": model,
        "input": args.input.as_posix(),
        "request_count": len(results),
        "wall_seconds": round(time.perf_counter() - started, 6),
        "cache_flushed_before": args.flush_before,
        "flush_message": flush_message,
        "aggregate_metric_delta": aggregate_delta,
        "counter_validation": {
            **validations,
            "clean": False,
            "response_prompt_tokens": prompt_tokens_from_responses,
            "response_cached_tokens": cached_tokens_from_responses,
        },
        "results": results,
    }
    reconciliation = initial_missing_cached_reconciliation(output)
    initial_missing_zero_reconciled = reconciliation["clean"]
    validations["strict_cached_response_coverage"] = (
        len(response_cached_values) == len(results)
    )
    validations["initial_missing_zero_reconciled"] = (
        initial_missing_zero_reconciled
    )
    validations["cached_counter_matches"] = (
        strict_cached_counter_matches or initial_missing_zero_reconciled
    )
    counters_clean = (
        validations["request_counter_matches"]
        and validations["prompt_counter_matches"]
        and validations["cached_counter_matches"]
        and validations["no_failed_requests"]
    )
    output["counter_validation"].update(
        {
            **validations,
            "clean": counters_clean,
            "failed_request_indices": failed_request_indices,
            "missing_cached_indices": reconciliation["missing_cached_indices"],
            "aggregate_cached_tokens": reconciliation["aggregate_cached_tokens"],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(results)} SGLang results to {args.output}")
    print(json.dumps(output["counter_validation"], indent=2, sort_keys=True))
    if not counters_clean and not args.allow_counter_mismatch:
        raise SystemExit(
            "SGLang counters do not match this replay; preserve and quarantine "
            "the result, then rerun with no other traffic"
        )


if __name__ == "__main__":
    main()
