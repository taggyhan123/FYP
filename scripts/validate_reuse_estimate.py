#!/usr/bin/env python3
"""Check the analytical trie reuse estimate against measured vLLM cache hits.

`trie_metrics` predicts reuse by counting shared *canonical tool* tokens and
rounding to block size. Nothing has ever confirmed that this tracks what vLLM
actually caches, so every reuse percentage in the reports is currently an
unvalidated model.

The two quantities are not expected to match exactly, and not only because the
model is optimistic:

- the model over-counts, because it credits reuse at tool boundaries while real
  blocks are 16-token aligned over the rendered prompt, so a shared run of tools
  usually loses a partial block at each end;
- the model under-counts, because it ignores the chat template and system
  preamble wrapping every request, which is identical across requests and is
  itself cached.

The deliverable is the size and direction of the gap, so later estimates can be
quoted with a known correction rather than at face value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.analysis import load_processed, trie_metrics
from tatm.io import read_jsonl
from tatm.measurement import (
    project_request_measurement,
    summarize_request_measurements,
)
from tatm.vllm_client import (
    fetch_text,
    metric_delta,
    parse_prometheus,
    request_json,
    reset_prefix_cache,
    served_model,
    server_cache_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model = served_model(base_url, args.model)
    cache_config = server_cache_config(base_url)
    if cache_config.get("enable_prefix_caching") is not True:
        raise SystemExit(
            "Server is not serving enable_prefix_caching=True; nothing to validate."
        )

    tools, _tasks = load_processed(args.processed_dir)
    records = list(read_jsonl(args.input))
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("Empty workload.")

    sequences = [tuple(record["tool_ids"]) for record in records]
    predicted = trie_metrics(sequences, tools)

    if not reset_prefix_cache(base_url):
        raise SystemExit(
            "vLLM prefix-cache reset failed; start the server with "
            "VLLM_SERVER_DEV_MODE=1 before validating reuse attribution."
        )
    measured_cached = 0.0
    measured_prompt = 0
    per_request: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        payload = {
            "model": model,
            "messages": record["messages"],
            "tools": record["tools"],
            "tool_choice": "auto",
            "temperature": 0,
            "seed": 0,
            "max_tokens": args.max_tokens,
        }
        before = parse_prometheus(fetch_text(f"{base_url}/metrics"))
        response = request_json("POST", f"{base_url}/v1/chat/completions", payload)
        after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
        delta = metric_delta(before, after)
        measurement = project_request_measurement(delta, response.get("usage", {}))
        cached = float(measurement["cached_tokens"] or 0.0)
        prompt_tokens = int(measurement["prompt_tokens"] or 0)
        measured_cached += cached
        measured_prompt += prompt_tokens
        per_request.append(
            {
                "index": index,
                "task_id": record["task_id"],
                "tool_count": len(record["tool_ids"]),
                "canonical_tool_tokens": record["canonical_tool_tokens"],
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached,
                "measurement": measurement,
                "metric_delta": delta,
            }
        )

    predicted_tokens = predicted["cacheable_block_tokens"]
    ratio = (measured_cached / predicted_tokens) if predicted_tokens else None
    # Rendered prompts are longer than the canonical tool text: chat template,
    # system preamble, and the user turn all add tokens the model never counts.
    template_overhead = measured_prompt - predicted["total_tool_tokens"]

    output = {
        "format_version": 2,
        "run_label": args.run_label,
        "server": base_url,
        "model": model,
        "server_cache_config": cache_config,
        "cache_reset_before": True,
        "input": args.input.as_posix(),
        "requests": len(records),
        "predicted": predicted,
        "measured": {
            "cached_prompt_tokens": measured_cached,
            "total_prompt_tokens": measured_prompt,
            "measured_reuse_ratio": round(measured_cached / measured_prompt, 6)
            if measured_prompt
            else 0.0,
        },
        "measured_over_predicted_ratio": round(ratio, 4) if ratio else None,
        "rendered_overhead_tokens": template_overhead,
        "direct_measurements_by_reuse_bucket": summarize_request_measurements(
            [row["measurement"] for row in per_request]
        ),
        "per_request": per_request,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"requests                     {len(records)}")
    print(f"predicted cacheable tokens   {predicted_tokens:,}")
    print(f"measured cached tokens       {measured_cached:,.0f}")
    print(f"measured / predicted         {ratio:.3f}" if ratio else "n/a")
    print(
        f"predicted reuse ratio        "
        f'{predicted["estimated_block_reuse_ratio"]:.2%} (of canonical tool tokens)'
    )
    print(
        f"measured reuse ratio         "
        f'{output["measured"]["measured_reuse_ratio"]:.2%} (of rendered prompt tokens)'
    )
    print(f"rendered overhead tokens     {template_overhead:,}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
