#!/usr/bin/env python3
"""Capture exact server-rendered tokens, block boundaries, and measured reuse.

The analytical trie works in canonical-tool-token units. This audit instead
asks the running vLLM server to render each exact chat/tools request via
``/tokenize``, partitions those IDs using the served block size, and optionally
replays the same requests to compare block-eligible prefixes with actual cache
hits, prefill time, and TTFT.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.io import read_jsonl
from tatm.measurement import (
    project_request_measurement,
    summarize_request_measurements,
)
from tatm.prefix_evidence import (
    RenderedPrefixIndex,
    prefix_pair_evidence,
    token_blocks,
)
from tatm.vllm_client import (
    fetch_text,
    metric_delta,
    parse_prometheus,
    request_json,
    reset_prefix_cache,
    response_projection,
    served_model,
    server_cache_config,
    tokenize_chat,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact rendered prompt IDs and vLLM block boundaries."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Replay each request after tokenization and capture cache/latency counters.",
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--allow-warm-start",
        action="store_true",
        help=(
            "Continue if /reset_prefix_cache is unavailable. The output is marked "
            "warm/contaminable and cannot establish exact prior-request attribution."
        ),
    )
    parser.add_argument("--allow-counter-mismatch", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model = served_model(base_url, args.model)
    cache_config = server_cache_config(base_url)
    try:
        block_size = int(cache_config.get("block_size") or 0)
    except (TypeError, ValueError) as error:
        raise SystemExit("vLLM did not expose a valid cache block size") from error
    if block_size < 1:
        raise SystemExit("vLLM did not expose a valid cache block size")

    records = list(read_jsonl(args.input))
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("Empty workload.")

    reset_ok: bool | None = None
    if args.measure:
        reset_ok = reset_prefix_cache(base_url)
        if not reset_ok and not args.allow_warm_start:
            raise SystemExit(
                "Prefix-cache reset failed. Start vLLM with "
                "VLLM_SERVER_DEV_MODE=1 or explicitly pass --allow-warm-start."
            )

    prior_prompts: list[list[int]] = []
    prefix_index = RenderedPrefixIndex(block_size)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        chat_template_kwargs = (
            {"enable_thinking": False} if args.disable_thinking else None
        )
        tokenized = tokenize_chat(
            base_url,
            model,
            record["messages"],
            record["tools"],
            chat_template_kwargs=chat_template_kwargs,
        )
        token_ids = tokenized["tokens"]
        immediate = (
            prefix_pair_evidence(prior_prompts[-1], token_ids, block_size)
            if prior_prompts
            else None
        )
        best = prefix_index.query(token_ids)
        row: dict[str, Any] = {
            "index": index,
            "case_id": record.get("case_id", record["task_id"]),
            "task_id": record["task_id"],
            "source": record["source"],
            "evidence_type": record["evidence_type"],
            "ordering": record["ordering"],
            "tool_ids": record["tool_ids"],
            "rendered_prompt_tokens": len(token_ids),
            "token_ids": token_ids,
            "blocks": token_blocks(token_ids, block_size),
            "immediate_prior_prefix": immediate,
            "best_prior_prefix": best,
        }

        if args.measure:
            payload: dict[str, Any] = {
                "model": model,
                "messages": record["messages"],
                "tools": record["tools"],
                "tool_choice": "auto",
                "temperature": 0,
                "seed": 0,
                "max_tokens": args.max_tokens,
            }
            if chat_template_kwargs:
                payload["chat_template_kwargs"] = chat_template_kwargs
            before = parse_prometheus(fetch_text(f"{base_url}/metrics"))
            started = time.perf_counter()
            response = request_json(
                "POST", f"{base_url}/v1/chat/completions", payload
            )
            wall_seconds = round(time.perf_counter() - started, 6)
            after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
            delta = metric_delta(before, after)
            projection = response_projection(response)
            measurement = project_request_measurement(delta, projection["usage"])
            measurement["wall_seconds"] = wall_seconds
            measured_cached = float(measurement["cached_tokens"] or 0.0)
            candidate_cached = int(best["cacheable_full_block_tokens"] or 0)
            row.update(
                {
                    **projection,
                    "metric_delta": delta,
                    "measurement": measurement,
                    "tokenize_count_matches_completion_usage": (
                        len(token_ids) == int(measurement["prompt_tokens"] or 0)
                    ),
                    "measured_minus_best_resident_candidate_tokens": (
                        measured_cached - candidate_cached
                    ),
                }
            )
        rows.append(row)
        prior_prompts.append(token_ids)
        observed_index = prefix_index.observe(token_ids)
        if observed_index != index:
            raise RuntimeError("rendered-prefix index drifted from workload order")

    validation: dict[str, Any] | None = None
    measurement_summary: dict[str, Any] | None = None
    if args.measure:
        measurements = [row["measurement"] for row in rows]
        usage_matches = all(
            bool(row["tokenize_count_matches_completion_usage"]) for row in rows
        )
        per_request_counter_matches = all(
            float(row["measurement"]["cached_tokens"] or 0.0)
            + float(row["measurement"]["computed_prefill_tokens"] or 0.0)
            == float(row["measurement"]["prompt_tokens"] or 0.0)
            for row in rows
        )
        validation = {
            "cache_reset_before": reset_ok,
            "tokenize_count_matches_completion_usage": usage_matches,
            "cached_plus_computed_matches_prompt_per_request": (
                per_request_counter_matches
            ),
            "measurement_consistent": (
                usage_matches and per_request_counter_matches
            ),
            "clean": bool(reset_ok and usage_matches and per_request_counter_matches),
        }
        measurement_summary = summarize_request_measurements(measurements)

    output = {
        "format_version": 1,
        "run_label": args.run_label,
        "server": base_url,
        "model": model,
        "input": args.input.as_posix(),
        "requests": len(rows),
        "server_cache_config": cache_config,
        "block_size": block_size,
        "token_source": "vllm_server_tokenize_chat_with_tools",
        "measurement_enabled": args.measure,
        "validation": validation,
        "measurement_by_reuse_bucket": measurement_summary,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote exact rendered-token evidence for {len(rows)} requests to {args.output}")
    if validation is not None:
        print(json.dumps(validation, indent=2, sort_keys=True))
        if (
            not validation["measurement_consistent"]
            and not args.allow_counter_mismatch
        ):
            raise SystemExit(
                "Rendered-token/replay validation failed; preserve but quarantine "
                "the result and rerun with an isolated server."
            )
        if not validation["cache_reset_before"]:
            print(
                "WARNING: warm-start output cannot attribute reuse solely to the "
                "requests in this file.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
