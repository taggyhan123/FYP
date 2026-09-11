#!/usr/bin/env python
"""Closed-loop replay of multi-turn sessions against a live vLLM server.

Sends the exact prompt token ids that `simulate_sessions.py` simulates, via
`/v1/completions` with `max_tokens=1`, so each request measures prefill (time
to first token) and the server's own cached-token count
(`--enable-prompt-tokens-details`). `--concurrency` sessions run at once and
sessions start in index order as slots free, as in the simulation. Each
session's rounds are strictly sequential; `--wait-scale` multiplies TraceLab
tool waits between rounds (0 = back-to-back).

After the run, the actual dispatch order is replayed through
`tatm.prefix_cache_sim` at the server's real capacity, and per-request
simulated and measured cached tokens are compared.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from array import array
from pathlib import Path

import aiohttp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import simulate_sessions as S  # noqa: E402
from tatm.prefix_cache_sim import block_hashes, simulate_block_cache  # noqa: E402
from tatm.vllm_client import fetch_text, parse_prometheus, reset_prefix_cache  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))] if ordered else float("nan")


def cache_blocks(metrics_text: str) -> int | None:
    for line in metrics_text.splitlines():
        if line.startswith("vllm:cache_config_info") and 'num_gpu_blocks="' in line:
            return int(line.split('num_gpu_blocks="')[1].split('"')[0])
    return None


async def send(http, base_url, model, ids, timeout):
    payload = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}}
    started = time.perf_counter()
    first = None
    usage = None
    async with http.post(f"{base_url}/v1/completions", json=payload,
                         timeout=aiohttp.ClientTimeout(total=timeout)) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:300]}")
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("choices") and first is None:
                first = time.perf_counter()
            if chunk.get("usage"):
                usage = chunk["usage"]
    done = time.perf_counter()
    details = (usage or {}).get("prompt_tokens_details") or {}
    return started, first or done, done, (usage or {}).get("prompt_tokens"), details.get("cached_tokens")


async def replay(args, prompts, meta, waits):
    records = []
    queue = list(range(len(meta)))
    rng = random.Random(args.seed)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 4)
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as http:
        async def worker():
            while queue:
                s = queue.pop(0)
                for r in range(meta[s]["rounds"]):
                    ids = prompts[(s, r)].tolist()
                    try:
                        start, first, done, ptok, cached = await send(http, args.base_url, args.model, ids, args.timeout)
                        error = None
                    except Exception as exc:  # recorded, never silently dropped
                        start = first = done = time.perf_counter(); ptok = cached = None; error = repr(exc)[:300]
                    records.append({"session": s, "round": r, "dispatch": start - t0, "ttft": first - start,
                                    "latency": done - start, "prompt_tokens": ptok, "expected_tokens": len(ids),
                                    "cached_tokens": cached, "error": error})
                    if args.wait_scale > 0 and r + 1 < meta[s]["rounds"]:
                        await asyncio.sleep(args.wait_scale * rng.choice(waits))
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    return records, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True, help="multiturn-sessions results directory")
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--ordering", required=True)
    parser.add_argument("--mode", default="onchange")
    parser.add_argument("--plan-concurrency", type=int, default=8,
                        help="which planning order the adaptive orderings were built for")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-scale", type=float, default=0.0)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    S._init(str(args.out_dir))
    out = args.out_dir
    timeline = json.loads((out / "timelines" / f"{args.timeline}.json").read_text())
    plan = None
    if args.ordering in S.ADAPTIVE:
        path = out / "plans" / f"{args.timeline}-{args.mode}-C{args.plan_concurrency}-{args.ordering}.jsonl"
        plan = {r["task_id"]: r["tool_ids"] for r in map(json.loads, path.read_text().splitlines())}
    meta = S._STATE["meta"]
    if args.max_sessions:
        meta = meta[: args.max_sessions]
        S._STATE["meta"] = meta
        S._STATE["sessions"] = S._STATE["sessions"][: args.max_sessions]
    prompts = {(s, r): array("i", ids) for s, r, ids, _ in S.iter_prompt_ids(timeline, args.ordering, plan)}
    waits = json.loads((out / "tracelab_params.json").read_text())["wait_samples_seconds"]
    completion = {(s, r): 1 for s, r in prompts}

    if not reset_prefix_cache(args.base_url):
        raise SystemExit("prefix cache reset failed (VLLM_SERVER_DEV_MODE=1 required)")
    before_text = fetch_text(f"{args.base_url}/metrics")
    records, wall = asyncio.run(replay(args, prompts, meta, waits))
    after_text = fetch_text(f"{args.base_url}/metrics")
    before, after = parse_prometheus(before_text), parse_prometheus(after_text)
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in
             ("vllm:prompt_tokens_cached", "vllm:request_prefill_kv_computed_tokens_sum",
              "vllm:prefix_cache_hits", "vllm:prefix_cache_queries", "vllm:num_preemptions")}
    blocks = cache_blocks(after_text)

    ok = [r for r in records if r["error"] is None]
    order = sorted(ok, key=lambda r: r["dispatch"])
    sim, _ = simulate_block_cache(
        [block_hashes(prompts[(r["session"], r["round"])].tolist(), S.BLOCK) for r in order],
        [len(prompts[(r["session"], r["round"])]) for r in order],
        [completion[(r["session"], r["round"])] for r in order],
        None if blocks is None else blocks - 1, S.BLOCK)
    for record, simulated in zip(order, sim):
        record["sim_cached_tokens"] = simulated
    measured_total = sum(r["cached_tokens"] or 0 for r in ok)
    prompt_total = sum(r["prompt_tokens"] or 0 for r in ok)
    ttft = [r["ttft"] for r in ok]
    summary = {
        "timeline": args.timeline, "ordering": args.ordering, "mode": args.mode, "concurrency": args.concurrency,
        "wait_scale": args.wait_scale, "model": args.model, "requests": len(records), "errors": len(records) - len(ok),
        "prompt_token_mismatches": sum(1 for r in ok if r["prompt_tokens"] != r["expected_tokens"]),
        "wall_seconds": wall, "requests_per_second": len(ok) / wall,
        "hit_pct_measured": 100 * measured_total / prompt_total if prompt_total else None,
        "hit_pct_simulated_same_order": 100 * sum(sim) / prompt_total if prompt_total else None,
        "requests_matching_simulation": sum(1 for r in order if r["cached_tokens"] == r["sim_cached_tokens"]),
        "ttft_ms": {"mean": 1000 * sum(ttft) / len(ttft), "p50": 1000 * percentile(ttft, .5),
                    "p95": 1000 * percentile(ttft, .95), "p99": 1000 * percentile(ttft, .99), "max": 1000 * max(ttft)},
        "server_num_gpu_blocks": blocks, "metrics_delta": delta,
    }
    args.output.write_text(json.dumps({"summary": summary, "records": records}))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
