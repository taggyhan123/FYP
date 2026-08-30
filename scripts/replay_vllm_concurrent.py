#!/usr/bin/env python
"""Open-loop concurrent replay driver with client-side streaming timing.

This is deliberately separate from `replay_vllm_workload.py`. That driver is
strictly serial and derives per-request TTFT by differencing the cumulative
Prometheus `vllm:time_to_first_token_seconds_sum` counter around each request.
That attribution is only valid with exactly one request in flight, so it cannot
be reused here. This driver instead times the arrival of the first streamed
chunk on the client, which is valid at any concurrency.

Two dispatch policies are supported behind a shared in-flight cap, so the only
difference between them is which pending request is chosen next:

  fifo      dispatch in arrival order (baseline)
  affinity  among the first --window pending requests, dispatch the one sharing
            the longest leading tool_ids prefix with the last dispatched
            request, subject to a --max-delay fairness bound

The affinity policy is the mean-vs-max adaptor: grouping prefix-affine requests
lowers mean TTFT but delays some requests, and --max-delay bounds that damage.
Both policies are implemented entirely client-side; the vLLM server and its
prefix cache are unmodified.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tatm.vllm_client import (  # noqa: E402
    fetch_text,
    metric_delta,
    parse_prometheus,
    reset_prefix_cache,
)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def arrival_offsets(count: int, rate: float, seed: int) -> list[float]:
    """Poisson arrivals: exponential inter-arrival gaps at `rate` req/s.

    rate <= 0 means every request is available at t=0, which reproduces a
    closed-loop saturation test under the in-flight cap.
    """
    if rate <= 0:
        return [0.0] * count
    rng = random.Random(seed)
    offsets: list[float] = []
    clock = 0.0
    for _ in range(count):
        offsets.append(clock)
        clock += rng.expovariate(rate)
    return offsets


_RNG = random.Random(0)


def shared_prefix_length(left: list[str], right: list[str]) -> int:
    shared = 0
    for a, b in zip(left, right):
        if a != b:
            break
        shared += 1
    return shared


class PrefixTrie:
    """Longest-leading-prefix lookup over every dispatched tool_ids sequence.

    This is the client-side stand-in for the engine's radix tree. The older
    `affinity` policy scores candidates only against the *last* dispatched
    request, which is a myopic greedy chain blind to everything else resident in
    the cache. Matching against all dispatched sequences is what a longest-prefix
    -match scheduler does against its actual cache state.

    The lookup ignores eviction, so it overestimates residency for long runs.
    That is acceptable here because it is used to rank candidates against each
    other, not to predict an absolute hit rate.
    """

    def __init__(self) -> None:
        self.root: dict = {}

    def insert(self, tool_ids: list[str]) -> None:
        node = self.root
        for tool in tool_ids:
            node = node.setdefault(tool, {})

    def longest_prefix(self, tool_ids: list[str]) -> int:
        node = self.root
        depth = 0
        for tool in tool_ids:
            nxt = node.get(tool)
            if nxt is None:
                break
            node = nxt
            depth += 1
        return depth


def choose_index(
    pending: list[dict],
    policy: str,
    window: int,
    max_delay: float,
    last_tool_ids: list[str],
    now: float,
    trie: "PrefixTrie | None" = None,
    aging_rate: float = 0.0,
) -> int:
    """Index into `pending` of the request to dispatch next.

    fifo      arrival order.
    affinity  longest shared tool_ids prefix with the LAST dispatched request.
    sjf       smallest job first, by canonical_tool_tokens. Job size is known
              exactly at arrival here because prefill dominates and the prompt
              is fixed, so no length prediction is needed.
    suffix    smallest ESTIMATED UNCACHED job first: canonical_tool_tokens
              scaled by the fraction of the menu not already covered by a
              dispatched prefix. This is simultaneously SJF and prefix-aware.

    `sjf` and `suffix` are scored lowest-wins and age with waiting time
    (`score -= aging_rate * wait`), which bounds starvation smoothly rather
    than with a hard wall-clock cutoff.
    """
    if policy == "fifo":
        return 0

    if policy == "random":
        # Control for the saturation artifact. Under a deep queue, FIFO makes
        # waiting time grow with arrival position, so ANY order uncorrelated
        # with arrival lowers the median and raises the tail even when the
        # priority key carries no information. `random` measures that baseline,
        # so a size-aware policy must be credited only with the margin it wins
        # ABOVE random, not with its full margin over FIFO.
        limit = len(pending) if window <= 0 else min(window, len(pending))
        return _RNG.randrange(limit)

    if policy == "affinity":
        if not last_tool_ids:
            return 0
        # Fairness bound: anything waiting longer than max_delay goes first.
        if max_delay > 0:
            for index, item in enumerate(pending):
                if now - item["available_at"] >= max_delay:
                    return index
        limit = len(pending) if window <= 0 else min(window, len(pending))
        best_index = 0
        best_score = -1
        for index in range(limit):
            score = shared_prefix_length(last_tool_ids, pending[index]["record"]["tool_ids"])
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    limit = len(pending) if window <= 0 else min(window, len(pending))
    best_index = 0
    best_score: float | None = None
    for index in range(limit):
        item = pending[index]
        record = item["record"]
        size = float(record.get("canonical_tool_tokens") or 0.0)
        if policy == "suffix" and trie is not None:
            tool_ids = record["tool_ids"]
            if tool_ids:
                shared = trie.longest_prefix(tool_ids)
                size *= 1.0 - shared / len(tool_ids)
        score = size - aging_rate * (now - item["available_at"])
        if best_score is None or score < best_score:
            best_score = score
            best_index = index
    return best_index


async def issue(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    record: dict,
    max_tokens: int,
    ignore_eos: bool,
) -> dict:
    payload = {
        "model": model,
        "messages": record["messages"],
        "tools": record["tools"],
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    sent = time.perf_counter()
    first_chunk: float | None = None
    usage: dict = {}
    async with session.post(
        f"{base_url}/v1/chat/completions", json=payload
    ) as response:
        response.raise_for_status()
        async for raw in response.content:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                continue
            if first_chunk is None:
                first_chunk = time.perf_counter()
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
    done = time.perf_counter()
    return {
        "ttft_seconds": (first_chunk - sent) if first_chunk else None,
        "e2e_seconds": done - sent,
        "usage": usage,
    }


async def run(args: argparse.Namespace) -> dict:
    workload = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.limit:
        workload = workload[: args.limit]

    # Refuse to run a dispatch policy that cannot actually act. Both of these
    # previously produced confident nulls for purely mechanical reasons.
    if args.dispatch != "fifo" and args.max_inflight <= 0:
        raise SystemExit(
            f"--dispatch {args.dispatch} requires a finite --max-inflight. With 0 the "
            "dispatch loop drains the pending queue every tick, so the policy never "
            "sees more than one candidate and silently degenerates to FIFO."
        )
    if args.dispatch in ("sjf", "suffix"):
        missing = sum(1 for r in workload if not r.get("canonical_tool_tokens"))
        if missing:
            raise SystemExit(
                f"--dispatch {args.dispatch} ranks by canonical_tool_tokens, but "
                f"{missing}/{len(workload)} records lack it; every job would score 0 "
                "and the policy would degenerate to FIFO."
            )

    _RNG.seed(args.arrival_seed)

    base_url = args.base_url.rstrip("/")
    if args.reset_before and not reset_prefix_cache(base_url):
        raise SystemExit("prefix cache reset failed (needs VLLM_SERVER_DEV_MODE=1)")

    metrics_before = parse_prometheus(fetch_text(f"{base_url}/metrics"))
    offsets = arrival_offsets(len(workload), args.rate, args.arrival_seed)

    pending: list[dict] = []
    results: list[dict] = []
    inflight: set[asyncio.Task] = set()
    last_tool_ids: list[str] = []
    trie = PrefixTrie()
    next_arrival = 0
    started = time.perf_counter()

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def track(item: dict) -> None:
            dispatched = time.perf_counter()
            measured = await issue(
                session, base_url, args.model, item["record"],
                args.max_tokens, args.ignore_eos,
            )
            record = item["record"]
            results.append(
                {
                    "index": item["index"],
                    "task_id": record["task_id"],
                    "source": record["source"],
                    "domain": record["domain"],
                    "ordering": record["ordering"],
                    "tool_ids": record["tool_ids"],
                    "arrival_offset": item["available_at"] - started,
                    "queue_delay_seconds": dispatched - item["available_at"],
                    "dispatch_offset": dispatched - started,
                    **measured,
                }
            )

        while next_arrival < len(workload) or pending or inflight:
            now = time.perf_counter()
            elapsed = now - started

            while next_arrival < len(workload) and offsets[next_arrival] <= elapsed:
                pending.append(
                    {
                        "index": next_arrival,
                        "record": workload[next_arrival],
                        "available_at": started + offsets[next_arrival],
                    }
                )
                next_arrival += 1

            while pending and (args.max_inflight <= 0 or len(inflight) < args.max_inflight):
                choice = choose_index(
                    pending, args.dispatch, args.window, args.max_delay,
                    last_tool_ids, time.perf_counter(),
                    trie, args.aging_rate,
                )
                item = pending.pop(choice)
                last_tool_ids = item["record"]["tool_ids"]
                trie.insert(last_tool_ids)
                task = asyncio.create_task(track(item))
                inflight.add(task)
                task.add_done_callback(inflight.discard)

            if inflight:
                await asyncio.wait(inflight, timeout=0.002, return_when=asyncio.FIRST_COMPLETED)
            elif next_arrival < len(workload):
                await asyncio.sleep(min(0.002, max(0.0, offsets[next_arrival] - (time.perf_counter() - started))))
            else:
                await asyncio.sleep(0.002)

    wall = time.perf_counter() - started
    metrics_after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
    delta = metric_delta(metrics_before, metrics_after)

    results.sort(key=lambda row: row["index"])
    ttft = [r["ttft_seconds"] for r in results if r["ttft_seconds"] is not None]
    e2e = [r["e2e_seconds"] for r in results]
    queue = [r["queue_delay_seconds"] for r in results]
    # The user-visible latency. ttft_seconds alone is timed from DISPATCH and is
    # structurally blind to any delay a dispatch policy imposes, so it must not
    # be used to evaluate one.
    arrival_to_first = [
        r["queue_delay_seconds"] + r["ttft_seconds"]
        for r in results
        if r["ttft_seconds"] is not None
    ]
    cached = delta.get("vllm:prompt_tokens_cached", 0.0)
    computed = delta.get("vllm:request_prefill_kv_computed_tokens_sum", 0.0)

    def summary(values: list[float]) -> dict:
        return {
            "n": len(values),
            "mean": round(statistics.fmean(values), 6) if values else 0.0,
            "p50": round(percentile(values, 0.50), 6),
            "p90": round(percentile(values, 0.90), 6),
            "p95": round(percentile(values, 0.95), 6),
            "p99": round(percentile(values, 0.99), 6),
            "max": round(max(values), 6) if values else 0.0,
        }

    return {
        "format_version": 1,
        "run_label": args.run_label,
        "input": str(args.input),
        "model": args.model,
        "server": base_url,
        "load": {
            "rate_requests_per_second": args.rate,
            "max_inflight": args.max_inflight,
            "arrival_seed": args.arrival_seed,
            "arrival_process": "poisson" if args.rate > 0 else "all_available_at_zero",
        },
        "dispatch": {
            "policy": args.dispatch,
            "window": args.window,
            "max_delay_seconds": args.max_delay,
            "aging_rate_tokens_per_second": args.aging_rate,
        },
        "decode": {"max_tokens": args.max_tokens, "ignore_eos": args.ignore_eos},
        "cache_reset_before": args.reset_before,
        "request_count": len(results),
        "wall_seconds": round(wall, 6),
        "achieved_rate": round(len(results) / wall, 4) if wall else 0.0,
        "reuse": {
            "cached_prompt_tokens": cached,
            "computed_prompt_tokens": computed,
            "cached_ratio": round(cached / (cached + computed), 6) if (cached + computed) else 0.0,
        },
        "peak_num_preemptions": delta.get("vllm:num_preemptions", 0.0),
        "arrival_to_first_token_seconds": summary(arrival_to_first),
        "ttft_seconds": summary(ttft),
        "e2e_seconds": summary(e2e),
        "queue_delay_seconds": summary(queue),
        "aggregate_metric_delta": delta,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    parser.add_argument("--rate", type=float, default=0.0,
                        help="Poisson arrival rate in req/s; 0 = all available at t=0")
    parser.add_argument("--max-inflight", type=int, default=0,
                        help="Client-side cap on concurrent in-flight requests; 0 = uncapped")
    parser.add_argument("--arrival-seed", type=int, default=42)
    parser.add_argument("--dispatch", choices=("fifo", "affinity", "sjf", "suffix", "random"), default="fifo")
    parser.add_argument("--aging-rate", type=float, default=0.0,
                        help="sjf/suffix: subtract this many tokens of score per second "
                             "waited, bounding starvation. 0 disables aging.")
    parser.add_argument("--window", type=int, default=0,
                        help="affinity: consider only the first W pending requests; 0 = all")
    parser.add_argument("--max-delay", type=float, default=0.0,
                        help="affinity: force-dispatch anything waiting this long (s); 0 = off")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--reset-before", action="store_true", default=True)
    parser.add_argument("--no-reset-before", dest="reset_before", action="store_false")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = asyncio.run(run(args))
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    t = payload["ttft_seconds"]
    print(
        f"{args.run_label}: n={payload['request_count']} "
        f"rate={payload['achieved_rate']} reuse={100*payload['reuse']['cached_ratio']:.2f}% "
        f"ttft p50={1000*t['p50']:.1f} p95={1000*t['p95']:.1f} p99={1000*t['p99']:.1f} max={1000*t['max']:.1f} ms"
    )


if __name__ == "__main__":
    main()
