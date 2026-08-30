#!/usr/bin/env python
"""Summarise dispatch-policy runs from `replay_vllm_concurrent.py`.

Reports on **arrival -> first token**, not TTFT. TTFT is timed from dispatch and
is structurally blind to the delay a dispatch policy imposes, so it cannot be
used to evaluate one; both metrics are printed side by side to show the gap.

Before any latency number is believed, this checks that the policy actually
acted. A policy that never reordered anything produces a null for a mechanical
reason, not a scientific one, and that failure has already occurred twice in
this project.

Usage:
    summarize_queuing_runs.py RUN_DIR [--baseline fifo]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    if low + 1 >= len(ordered):
        return ordered[low]
    return ordered[low] + (pos - low) * (ordered[low + 1] - ordered[low])


def arrival_to_first(run: dict) -> list[float]:
    return [
        r["queue_delay_seconds"] + r["ttft_seconds"]
        for r in run["results"]
        if r.get("ttft_seconds") is not None
    ]


def dispatch_order(run: dict) -> list:
    return [r["task_id"] for r in sorted(run["results"], key=lambda x: x["dispatch_offset"])]


def displacement(run: dict) -> int:
    """Largest |dispatch position - arrival position| over the run."""
    arrival = {r["task_id"]: i for i, r in enumerate(sorted(run["results"], key=lambda x: x["arrival_offset"]))}
    disp = {t: i for i, t in enumerate(dispatch_order(run))}
    return max(abs(disp[t] - arrival[t]) for t in arrival) if arrival else 0


def group(paths: list[Path]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for p in paths:
        run = json.loads(p.read_text())
        name = p.stem
        workload, _, policy = name.partition("-")
        out.setdefault(workload, {})[policy] = run
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--baseline", default="fifo")
    args = ap.parse_args()

    paths = sorted((args.run_dir / "replays").glob("*.json"))
    paths = [p for p in paths if not p.stem.startswith("stageA")]
    grouped = group(paths)

    for workload, runs in grouped.items():
        base = runs.get(args.baseline)
        print(f"\n{'=' * 100}\n{workload}\n{'=' * 100}")

        print("\n-- integrity --")
        for policy, run in sorted(runs.items()):
            d = run["aggregate_metric_delta"]
            itl = d.get("vllm:inter_token_latency_seconds_count", 0.0)
            expect = run["request_count"] * (run["decode"]["max_tokens"] - 1)
            foreign = "OK" if itl == expect else f"*** {itl:.0f} != {expect} ***"
            q = run["queue_delay_seconds"]["p50"] * 1000
            queue = "OK" if q > 50 else "*** NO QUEUE ***"
            print(f"   {policy:16s} reuse={run['reuse']['cached_ratio']*100:6.2f}%  "
                  f"achieved={run['achieved_rate']:7.4f}  qdelay_p50={q:9.1f}ms {queue:15s} traffic={foreign}")

        if base is not None:
            print("\n-- did the policy actually act? (vs baseline dispatch order) --")
            bseq = dispatch_order(base)
            for policy, run in sorted(runs.items()):
                if policy == args.baseline:
                    continue
                seq = dispatch_order(run)
                ndiff = sum(1 for a, b in zip(bseq, seq) if a != b)
                verdict = "ACTED" if ndiff else "*** INERT - null is mechanical, not scientific ***"
                print(f"   {policy:16s} reordered {ndiff:3d}/{len(seq)}   "
                      f"max displacement {displacement(run):3d}   {verdict}")

        print("\n-- arrival -> first token (ms) : the user-visible latency --")
        hdr = f"   {'policy':16s}{'mean':>10s}{'p50':>10s}{'p95':>10s}{'p99':>10s}{'max':>11s}"
        if base is not None:
            hdr += f"{'d_mean':>9s}{'d_p99':>9s}"
        print(hdr)
        bv = arrival_to_first(base) if base is not None else None
        for policy, run in sorted(runs.items()):
            v = arrival_to_first(run)
            row = (f"   {policy:16s}{statistics.fmean(v)*1000:10.1f}{percentile(v,.5)*1000:10.1f}"
                   f"{percentile(v,.95)*1000:10.1f}{percentile(v,.99)*1000:10.1f}{max(v)*1000:11.1f}")
            if bv is not None and policy != args.baseline:
                dm = (statistics.fmean(v) / statistics.fmean(bv) - 1) * 100
                dp = (percentile(v, .99) / percentile(bv, .99) - 1) * 100
                row += f"{dm:+8.1f}%{dp:+8.1f}%"
            print(row)

        print("\n-- TTFT (ms) : shown only to demonstrate it is the WRONG metric here --")
        for policy, run in sorted(runs.items()):
            t = run["ttft_seconds"]
            a = arrival_to_first(run)
            print(f"   {policy:16s}ttft_p50={t['p50']*1000:9.1f}   "
                  f"arrival_p50={percentile(a,.5)*1000:9.1f}   "
                  f"ratio={percentile(a,.5)/t['p50']:5.1f}x")


if __name__ == "__main__":
    main()
