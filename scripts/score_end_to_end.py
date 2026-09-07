#!/usr/bin/env python
"""Score a tool-selection replay on all four accuracy definitions.

`score_tool_selection.py` reports `gold_hit_ceil` -- correct if the model
called *any one* gold tool, restricted to requests whose menu contained a gold
tool at all. That is a useful "did retrieval and ordering put the tool within
reach" number, but it is not accuracy: it ignores how many gold tools a task
needs (44% need 2+), it ignores spurious calls, and being ceiling-conditioned
it hides the cost of a policy that pushes gold tools out of the menu entirely.

Both failure modes are real and observed. On the retrieve-64/present-10
design, `frequency` posts the highest conditional F1 in its table while
delivering no end-to-end gain, because the gold tools surviving its truncation
are the easy ones.

Reported per arm:

  ceiling         fraction of requests whose menu contains a gold tool
  gold_hit_ceil   any-gold-hit, ceiling-conditioned (the legacy metric)
  precision       correct calls / calls made, averaged over ceiling requests
  recall          correct calls / gold needed, averaged the same way
  f1              harmonic mean, per request, then averaged
  end_to_end_f1   ceiling * f1 -- the number a deployment actually sees

`end_to_end_f1` is the primary metric. Compare arms on it. Use `f1` only when
the menus are identical across arms (so `ceiling` matches), and say so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fname(tool: dict) -> str | None:
    fn = tool.get("function")
    if isinstance(fn, dict):
        return fn.get("name")
    return tool.get("name")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    wl = {}
    for line in args.workload.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if len(r["tools"]) != len(r["tool_ids"]):
            raise SystemExit("tools/tool_ids length mismatch")
        wl[r["task_id"]] = {
            "name_to_id": {fname(t): tid for t, tid in zip(r["tools"], r["tool_ids"])},
            "gold": set(r.get("gold_tool_ids") or []),
            "menu": set(r["tool_ids"]),
        }

    results = json.loads(args.replay.read_text())["results"]
    n = ceil_n = hit = 0
    precs: list[float] = []
    recs: list[float] = []
    f1s: list[float] = []
    for row in results:
        meta = wl.get(row["task_id"])
        if meta is None:
            continue
        n += 1
        gold = meta["gold"]
        if not gold or not (gold & meta["menu"]):
            continue
        ceil_n += 1
        called = set()
        for call in row.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else None
            tid = meta["name_to_id"].get(name)
            if tid:
                called.add(tid)
        tp = len(called & gold)
        hit += bool(tp)
        p = tp / len(called) if called else 0.0
        r_ = tp / len(gold)
        precs.append(p)
        recs.append(r_)
        f1s.append(2 * p * r_ / (p + r_) if (p + r_) else 0.0)

    ceiling = ceil_n / n if n else 0.0
    f1 = sum(f1s) / len(f1s) if f1s else 0.0
    print(json.dumps({
        "label": args.label or args.replay.stem,
        "n": n,
        "ceiling": round(ceiling, 4),
        "gold_hit_ceil": round(hit / ceil_n, 4) if ceil_n else 0.0,
        "precision": round(sum(precs) / len(precs), 4) if precs else 0.0,
        "recall": round(sum(recs) / len(recs), 4) if recs else 0.0,
        "f1": round(f1, 4),
        "end_to_end_f1": round(ceiling * f1, 4),
    }))


if __name__ == "__main__":
    main()
