#!/usr/bin/env python
"""Score tool-selection accuracy on a retrieved-menu (ToolRet) replay.

`score_bfcl_quality.py` scores against BFCL ground truth and does not apply
here: the BM25 workloads are ToolRet (`source: toolret:*`) and carry
`gold_tool_ids` instead.

The model calls tools by *function name*, while the labels are tool *ids*, so
the mapping is rebuilt per record from the workload's parallel `tools` /
`tool_ids` arrays (their positional pairing is verified before use).

Reported per arm:

  answered        fraction of requests that emitted any tool call at all
  gold_hit        fraction where a called tool is in gold_tool_ids
  gold_hit_ceil   the same, restricted to requests whose menu actually
                  contains a gold tool -- retrieval already loses the rest,
                  so this isolates the ordering's contribution
  ceiling         fraction of requests whose menu contains a gold tool

Menus are identical across arms (every arm is a permutation of the same
retrieval), so `ceiling` must match across arms; if it does not, the arms are
not comparable and the run is invalid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fname(tool: dict) -> str | None:
    if "function" in tool and isinstance(tool["function"], dict):
        return tool["function"].get("name")
    return tool.get("name")


def called_names(tool_calls) -> list[str]:
    out = []
    for call in tool_calls or []:
        fn = call.get("function") if isinstance(call, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            out.append(fn["name"])
        elif isinstance(call, dict) and call.get("name"):
            out.append(call["name"])
    return out


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
            raise SystemExit("tools/tool_ids length mismatch; cannot map names to ids")
        wl[r["task_id"]] = {
            "name_to_id": {fname(t): tid for t, tid in zip(r["tools"], r["tool_ids"])},
            "gold": set(r.get("gold_tool_ids") or []),
            "menu": set(r["tool_ids"]),
        }

    results = json.loads(args.replay.read_text())["results"]
    n = answered = hit = ceil_n = ceil_hit = 0
    for row in results:
        meta = wl.get(row["task_id"])
        if meta is None:
            continue
        n += 1
        names = called_names(row.get("tool_calls"))
        ids = {meta["name_to_id"].get(nm) for nm in names}
        ids.discard(None)
        if names:
            answered += 1
        got = bool(ids & meta["gold"])
        hit += got
        if meta["gold"] & meta["menu"]:
            ceil_n += 1
            ceil_hit += got

    label = args.label or args.replay.stem
    print(json.dumps({
        "label": label,
        "n": n,
        "answered": round(answered / n, 4) if n else 0.0,
        "gold_hit": round(hit / n, 4) if n else 0.0,
        "gold_hit_ceil": round(ceil_hit / ceil_n, 4) if ceil_n else 0.0,
        "ceiling": round(ceil_n / n, 4) if n else 0.0,
    }))


if __name__ == "__main__":
    main()
