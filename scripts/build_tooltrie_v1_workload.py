#!/usr/bin/env python
"""Apply the ToolTrie-v1 ordering: reorder only what the trie has evidence for.

ToolTrie-v0 sorts unmatched tools alphabetically. Measured against the retriever's
own ordering, that displaces 99.1% of the menu by a mean of 41.9 positions at
k128 and returns 1.13% reuse -- 37 positions of displacement per point of reuse,
against ContextPilot's 8.0. Tool name correlates with neither relevance nor
commonality, so the permutation pays the full accuracy cost of moving the gold
tool (to depth 62.8) while creating almost no cross-request agreement.

This keeps unmatched tools in the order the retriever produced. The trie still
places whatever prefix it can match; everything else stays put.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tatm.analysis import load_processed
from tatm.tooltrie_v1 import RelevancePreservingToolTrie


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    ap.add_argument("--recency-window", type=int, default=128)
    ap.add_argument("--capacity-tokens", type=int, default=190896)
    ap.add_argument("--max-nodes", type=int, default=100_000)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    tools, _ = load_processed(args.processed_dir)
    records = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]

    planner = RelevancePreservingToolTrie(
        tools, fallback="alphabetical", recency_window=args.recency_window,
        capacity_tokens=args.capacity_tokens, max_nodes=args.max_nodes)

    out = []
    for record in records:
        payload = dict(zip(record["tool_ids"], record["tools"]))
        plan = planner.plan(tuple(record["tool_ids"]))
        ordered = list(plan.ordered_ids)
        assert sorted(ordered) == sorted(record["tool_ids"]), "menu membership changed"
        new = dict(record)
        new["base_ordering"] = record.get("ordering")
        new["ordering"] = "tooltrie_v1"
        new["tool_ids"] = ordered
        new["tools"] = [payload[t] for t in ordered]
        new["tooltrie_plan"] = plan.to_record()
        out.append(new)
        planner.observe(plan.ordered_ids)

    args.output.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{args.output.name}: {len(out)} requests, ordering=tooltrie_v1")


if __name__ == "__main__":
    main()
