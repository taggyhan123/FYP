#!/usr/bin/env python
"""ToolTrie with a causal-frequency fallback instead of an alphabetical one.

ToolTrie-v0 orders the tools it cannot match against its trie alphabetically.
That scatters the tools common to many requests through the menu, so the very
first ordering it emits has a short usable prefix and every later request
inherits it. Measured on a constructed 50%-overlap workload, v0 places the common
core at mean position 31.1 of 64 -- no better than plain alphabetical -- and
reaches a shared prefix of 7.85 of a possible 32.

Sorting the fallback by frequency observed in STRICTLY EARLIER requests, with a
global tie-break on tool_id so every request emits ties in the same order, hoists
the core consistently: position 15.6, prefix 31.85. Both conditions are needed --
hoisting without a global tie-break leaves 200 different core orderings and a
prefix of 0.94.

Causal: the counter is updated only after a request has been planned, so no
future information is used and no training set is required.
"""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tatm.analysis import load_processed
from tatm.tooltrie import ToolTrie


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

    freq: collections.Counter = collections.Counter()
    planner = ToolTrie(tools, fallback="alphabetical", recency_window=args.recency_window,
                       capacity_tokens=args.capacity_tokens, max_nodes=args.max_nodes)
    planner._fallback_order = lambda ids: tuple(sorted(ids, key=lambda t: (-freq[t], t)))

    out = []
    for record in records:
        payload = dict(zip(record["tool_ids"], record["tools"]))
        plan = planner.plan(tuple(record["tool_ids"]))
        ordered = list(plan.ordered_ids)
        assert sorted(ordered) == sorted(record["tool_ids"]), "menu membership changed"
        new = dict(record)
        new["base_ordering"] = record.get("ordering")
        new["ordering"] = "tooltrie_freqfallback"
        new["tool_ids"] = ordered
        new["tools"] = [payload[t] for t in ordered]
        new["tooltrie_plan"] = plan.to_record()
        out.append(new)
        planner.observe(plan.ordered_ids)
        freq.update(record["tool_ids"])          # causal: only after planning

    args.output.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{args.output.name}: {len(out)} requests, fallback=causal-frequency")


if __name__ == "__main__":
    main()
