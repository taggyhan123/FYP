#!/usr/bin/env python
"""Emit a canonical-order arm: every request lists its tools in ONE global order.

Motivation. Measured on the BM25-retrieved workloads, both ToolTrie and
ContextPilot convert only a small fraction of the available cross-request tool
overlap into a shared *leading prefix* (k128: 8.32 tools overlap on average,
only 0.23-0.36 land in a shared prefix). ContextPilot's alignment emits
"matched prefix + remaining documents in their original order", so the tail
keeps each request's own BM25 ranking and can never share. A single global
canonical order removes that: any two requests agree on the relative order of
every tool they both hold, so their shared prefix runs until the first tool
present in one and not the other.

Two regimes:

  oracle  rank by tool frequency over the WHOLE workload. Uses future
          information, so it is an upper bound, not a deployable policy.
  causal  rank by frequency observed in strictly earlier requests only. This is
          deployable. Ties, and tools never seen before, keep their original
          retrieval rank, so BM25 relevance order is preserved wherever the
          counter has nothing to say.

Only the ordering changes: `tool_ids` and `tools` are permuted together (their
positional pairing is verified against an existing arm before use), and the menu
membership of every request is left exactly as retrieved.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def permute(record: dict, order: list[int]) -> dict:
    out = dict(record)
    out["tool_ids"] = [record["tool_ids"][i] for i in order]
    out["tools"] = [record["tools"][i] for i in order]
    out["ordering"] = record.get("ordering", "") and out.get("ordering")
    return out


def canonical_oracle(records: list[dict]) -> list[dict]:
    freq = collections.Counter(t for r in records for t in r["tool_ids"])
    out = []
    for r in records:
        idx = sorted(range(len(r["tool_ids"])),
                     key=lambda i: (-freq[r["tool_ids"][i]], i))
        out.append(permute(r, idx))
    return out


def canonical_causal(records: list[dict]) -> list[dict]:
    seen: collections.Counter = collections.Counter()
    out = []
    for r in records:
        idx = sorted(range(len(r["tool_ids"])),
                     key=lambda i: (-seen[r["tool_ids"][i]], i))
        out.append(permute(r, idx))
        seen.update(r["tool_ids"])          # update only AFTER emitting
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--regime", choices=("oracle", "causal"), required=True)
    args = ap.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    records = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    built = canonical_oracle(records) if args.regime == "oracle" else canonical_causal(records)

    for a, b in zip(records, built):
        assert sorted(a["tool_ids"]) == sorted(b["tool_ids"]), "menu membership changed"
        assert len(a["tools"]) == len(b["tools"])
    for b in built:
        b["ordering"] = f"canonical_{args.regime}"

    args.output.write_text("".join(json.dumps(r) + "\n" for r in built))
    print(f"{args.output}: {len(built)} records, regime={args.regime}")


if __name__ == "__main__":
    main()
