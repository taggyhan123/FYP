#!/usr/bin/env python
"""Build a menu workload with a controlled fraction of tools shared by all requests.

Every workload in this project sits at one of two extremes -- padded menus share
98.4% of their tools between any two requests, BM25-retrieved menus share 0% --
and nothing exists between. This fills that band.

Construction, from `bfcl-padded64`: each request keeps its own task-specific tool
and a 64-tool menu. Of the remaining 63, `--core` come from a fixed core shared by
every request; the rest are drawn per-request from the processed tool corpus and
are disjoint across requests. Tools shared by all is therefore core/64.

Replacements are matched on `schema_tokens` to the core tool they displace, so
prompt length stays put and reuse differences cannot be an artefact of menu size.

`original` here is a per-request deterministic shuffle: genuinely unordered,
rather than padded-64's "put the differing tool first", which is a worst case.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="padded-64 original workload")
    ap.add_argument("--tools", type=Path, default=Path("data/processed/tools.jsonl"))
    ap.add_argument("--core", type=int, required=True, help="tools shared by every request")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    records = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    shared = set.intersection(*[set(r["tool_ids"]) for r in records])
    menu = len(records[0]["tool_ids"])
    if not 0 <= args.core <= len(shared):
        raise SystemExit(f"--core must be within 0..{len(shared)}")

    corpus = {}
    for line in args.tools.read_text().splitlines():
        if line.strip():
            t = json.loads(line)
            corpus[t["tool_id"]] = t
    payload = {}
    for r in records:
        for tid, tool in zip(r["tool_ids"], r["tools"]):
            payload.setdefault(tid, tool)

    rng = random.Random(args.seed)
    core = sorted(shared)
    rng.shuffle(core)
    keep, drop = core[: args.core], core[args.core :]

    # candidates: corpus tools never used by this workload, bucketed by token size
    used = set(payload) | set(shared)
    cand = [t for tid, t in corpus.items() if tid not in used and t.get("schema_tokens")]
    rng.shuffle(cand)
    by_size = sorted(cand, key=lambda t: t["schema_tokens"])
    taken: set[str] = set()

    def nearest(target: int):
        """Closest unused corpus tool by schema_tokens."""
        lo, hi = 0, len(by_size) - 1
        while lo < hi:                      # binary search for insertion point
            mid = (lo + hi) // 2
            if by_size[mid]["schema_tokens"] < target: lo = mid + 1
            else: hi = mid
        for off in range(len(by_size)):     # expand outwards until one is free
            for j in (lo - off, lo + off):
                if 0 <= j < len(by_size) and by_size[j]["tool_id"] not in taken:
                    taken.add(by_size[j]["tool_id"]); return by_size[j]
        raise SystemExit("corpus exhausted")

    out = []
    for r in records:
        own = [t for t in r["tool_ids"] if t not in shared]
        ids = list(own) + list(keep)
        tools = {t: payload[t] for t in ids}
        for gone in drop:                   # replace each dropped core tool
            repl = nearest(corpus[gone]["schema_tokens"] if gone in corpus
                           else payload[gone].get("schema_tokens", 88))
            ids.append(repl["tool_id"])
            tools[repl["tool_id"]] = json.loads(repl["canonical_json"])
        assert len(ids) == menu, (len(ids), menu)
        order = list(ids)
        random.Random(f"{args.seed}:{r['task_id']}".__hash__() & 0xFFFFFFFF).shuffle(order)
        rec = dict(r)
        rec["tool_ids"] = order
        rec["tools"] = [tools[t] for t in order]
        rec["ordering"] = "original"
        rec["overlap_core"] = args.core
        out.append(rec)

    built = [set(r["tool_ids"]) for r in out]
    common = set.intersection(*built)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{args.output.name}: {len(out)} requests, menu {menu}, "
          f"shared by all {len(common)} ({len(common)/menu*100:.1f}%)")


if __name__ == "__main__":
    main()
