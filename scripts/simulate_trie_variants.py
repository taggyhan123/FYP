#!/usr/bin/env python
"""Simulate candidate ToolTrie-v1 improvements on the existing dense workloads.

CPU only. Re-renders every prompt exactly as vLLM does and replays it through
`tatm.prefix_cache_sim` (validated against vLLM's per-request cached-token
counts in 198-200 of 200 requests per run), at the KV-cache capacity each model
actually had. Reports the prompt-token hit rate per variant, per model cache,
on 64 tools shown and on retrieve-64 / show-10, plus - for show-10 - how often
a gold tool survives in the 10 shown, which is the accuracy ceiling a variant
can reach without a GPU run.

Variants (all causal: each uses only requests already served):

  original        retriever order (control)
  v1              ToolTrie-v1 as shipped (trie capacity 190,896 tokens,
                  recency window 128)
  v1-cap          v1 with the trie's capacity set to the model's real cache, so
                  its "still cached" hint matches the server
  v1-p10obs       show-10 only: the trie records the 10 tools actually served
                  rather than the planned order of all 64
  v1-anchor-m     when the trie matches nothing, lead with up to m tools that
                  have recurred most in the served stream (>= 2 appearances),
                  then retriever order; otherwise identical to v1
  v1-window-W     the dispatcher may reorder arrivals inside windows of W
                  requests: greedy chain by tool-set overlap with the request
                  just served, then v1 plans in that order (the coordination
                  lever; needs a scheduler, not just a planner)
  v1-lpm-W        same window, but the next request is the pending one whose
                  v1 plan reuses the most trie tokens right now (longest-prefix
                  -match scheduling at tool level); W = 200 is the whole stream

Writes <out-dir>/results.json; refuses to overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_prefix_cache import MARKER, encode, render_text, vllm_tool  # noqa: E402
from tatm.analysis import load_processed  # noqa: E402
from tatm.prefix_cache_sim import simulate_prefix_cache  # noqa: E402
from tatm.tooltrie_v1 import RelevancePreservingToolTrie  # noqa: E402

BLOCK = 16
CACHE_TOKENS = {"0.6B": 189_712, "4B": 97_904, "8B": 36_704}  # measured server caches
DENSE = PROJECT_ROOT / "cluster/results/dense-retrieval-20260902-234630/workloads/k64-original.jsonl"
COMPLETIONS = {  # completion tokens per task from the served 4B runs, shared by every variant
    64: PROJECT_ROOT / "cluster/results/dense-accuracy-20260903-002742/replays/4b-k64-original.json",
    10: PROJECT_ROOT / "cluster/results/eval-gaps-20260907-224433/replays/4b-k64p10-original.json",
}


class AnchorTrie(RelevancePreservingToolTrie):
    """v1, but a request the trie cannot match leads with recurring tools."""

    def __init__(self, *args, anchors: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchors = anchors
        self.seen: Counter[str] = Counter()

    def plan(self, tool_ids):
        plan = super().plan(tool_ids)
        if plan.matched_prefix_ids:
            return plan
        incoming = list(plan.ordered_ids)
        hot = sorted((t for t in incoming if self.seen[t] >= 2),
                     key=lambda t: (-self.seen[t], incoming.index(t)))[: self.anchors]
        if not hot:
            return plan
        ordered = tuple(hot + [t for t in incoming if t not in set(hot)])
        return type(plan)(ordered_ids=ordered, matched_prefix_ids=tuple(hot),
                          fallback_ids=ordered[len(hot):], hinted_schema_tokens=0)

    def observe(self, ordered_ids):
        self.seen.update(ordered_ids)
        return super().observe(ordered_ids)


def window_order(records: list[dict], width: int) -> list[int]:
    """Within each window of `width` arrivals, chain requests by tool overlap."""
    order: list[int] = []
    for start in range(0, len(records), width):
        pending = list(range(start, min(start + width, len(records))))
        prev = order[-1] if order else None
        while pending:
            if prev is None:
                nxt = pending[0]
            else:
                pset = set(records[prev]["tool_ids"])
                nxt = max(pending, key=lambda i: (len(pset & set(records[i]["tool_ids"])), -i))
            order.append(nxt)
            pending.remove(nxt)
            prev = nxt
    return order


def run_variant(name, records, tools, present, completions, parts, capacity_default=190_896,
                cap_tokens=None, anchors=0, window=0, observe_presented=False, lpm=0):
    seq = window_order(records, window) if window else list(range(len(records)))
    kw = dict(fallback="alphabetical", recency_window=128, max_nodes=100_000,
              capacity_tokens=cap_tokens or capacity_default)
    planner = AnchorTrie(tools, anchors=anchors, **kw) if anchors else RelevancePreservingToolTrie(tools, **kw)
    head, tool_text, tails = parts
    prompts, comp, shown_gold = [], [], 0
    if lpm:  # pick the next request by the trie's own reuse estimate, within the window
        seq, pending = [], []
        for start in range(0, len(records), lpm):
            pending = list(range(start, min(start + lpm, len(records))))
            while pending:
                nxt = max(pending, key=lambda i: (planner.plan(tuple(records[i]["tool_ids"])).hinted_schema_tokens, -i))
                pending.remove(nxt)
                seq.append(nxt)
                ordered = list(planner.plan(tuple(records[nxt]["tool_ids"])).ordered_ids)
                planner.observe(ordered[:present] if observe_presented else ordered)
        planner = AnchorTrie(tools, anchors=anchors, **kw) if anchors else RelevancePreservingToolTrie(tools, **kw)
    for i in seq:
        rec = records[i]
        ordered = list(rec["tool_ids"]) if name == "original" else list(planner.plan(tuple(rec["tool_ids"])).ordered_ids)
        shown = ordered[:present]
        prompts.append(encode(head + "".join(tool_text[t] for t in shown) + tails[i]))
        comp.append(completions[rec["task_id"]])
        shown_gold += bool(set(rec.get("gold_tool_ids") or []) & set(shown))
        if name != "original":
            planner.observe(shown if observe_presented else ordered)
    total = sum(map(len, prompts))
    hits = {m: 100 * sum(simulate_prefix_cache(prompts, comp, CACHE_TOKENS[m] // BLOCK, BLOCK)) / total
            for m in CACHE_TOKENS}
    return {"variant": name, "present": present, "hit_pct": hits, "prompt_tokens": total,
            "gold_in_shown_pct": 100 * shown_gold / len(records)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.out_dir}")
    tools, _ = load_processed(PROJECT_ROOT / "data/processed")
    records = [json.loads(l) for l in DENSE.read_text().splitlines() if l.strip()]

    # template pieces, exactly as analyze_prefix_cache splits them
    texts = [render_text(r) for r in records]
    head = texts[0].split(MARKER, 1)[0] + MARKER
    tool_text, tails = {}, []
    for rec, text in zip(records, texts):
        pieces = ["\n" + json.dumps(vllm_tool(t), ensure_ascii=False) for t in rec["tools"]]
        rest = text.split(MARKER, 1)[1]
        body = "".join(pieces)
        assert rest.startswith(body), "template layout differs"
        tails.append(rest[len(body):])
        tool_text.update(zip(rec["tool_ids"], pieces))
    parts = (head, tool_text, tails)

    rows = []
    for present in (64, 10):
        rep = json.loads(COMPLETIONS[present].read_text())["results"]
        completions = {r["task_id"]: r["usage"]["completion_tokens"] for r in rep}
        common = dict(records=records, tools=tools, present=present, completions=completions, parts=parts)
        rows.append(run_variant("original", **common))
        rows.append(run_variant("v1", **common))
        for m, cap in CACHE_TOKENS.items():
            r = run_variant(f"v1-cap-{m}", cap_tokens=cap, **common)
            r["hit_pct"] = {m: r["hit_pct"][m]}  # only meaningful at its own capacity
            rows.append(r)
        if present == 10:
            rows.append(run_variant("v1-p10obs", observe_presented=True, **common))
        for a in (1, 2, 3):
            rows.append(run_variant(f"v1-anchor-{a}", anchors=a, **common))
        for w in (4, 8, 16, 32):
            rows.append(run_variant(f"v1-window-{w}", window=w, **common))
        for w in (8, 32):
            rows.append(run_variant(f"v1-window-{w}-anchor-2", window=w, anchors=2, **common))
        for w in (4, 8, 16, 32, 200):
            rows.append(run_variant(f"v1-lpm-{w}", lpm=w, **common))
        if present == 10:
            rows.append(run_variant("v1-lpm-8-p10obs", lpm=8, observe_presented=True, **common))
        for r in rows:
            if r["present"] == present and "printed" not in r:
                r["printed"] = True
                h = r["hit_pct"]
                print(f"show {present:>2} | {r['variant']:<22} | " + " ".join(f"{m}:{h[m]:6.2f}%" if m in h else f"{m}:   -   " for m in CACHE_TOKENS)
                      + (f" | gold in shown {r['gold_in_shown_pct']:.1f}%" if present == 10 else ""))
    args.out_dir.mkdir(parents=True)
    for r in rows:
        r.pop("printed", None)
    (args.out_dir / "results.json").write_text(json.dumps(rows, indent=1))
    print("wrote", args.out_dir / "results.json")


if __name__ == "__main__":
    main()
