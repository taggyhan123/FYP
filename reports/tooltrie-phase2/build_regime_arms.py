#!/usr/bin/env python3
"""Fill the two missing cells of the algorithm x information-regime 2x2.

Phase 2 compared ToolTrie-v0 (causal: may use only already-served requests)
against ContextPilot (offline: sees the whole evaluation batch). Those are
different information regimes, so the 87.19% vs 96.64% gap conflates "which
algorithm" with "how much it was allowed to see". This builds:

  contextpilot_static_refit_causal
                       - ContextPilot restricted to a causal information regime.
                         For request n we call fit_transform on contexts[0..n]
                         and keep only the ordering it assigns to n. This is a
                         static-refit adaptation, not the official persistent
                         ContextPilot.reorder online API.

  tooltrie_offline     - ToolTrie granted the offline regime. A trie is built by
                         observing EVERY request's ordering, then every request
                         is re-planned against that full trie; iterated to a
                         fixpoint so the ordering is self-consistent. The
                         recency window is disabled, since "recent" is
                         meaningless once the batch is atemporal.

Historical note: the published Phase 2 ``contextpilot_causal`` artifacts were
created from this script with alpha=0.5.  Future runs default to the paper's
alpha=0.001 and receive a distinct ordering label.  Reproduce the historical
condition only with ``--contextpilot-alpha 0.5 --allow-nonstandard-alpha``.

Usage: build_regime_arms.py <dataset> <capacity_tokens> <in.jsonl> <out_prefix>
"""
from __future__ import annotations

import argparse
import importlib
import io
import json
import sys
import contextlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tatm.analysis import load_processed
from tatm.io import read_jsonl, write_jsonl
from tatm.tooltrie import ToolTrie

MAX_ITERS = 6


def build_contextpilot_static_refit(records, ci_module, *, alpha: float):
    ids = sorted({str(t) for r in records for t in r["tool_ids"]})
    m = {t: i for i, t in enumerate(ids)}
    inv = {i: t for t, i in m.items()}
    contexts = [[m[str(t)] for t in r["tool_ids"]] for r in records]
    out = []
    for n in range(len(records)):
        with contextlib.redirect_stdout(io.StringIO()):
            idx = ci_module.ContextIndex(
                linkage_method="average", use_gpu=False, alpha=alpha,
                num_workers=1, batch_size=32,
            )
            res = idx.fit_transform(contexts[: n + 1])
        ordering = [inv[int(x)] for x in res.reordered_contexts[n]]
        assert set(ordering) == set(records[n]["tool_ids"]), f"record {n}: tool set changed"
        out.append(ordering)
        if (n + 1) % 50 == 0:
            print(
                f"    contextpilot_static_refit_causal: {n + 1}/{len(records)}",
                flush=True,
            )
    return out


def build_tooltrie_offline(records, tools, capacity):
    orderings = [list(r["tool_ids"]) for r in records]
    for it in range(MAX_ITERS):
        trie = ToolTrie(
            tools, fallback="alphabetical",
            recency_window=None,           # atemporal batch: nothing "expires"
            capacity_tokens=capacity, max_nodes=100_000,
        )
        for o in orderings:                # observe ALL requests, incl. future
            trie.observe(o)
        new = [list(trie.plan(r["tool_ids"]).ordered_ids) for r in records]
        changed = sum(1 for a, b in zip(orderings, new) if a != b)
        print(f"    tooltrie_offline iter {it + 1}: {changed} orderings changed", flush=True)
        orderings = new
        if changed == 0:
            break
    return orderings


def emit(records, orderings, label, path, *, plan_metadata=None):
    out = []
    for rec, order in zip(records, orderings):
        payload = dict(zip(rec["tool_ids"], rec["tools"], strict=True))
        new = dict(rec)
        new["base_ordering"] = rec.get("ordering")
        new["ordering"] = label
        new["tool_ids"] = list(order)
        new["tools"] = [payload[t] for t in order]
        if plan_metadata is not None:
            new["contextpilot_plan"] = dict(plan_metadata)
        assert set(new["tool_ids"]) == set(rec["tool_ids"])
        assert len(new["tool_ids"]) == len(rec["tool_ids"])
        out.append(new)
    n = write_jsonl(path, out)
    print(f"  wrote {n} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("capacity_tokens", type=int)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_prefix")
    parser.add_argument("--contextpilot-alpha", type=float, default=0.001)
    parser.add_argument("--allow-nonstandard-alpha", action="store_true")
    args = parser.parse_args()
    if (
        not 0.001 <= args.contextpilot_alpha <= 0.01
        and not args.allow_nonstandard_alpha
    ):
        parser.error(
            "--contextpilot-alpha must be in [0.001, 0.01]; use "
            "--allow-nonstandard-alpha only for historical reproduction"
        )

    capacity = args.capacity_tokens
    src = args.input
    prefix = args.output_prefix
    records = list(read_jsonl(src))
    tools, _ = load_processed(PROJECT_ROOT / "data" / "processed")

    print("  building tooltrie_offline...")
    emit(records, build_tooltrie_offline(records, tools, capacity),
         "tooltrie_offline", Path(f"{prefix}-tooltrie_offline.jsonl"))

    print("  building contextpilot_static_refit_causal...")
    ci = importlib.import_module("contextpilot.context_index")
    label = "contextpilot_static_refit_causal"
    emit(
        records,
        build_contextpilot_static_refit(
            records, ci, alpha=args.contextpilot_alpha
        ),
        label,
        Path(f"{prefix}-{label}.jsonl"),
        plan_metadata={
            "mode": "static_refit_causal",
            "information_regime": "causal",
            "official_online_api_used": False,
            "alpha": args.contextpilot_alpha,
            "annotations_enabled": False,
            "eviction_feedback_enabled": False,
        },
    )


if __name__ == "__main__":
    main()
