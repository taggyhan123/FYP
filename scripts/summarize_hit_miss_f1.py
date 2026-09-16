#!/usr/bin/env python
"""Cache hit/miss rate and F1 for every scored single-turn replay, side by side.

For each serial accuracy replay (whole-menu dense k4/k16/k64/k128 at 0.6B, 4B
and 8B; retrieve-10 and retrieve-64/present-10 on dense and BM25) this reports,
from the same 200 requests:

  hit        prompt tokens served from vLLM's prefix cache / all prompt tokens
  miss       prompt tokens computed / all prompt tokens   (hit + miss = 100)
  tool_hit   hit tokens beyond the first HEADER tokens, i.e. the part of the
             hit that lies in the tool list rather than in the chat-template
             header every request shares whatever the ordering
  req_tool   requests whose hit reaches into the tool list
  ceiling, precision, recall, f1, end_to_end_f1
             exactly as scripts/score_end_to_end.py computes them (it is called
             for each replay, and the per-request recomputation used for the
             paired test is asserted to reproduce its end_to_end_f1)

and a paired comparison of end-to-end F1, ToolTrie-v1 minus each rival, over
the same task ids: mean difference, its standard error and z = diff / se.

Writes one JSON file; refuses to overwrite it.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
import sys
from pathlib import Path

HEADER = 48  # cacheable Qwen3 chat-template tokens before the first tool
RIVALS = ("cp_online", "frequency", "original")
DENSE = "cluster/results/dense-retrieval-20260902-234630/workloads"
GAPS = "cluster/results/eval-gaps-20260907-224433"
BM25 = "cluster/results/eval-bm25trunc-20260908-000327"
MODELS = {"06": "0.6B", "4b": "4B", "8b": "8B"}


def replays():
    """(setting, model, k, arm, replay, workload) for every scored replay."""
    for pattern in ("cluster/results/eval-validity-20260906-165826/replays/*-k*-*.json",
                    "cluster/results/dense-accuracy-20260903-002742/replays/*-k*-*.json"):
        for path in sorted(glob.glob(pattern)):
            name = Path(path).stem
            if name.startswith("sla"):
                continue
            model, k, arm = name.split("-", 2)
            yield "whole menu, dense", model, k, arm, path, f"{DENSE}/{k}-{arm}.jsonl"
    for path in sorted(glob.glob(f"{GAPS}/replays/*.json")):
        name = Path(path).stem
        if name.startswith("lat-"):
            continue
        model, k, arm = name.split("-", 2)
        setting = "retrieve 64 / present 10, dense" if k == "k64p10" else "retrieve 10 / present 10, dense"
        yield setting, model, k, arm, path, f"{GAPS}/workloads/{k}-{arm}.jsonl"
    for path in sorted(glob.glob(f"{BM25}/replays/*.json")):
        model, k, arm = Path(path).stem.split("-", 2)
        yield "retrieve 64 / present 10, BM25", model, k, arm, path, f"{BM25}/workloads/{k}-{arm}.jsonl"


def tool_name(tool: dict) -> str | None:
    fn = tool.get("function")
    return fn.get("name") if isinstance(fn, dict) else tool.get("name")


def per_request_f1(replay: str, workload: str) -> dict[str, float]:
    """End-to-end F1 contribution per task: F1 if a gold tool is in the menu, else 0."""
    meta = {}
    for line in Path(workload).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            meta[r["task_id"]] = ({tool_name(t): tid for t, tid in zip(r["tools"], r["tool_ids"])},
                                  set(r.get("gold_tool_ids") or []), set(r["tool_ids"]))
    out = {}
    for row in json.loads(Path(replay).read_text())["results"]:
        if row["task_id"] not in meta:
            continue
        names, gold, menu = meta[row["task_id"]]
        if not gold or not gold & menu:
            out[row["task_id"]] = 0.0
            continue
        called = {names.get((c.get("function") or {}).get("name"))
                  for c in row.get("tool_calls") or [] if isinstance(c, dict)} - {None}
        tp = len(called & gold)
        p, r = (tp / len(called) if called else 0.0), tp / len(gold)
        out[row["task_id"]] = 2 * p * r / (p + r) if p + r else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    rows, f1_req = [], {}
    for setting, model, k, arm, replay, workload in replays():
        key = (setting, MODELS[model], k, arm)
        if any((r["setting"], r["model"], r["k"], r["arm"]) == key for r in rows):
            raise SystemExit(f"duplicate replay for {key}: {replay}")
        results = json.loads(Path(replay).read_text())["results"]
        cached = [r["measurement"]["cached_tokens"] for r in results]
        computed = [r["measurement"]["computed_prefill_tokens"] for r in results]
        total = sum(cached) + sum(computed)
        score = json.loads(subprocess.check_output(
            [sys.executable, "scripts/score_end_to_end.py", "--replay", replay, "--workload", workload]))
        f1_req[key] = per_request_f1(replay, workload)
        mean = sum(f1_req[key].values()) / len(f1_req[key])
        if abs(mean - score["end_to_end_f1"]) > 6e-4:
            raise SystemExit(f"per-request F1 {mean:.4f} disagrees with scorer {score['end_to_end_f1']} for {replay}")
        rows.append({
            "setting": setting, "model": MODELS[model], "k": k, "arm": arm, "requests": len(results),
            "hit_pct": 100 * sum(cached) / total, "miss_pct": 100 * sum(computed) / total,
            "tool_hit_pct": 100 * sum(max(c - HEADER, 0) for c in cached) / total,
            "requests_any_hit": sum(1 for c in cached if c > 0),
            "requests_tool_hit": sum(1 for c in cached if c > HEADER),
            **{m: score[m] for m in ("ceiling", "gold_hit_ceil", "precision", "recall", "f1", "end_to_end_f1")},
            "replay": replay, "workload": workload,
        })

    paired = []
    for setting, model, k, arm in f1_req:
        if arm != "tooltrie_v1":
            continue
        v1 = f1_req[(setting, model, k, arm)]
        for rival in RIVALS:
            other = f1_req.get((setting, model, k, rival))
            if other is None:
                continue
            diffs = [v1[t] - other[t] for t in sorted(set(v1) & set(other))]
            n = len(diffs)
            mu = sum(diffs) / n
            se = math.sqrt(sum((d - mu) ** 2 for d in diffs) / (n - 1)) / math.sqrt(n)
            paired.append({"setting": setting, "model": model, "k": k, "rival": rival, "tasks": n,
                           "diff_points": 100 * mu, "se_points": 100 * se, "z": mu / se if se else 0.0})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"header_tokens": HEADER, "rows": rows, "paired_end_to_end_f1": paired}, indent=1))
    print(f"{args.output}: {len(rows)} replays, {len(paired)} paired comparisons")


if __name__ == "__main__":
    main()
