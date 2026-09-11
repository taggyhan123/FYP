#!/usr/bin/env python
"""Simulate the prefix cache over multi-turn session workloads (CPU only).

Reads the output of `build_session_workload.py events` plus the ordering plans
written by the existing builders into <out-dir>/plans/, renders every round's
exact prompt as vLLM 0.26 would, serves rounds in each concurrency level's
closed-loop order, and replays them through `tatm.prefix_cache_sim` at several
cache capacities. Writes one JSON line per (configuration, capacity).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

BLOCK = 16
CAPACITY = {"0.6B": 11857 * BLOCK, "4B": 6119 * BLOCK, "8B": 2294 * BLOCK, "unlimited": None}
STATIC = ("original", "alphabetical", "frequency")
ADAPTIVE = ("tooltrie_v0", "cp_online", "tooltrie_v1")
_STATE: dict = {}


def _init(out_dir: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from transformers import AutoTokenizer
    from tatm.analysis import load_processed
    out = Path(out_dir)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tools, _ = load_processed(PROJECT_ROOT / "data/processed")
    added = sorted(tok.added_tokens_encoder, key=len, reverse=True)
    _STATE.update(
        out=out, tok=tok, tools=tools,
        split=re.compile("(" + "|".join(re.escape(a) for a in added) + ")"),
        added=tok.added_tokens_encoder, memo={},
        sessions=[json.loads(l) for l in (out / "sessions.jsonl").read_text().splitlines()],
        meta=[json.loads(l) for l in (out / "session_meta.jsonl").read_text().splitlines()],
        support=Counter(json.loads((out / "support.json").read_text())),
    )


def encode(text: str) -> list[int]:
    """Tokenise piecewise between added tokens, memoising repeated pieces.

    Identical to tokenising the whole text: the tokenizer itself splits on added
    tokens and encodes each piece independently. Checked on sampled prompts.
    """
    ids: list[int] = []
    memo, added, tok = _STATE["memo"], _STATE["added"], _STATE["tok"]
    for piece in _STATE["split"].split(text):
        if not piece:
            continue
        if piece in added:
            ids.append(added[piece])
            continue
        cached = memo.get(piece)
        if cached is None:
            cached = tok(piece, add_special_tokens=False)["input_ids"]
            if len(memo) < 200000:
                memo[piece] = cached
        ids.extend(cached)
    return ids


def build_prompts(timeline: dict, ordering: str, plan: dict | None) -> tuple[list, list, list]:
    """Per (session index, round): block hashes, prompt length; plus a check count."""
    from tatm.prefix_cache_sim import block_hashes
    hashes, lengths, checked = {}, {}, 0
    for index, r, ids, was_checked in iter_prompt_ids(timeline, ordering, plan):
        hashes[(index, r)] = block_hashes(ids, BLOCK)
        lengths[(index, r)] = len(ids)
        checked += was_checked
    return hashes, lengths, checked


def iter_prompt_ids(timeline: dict, ordering: str, plan: dict | None):
    """Yield (session index, round, prompt token ids, spot-checked?) for every round."""
    from analyze_prefix_cache import vllm_tool
    from tatm.prompting import openai_tool, order_tool_ids
    tok, tools, support = _STATE["tok"], _STATE["tools"], _STATE["support"]
    for index, (session, meta) in enumerate(zip(_STATE["sessions"], _STATE["meta"])):
        sid, starts = session["session_id"], meta["round_starts"]
        messages = session["messages"]
        latest_plan = None
        for r in range(meta["rounds"]):
            front, appended = timeline[sid][r]
            if ordering in STATIC:
                ordered = list(order_tool_ids(front, tools, support, ordering))
            else:
                latest_plan = plan.get(f"{sid}#{r}", latest_plan)
                ordered = latest_plan
                if ordered is None or sorted(ordered) != sorted(front):
                    raise SystemExit(f"{ordering}: no plan covering {sid} round {r}")
            history = list(messages[:starts[r]])
            for change_round, ids in sorted(appended, key=lambda a: a[0], reverse=True):
                body = "\n".join(json.dumps(vllm_tool(openai_tool(tools[t])), ensure_ascii=False) for t in ids)
                history.insert(starts[change_round], {"role": "user", "content": "New tools are available:\n<tools>\n" + body + "\n</tools>"})
            text = tok.apply_chat_template(
                [{"role": "system", "content": session["system_prompt"]}, *history],
                tools=[vllm_tool(openai_tool(tools[t])) for t in ordered],
                add_generation_prompt=True, tokenize=False, enable_thinking=False)
            ids = encode(text)
            spot = (index * 7 + r) % 97 == 0  # spot-check the piecewise tokenizer
            if spot and tok(text, add_special_tokens=False)["input_ids"] != ids:
                raise SystemExit(f"piecewise tokenisation differs for {sid} round {r}")
            yield index, r, ids, int(spot)


def run_job(job: dict) -> list[dict]:
    from tatm.prefix_cache_sim import simulate_block_cache
    out = _STATE["out"]
    timeline = json.loads((out / "timelines" / f"{job['timeline']}.json").read_text())
    meta = _STATE["meta"]
    rows = []
    built = None
    for concurrency in job["concurrency"]:
        plan = None
        if job["ordering"] in ADAPTIVE:
            path = out / "plans" / f"{job['timeline']}-{job['mode']}-C{concurrency}-{job['ordering']}.jsonl"
            plan = {r["task_id"]: r["tool_ids"] for r in map(json.loads, path.read_text().splitlines())}
            built = build_prompts(timeline, job["ordering"], plan)
        elif built is None:
            built = build_prompts(timeline, job["ordering"], None)
        hashes, lengths, checked = built
        order = [tuple(x) for x in json.loads((out / "events" / f"order-C{concurrency}.json").read_text())]
        order = [(s, r) for s, r in order if r < meta[s]["rounds"]]
        completions = [meta[s]["completion_tokens"][r] for s, r in order]
        for label in job["capacities"]:
            capacity = CAPACITY[label]
            cached, foreign = simulate_block_cache(
                [hashes[x] for x in order], [lengths[x] for x in order], completions,
                None if capacity is None else capacity // BLOCK, BLOCK, owners=[s for s, _ in order])
            by_key = {x: (c, f) for x, c, f in zip(order, cached, foreign)}
            total = sum(lengths[x] for x in order)
            computed = sorted(lengths[x] - by_key[x][0] for x in order)
            later = [(s, r) for s, r in order if r > 0]
            kept = sum(1 for s, r in later
                       if by_key[(s, r)][0] >= ((lengths[(s, r - 1)] - 4) // BLOCK) * BLOCK)
            first = [(s, 0) for s in range(len(meta)) if meta[s]["rounds"] > 0]
            rows.append({
                **{k: job[k] for k in ("timeline", "ordering", "mode")},
                "concurrency": concurrency, "capacity": label, "rounds": len(order),
                "prompt_tokens": total, "cached_tokens": sum(cached),
                "cross_session_cached": sum(foreign),
                "hit_pct": 100 * sum(cached) / total,
                "cross_session_pct": 100 * sum(foreign) / total,
                "computed_per_round_mean": sum(computed) / len(computed),
                "computed_per_round_p95": computed[int(0.95 * (len(computed) - 1))],
                "rounds_keeping_history_pct": 100 * kept / max(len(later), 1),
                "session_start_hit_pct": 100 * sum(by_key[x][0] for x in first) / sum(lengths[x] for x in first),
                "tokenizer_spot_checks": checked,
            })
    return rows


def jobs() -> list[dict]:
    main = ["D1-k64", "D2-front-k64", "D2-append-k64", "D3-front-k64", "D3-append-k64"]
    out = []
    for timeline in main:
        for ordering in STATIC + ADAPTIVE:
            capacities = ["0.6B"]
            if timeline in ("D1-k64", "D2-append-k64") and ordering in ("original", "cp_online", "tooltrie_v1"):
                capacities = ["0.6B", "4B", "8B", "unlimited"]
            out.append({"timeline": timeline, "ordering": ordering, "mode": "onchange",
                        "concurrency": [1, 8, 32], "capacities": capacities})
    for ordering in ("cp_online", "tooltrie_v1"):
        out.append({"timeline": "D1-k64", "ordering": ordering, "mode": "everyround",
                    "concurrency": [1, 8, 32], "capacities": ["0.6B"]})
    for timeline in ("D1-k16", "D2-front-k16", "D2-append-k16"):
        for ordering in ("original", "cp_online", "tooltrie_v1"):
            out.append({"timeline": timeline, "ordering": ordering, "mode": "onchange",
                        "concurrency": [1, 8, 32], "capacities": ["0.6B"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--only", help="run a single job, e.g. D1-k64:original:onchange")
    args = parser.parse_args()
    todo = jobs()
    if args.only:
        todo = [j for j in todo if f"{j['timeline']}:{j['ordering']}:{j['mode']}" == args.only]
    results = args.out_dir / ("results.jsonl" if not args.only else f"results-{args.only.replace(':', '_')}.jsonl")
    if results.exists():
        raise SystemExit(f"refusing to overwrite {results}")
    with Pool(args.workers, initializer=_init, initargs=(str(args.out_dir),)) as pool, results.open("w") as handle:
        for rows in pool.imap_unordered(run_job, todo):
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(rows[0]["timeline"], rows[0]["ordering"], rows[0]["mode"], "done", flush=True)


if __name__ == "__main__":
    main()
